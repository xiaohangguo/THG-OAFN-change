"""Route B v2: behavior-group counterfactuals + accuracy-attributability frontier.

Upgrades over v1:
  1. Driver selection per case via local TreeSHAP (not value percentile).
  2. Interventions replace feature values with the TEST-population median
     ("return to ordinary behaviour") instead of zero (trees route zeros
     through default branches, which is not a real counterfactual).
  3. Behavior-GROUP interventions: 78 columns organized into 8 financial
     behaviour families; zeroing a whole family measures pattern-level
     attribution ("what if the fan-out pattern didn't exist").
  4. Frontier: a family of increasingly sparse XGBoost models (78 / top-30 /
     top-15 / top-8 features) traced as AUPRC vs counterfactual sensitivity
     — the empirical accuracy-vs-attributability trade-off.

Example:
    python scripts/counterfactual_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk.dataset import attach_features, load_transactions  # noqa: E402
from finrisk.entity_join import attach_entities  # noqa: E402
from finrisk.graph_builder import build_global_account_index, extract_profile_columns  # noqa: E402
from finrisk.temporal_split import temporal_split  # noqa: E402

BEHAVIOR_GROUPS = {
    "pair_frequency": ["pair"],
    "burst_activity": ["src_w1h", "dst_w1h", "pair_w1h", "src_bro_w1h", "dst_bro_w1h"],
    "daily_volume": ["src_w24h", "dst_w24h", "src_bro_w24h", "dst_bro_w24h"],
    "weekly_volume": ["src_w7d", "dst_w7d", "src_bro_w7d", "dst_bro_w7d"],
    "entity_diff": ["ent_"],
    "src_profile": ["prof_src"],
    "dst_profile": ["prof_dst"],
    "amount": ["amount_paid", "src_w1h_sum", "src_w24h_sum", "src_w7d_sum",
               "dst_w1h_sum", "dst_w24h_sum", "dst_w7d_sum"],
}


def group_columns(feature_cols: list[str], needle: str) -> list[int]:
    return [j for j, c in enumerate(feature_cols) if needle in c]


def train_champion(Xn, y, masks, seed):
    model = xgb.XGBClassifier(
        n_estimators=4000, learning_rate=0.05, max_depth=8, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25.0,
        eval_metric="aucpr", early_stopping_rounds=200, random_state=seed,
        tree_method="hist", device="cpu",
    )
    model.fit(Xn[masks["train"]], y[masks["train"]],
              eval_set=[(Xn[masks["validation"]], y[masks["validation"]])], verbose=False)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cases", type=int, default=20)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = load_transactions(Path(config["data"]["transactions_path"]))
    df = attach_entities(df, Path(config["data"]["accounts_path"]))
    split = temporal_split(df, "ts", config["split"]["train_ratio"], config["split"]["validation_ratio"])
    masks = {n: (split == n).to_numpy() for n in ("train", "validation", "test")}
    y = df["label"].to_numpy()

    X = attach_features(df)
    src_gid, dst_gid, _ = build_global_account_index(df)
    X = X.join(extract_profile_columns(df, src_gid, dst_gid))
    feature_cols = list(X.columns)
    Xn = X.to_numpy(dtype=np.float64)
    medians = np.median(Xn[masks["test"]], axis=0)
    print(f"features: {len(feature_cols)}", flush=True)

    # ---- Champion + local SHAP driver ranking --------------------------------
    model = train_champion(Xn, y, masks, args.seed)
    score = model.predict_proba(Xn)[:, 1]
    te = masks["test"]
    print(f"champion AUPRC {average_precision_score(y[te], score[te]):.4f}", flush=True)

    X_te = pd.DataFrame(Xn[te], columns=feature_cols)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    ranking = np.argsort(-mean_abs)

    # ---- Case-level: group interventions with median replacement -------------
    order = np.argsort(-score[te])
    flagged_tp = [i for i in order if y[te][i] == 1][: args.n_cases]
    cases = []
    group_drop = {g: [] for g in BEHAVIOR_GROUPS}
    for i in flagged_tp:
        row = X_te.iloc[i].copy()
        base = float(score[te][i])
        case = {"row": int(np.flatnonzero(te)[i]), "base_score": base, "groups": {}}
        for gname, needles in BEHAVIOR_GROUPS.items():
            cols = sorted({j for n in needles for j in group_columns(feature_cols, n)})
            if not cols:
                continue
            row2 = row.copy()
            row2.iloc[cols] = medians[cols]
            after = float(model.predict_proba(pd.DataFrame([row2], columns=feature_cols))[0, 1])
            case["groups"][gname] = {"n_cols": len(cols), "score_after": after,
                                     "drop": base - after}
            group_drop[gname].append(base - after)
        # top-5 local SHAP drivers for reference
        top_local = np.argsort(-np.abs(sv[i]))[:5]
        case["local_shap_top5"] = [(feature_cols[j], float(sv[i, j])) for j in top_local]
        cases.append(case)

    group_summary = {
        g: {"mean_drop": float(np.mean(v)), "max_drop": float(np.max(v)), "n": len(v)}
        for g, v in group_drop.items() if v
    }

    # ---- Frontier: sparse models (accuracy vs attributability) ----------------
    frontier = []
    subsets = [len(feature_cols), 30, 15, 8]
    for k in subsets:
        cols = ranking[:k]
        Xk = Xn[:, cols]
        mk = train_champion(Xk, y, masks, args.seed)
        sk = mk.predict_proba(Xk)[:, 1]
        auprc = float(average_precision_score(y[te], sk[te]))
        # sensitivity: mean group-drop for top pair_frequency group on the
        # same flagged cases (median replacement within the subset)
        med_k = np.median(Xk[te], axis=0)
        drops = []
        pair_local = [j for j, c in enumerate([feature_cols[c] for c in cols]) if "pair" in c]
        for i in flagged_tp:
            row2 = Xk[te][i].copy()
            if pair_local:
                row2[pair_local] = med_k[pair_local]
                after = float(mk.predict_proba(pd.DataFrame([row2], columns=[feature_cols[c] for c in cols]).values)[0, 1])
                drops.append(float(sk[te][i]) - after)
        frontier.append({
            "n_features": k,
            "test_auprc": auprc,
            "pair_group_mean_drop": float(np.mean(drops)) if drops else None,
        })
        print(f"[frontier] top-{k}: AUPRC {auprc:.4f}, pair-drop {np.mean(drops) if drops else float('nan'):.4f}", flush=True)

    report = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "group_interventions_summary": group_summary,
        "accuracy_attributability_frontier": frontier,
        "intervention": "test-population median replacement (return to ordinary behaviour)",
        "cases": cases,
    }
    out = Path(config["project"]["local_workspace"]) / "results" / "routeB_counterfactual"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"counterfactual_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True)
    print(f"saved -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
