"""Attribution report for the strongest table model (XGBoost) via SHAP.

Answers the thesis question "why was this transaction flagged" with
verifiable evidence instead of decorative plots:

1. Global: mean |SHAP| feature ranking on the test split.
2. Case level: per-transaction SHAP decompositions for the highest-risk
   alerts (true positives first).
3. Fidelity check: masking the top-k SHAP features must collapse the score
   of flagged transactions; if it does not, the attribution is decorative.

Run after phase2_baseline. Model is retrained on the fixed temporal split
(seed 42) with the identical protocol; nothing about the split is re-chosen.

Example:
    python scripts/attribution_report.py --attach-entities
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk.dataset import attach_features, load_transactions  # noqa: E402
from finrisk.entity_join import attach_entities  # noqa: E402
from finrisk.temporal_split import assert_split_properties, temporal_split  # noqa: E402

DEVICE = "cuda" if xgb.build_info().get("USE_CUDA") else "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="SHAP attribution for XGBoost risk model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    parser.add_argument("--attach-entities", action="store_true")
    parser.add_argument("--with-profiles", action="store_true",
                        help="append 18 snapshot profile columns before attribution")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--fidelity-k", type=int, default=10)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tx_path = Path(config["data"]["transactions_path"])
    df = load_transactions(tx_path)
    if args.attach_entities:
        df = attach_entities(df, Path(config["data"]["accounts_path"]))

    split = temporal_split(df, "ts", config["split"]["train_ratio"], config["split"]["validation_ratio"])
    assert_split_properties(df, "ts", split)
    masks = {name: (split == name).to_numpy() for name in ("train", "validation", "test")}
    y = df["label"].to_numpy()

    X = attach_features(df)
    feature_cols = list(X.columns)
    if args.with_profiles:
        from finrisk.graph_builder import build_global_account_index, extract_profile_columns

        src_gid, dst_gid, _ = build_global_account_index(df)
        X = X.join(extract_profile_columns(df, src_gid, dst_gid))
        feature_cols = list(X.columns)
        print(f"attribution feature set: {len(feature_cols)} cols", flush=True)

    model = xgb.XGBClassifier(
        n_estimators=4000, learning_rate=0.05, max_depth=8, min_child_weight=50, subsample=0.8,
        colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25.0, eval_metric="aucpr",
        early_stopping_rounds=200, random_state=args.seed, device=DEVICE, tree_method="hist",
    )
    model.fit(X[masks["train"]], y[masks["train"]], eval_set=[(X[masks["validation"]], y[masks["validation"]])], verbose=False)
    score = model.predict_proba(X[masks["test"]])[:, 1]
    y_te = y[masks["test"]]

    out_dir = Path(config["project"]["local_workspace"]) / "results" / "attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ---- Global SHAP on the test split ------------------------------------
    X_te = X[masks["test"]].reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    if isinstance(sv, list):  # binary xgboost may return [class0, class1]
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    ranking = sorted(
        [{"feature": c, "mean_abs_shap": float(v)} for c, v in zip(feature_cols, mean_abs)],
        key=lambda d: -d["mean_abs_shap"],
    )

    # ---- Case-level attribution on top alerts (TPs first) ------------------
    order = np.argsort(-score)
    flagged = order[: max(args.n_cases * 20, 1000)]  # search pool of top alerts
    tp_idx = flagged[y_te[flagged] == 1][: args.n_cases]
    fp_idx = flagged[y_te[flagged] == 0][: args.n_cases]
    cases = []
    for tag, idxs in (("true_positive", tp_idx), ("false_positive", fp_idx)):
        for i in idxs:
            contributions = sorted(
                [
                    {"feature": c, "shap": float(sv[i, j]),
                     "value": float(X_te.iloc[i, j])}
                    for j, c in enumerate(feature_cols)
                ],
                key=lambda d: -abs(d["shap"]),
            )[:8]
            cases.append({
                "kind": tag, "row": int(i), "score": float(score[i]),
                "label": int(y_te[i]), "top_contributors": contributions,
            })

    # ---- Fidelity: mask top-k features -> score must collapse -------------
    k = args.fidelity_k
    top_feats = [r["feature"] for r in ranking[:k]]
    top_cols = [feature_cols.index(f) for f in top_feats]
    X_masked = X_te.to_numpy().copy()
    X_masked[:, top_cols] = 0.0
    score_masked = model.predict_proba(pd.DataFrame(X_masked, columns=feature_cols))[:, 1]
    flagged_mask = score >= np.quantile(score, 0.999)
    fidelity = {
        "masked_features": top_feats,
        "flagged_score_before": float(score[flagged_mask].mean()),
        "flagged_score_after": float(score_masked[flagged_mask].mean()),
        "score_drop_ratio": float(1 - score_masked[flagged_mask].mean() / max(score[flagged_mask].mean(), 1e-9)),
        "mean_abs_shap_share_of_top_k": float(sum(r["mean_abs_shap"] for r in ranking[:k]) / sum(r["mean_abs_shap"] for r in ranking)),
    }

    report = {
        "run_id": run_id,
        "model": "xgboost",
        "seed": args.seed,
        "n_features": len(feature_cols),
        "global_ranking_top20": ranking[:20],
        "fidelity": fidelity,
        "cases": cases,
    }
    path = out_dir / f"xgb_shap_{run_id}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ranking_top10": ranking[:10], "fidelity": fidelity}, indent=2, ensure_ascii=False))
    print(f"cases saved: {len(cases)} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
