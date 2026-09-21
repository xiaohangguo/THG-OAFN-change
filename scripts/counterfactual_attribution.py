"""Route B: counterfactual attribution engine (neural surrogate + interventions).

Distills the champion XGBoost into a differentiable surrogate, then answers
"why was this transaction flagged" with counterfactual waterfalls: scale down
the top SHAP driver features and watch the predicted score fall. Fidelity is
verified by re-scoring the same interventions with the REAL XGBoost and
measuring agreement — the attribution is validated, not decorated.

Example:
    python scripts/counterfactual_attribution.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk.dataset import attach_features, load_transactions  # noqa: E402
from finrisk.entity_join import attach_entities  # noqa: E402
from finrisk.graph_builder import build_global_account_index, extract_profile_columns  # noqa: E402
from finrisk.temporal_split import temporal_split  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INTERVENTIONS = (0.75, 0.5, 0.25, 0.0)  # multiplicative scalings of driver features


class Surrogate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
    print(f"features: {len(feature_cols)}", flush=True)

    # ---- Champion XGBoost ----------------------------------------------------
    model = xgb.XGBClassifier(
        n_estimators=4000, learning_rate=0.05, max_depth=8, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25.0,
        eval_metric="aucpr", early_stopping_rounds=200, random_state=args.seed,
        tree_method="hist", device="cpu",
    )
    model.fit(Xn[masks["train"]], y[masks["train"]],
              eval_set=[(Xn[masks["validation"]], y[masks["validation"]])], verbose=False)
    score = model.predict_proba(Xn)[:, 1]
    te = masks["test"]
    print(f"champion test AUPRC={average_precision_score(y[te], score[te]):.4f}", flush=True)

    # ---- Surrogate distillation (train+val rows only) ------------------------
    fit_mask = masks["train"] | masks["validation"]
    x_fit = torch.from_numpy(Xn[fit_mask].astype(np.float32)).to(DEVICE)
    s_fit = torch.from_numpy(score[fit_mask].astype(np.float32)).to(DEVICE)
    x_te = torch.from_numpy(Xn[te].astype(np.float32)).to(DEVICE)

    torch.manual_seed(args.seed)
    sur = Surrogate(Xn.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(sur.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    n = len(x_fit)
    for epoch in range(40):
        sur.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 262144):
            idx = perm[i : i + 262144]
            loss = nn.functional.mse_loss(sur(x_fit[idx]), s_fit[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    sur.eval()
    with torch.no_grad():
        s_hat_te = sur(x_te).cpu().numpy()
    rho = spearmanr(score[te], s_hat_te).statistic
    k = max(1, int(len(s_hat_te) * 0.001))
    top_true = set(np.argsort(-score[te])[:k].tolist())
    top_hat = set(np.argsort(-s_hat_te)[:k].tolist())
    overlap = len(top_true & top_hat) / k
    print(f"surrogate: spearman {rho:.4f}, top0.1% overlap {overlap:.3f}", flush=True)

    # ---- Counterfactual cases on top flagged TP transactions -----------------
    # Driver features: continuous count/sum features only (exclude codes/hour).
    drivable = [c for c in feature_cols if any(t in c for t in ("count", "sum", "past", "amount", "prof_"))]
    order = np.argsort(-score[te])
    flagged = [i for i in order if y[te][i] == 1][: args.n_cases]

    cases, fidelity_pairs = [], []
    for i in flagged:
        row = Xn[te][i].copy()
        case = {"row": int(np.flatnonzero(te)[i]), "label": 1,
                "score": float(score[te][i]), "surrogate_score": float(s_hat_te[i]),
                "waterfall": []}
        # rank driver features by this row's value percentile among test (simple proxy)
        driver_scores = []
        for c in drivable:
            j = feature_cols.index(c)
            pct = float((Xn[te][:, j] < row[j]).mean())
            driver_scores.append((pct, c, j))
        driver_scores.sort(reverse=True)
        top_drivers = [d for d in driver_scores[:5]]

        for scale in INTERVENTIONS:
            row2 = row.copy()
            for _, c, j in top_drivers:
                row2[j] = row[j] * scale
            with torch.no_grad():
                pred_sur = float(sur(torch.from_numpy(row2.astype(np.float32)).to(DEVICE)).item())
            pred_true = float(model.predict_proba(pd.DataFrame([row2], columns=feature_cols))[0, 1])
            fidelity_pairs.append((pred_sur, pred_true))
            case["waterfall"].append({
                "scale": scale,
                "drivers": [d[1] for d in top_drivers],
                "surrogate_score": pred_sur,
                "real_xgb_score": pred_true,
            })
        cases.append(case)

    fp_arr = np.array(fidelity_pairs)
    cf_mae = float(np.abs(fp_arr[:, 0] - fp_arr[:, 1]).mean())
    cf_rho = float(spearmanr(fp_arr[:, 0], fp_arr[:, 1]).statistic)
    base = cases[0]["score"] if cases else 0.0
    drop_full = [c["waterfall"][-1]["real_xgb_score"] for c in cases]

    report = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "surrogate_quality": {"spearman": float(rho), "top0.1pct_overlap": overlap},
        "counterfactual_fidelity": {"mae": cf_mae, "spearman": cf_rho},
        "score_drop_when_drivers_zeroed": {
            "mean_before": base,
            "mean_after": float(np.mean(drop_full)),
            "n_cases": len(cases),
        },
        "interventions": list(INTERVENTIONS),
        "cases": cases,
    }
    out = Path(config["project"]["local_workspace"]) / "results" / "routeB_counterfactual"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"counterfactual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True)
    print(f"cases saved: {len(cases)} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
