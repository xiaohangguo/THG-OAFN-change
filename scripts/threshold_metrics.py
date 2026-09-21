"""Threshold-locked F1/Precision/Recall per the experiment protocol.

The protocol requires: threshold selected on VALIDATION (max F1), locked,
then reported once on test. Predictions parquets only stored test scores, so
this reruns the champion (5 seeds) and emits the threshold metrics.

Example:
    python scripts/threshold_metrics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk.dataset import attach_features, load_transactions  # noqa: E402
from finrisk.entity_join import attach_entities  # noqa: E402
from finrisk.graph_builder import build_global_account_index, extract_profile_columns  # noqa: E402
from finrisk.temporal_split import temporal_split  # noqa: E402


def best_f1_threshold(y_true, score):
    order = np.argsort(-score)
    tp = np.cumsum(y_true[order])
    fp = np.cumsum(1 - y_true[order])
    fn = y_true.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    i = int(np.argmax(f1))
    thr = 0.5 * (score[order][i] + (score[order][i + 1] if i + 1 < len(score) else 1.0))
    return float(thr), float(f1[i])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
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
    Xn = X.to_numpy(dtype=np.float64)

    rows = []
    for seed in config["project"]["seed_list"]:
        model = xgb.XGBClassifier(
            n_estimators=4000, learning_rate=0.05, max_depth=8, min_child_weight=50,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25.0,
            eval_metric="aucpr", early_stopping_rounds=200, random_state=seed,
            tree_method="hist", device="cpu",
        )
        model.fit(Xn[masks["train"]], y[masks["train"]],
                  eval_set=[(Xn[masks["validation"]], y[masks["validation"]])], verbose=False)
        s_va = model.predict_proba(Xn[masks["validation"]])[:, 1]
        s_te = model.predict_proba(Xn[masks["test"]])[:, 1]
        thr, va_f1 = best_f1_threshold(y[masks["validation"]], s_va)
        pred = (s_te >= thr).astype(int)
        rows.append({
            "seed": seed, "threshold": thr, "val_f1_at_threshold": va_f1,
            "test_f1": float(f1_score(y[masks["test"]], pred)),
            "test_precision": float(precision_score(y[masks["test"]], pred)),
            "test_recall": float(recall_score(y[masks["test"]], pred)),
            "test_auprc": float(average_precision_score(y[masks["test"]], s_te)),
        })
        print(f"seed {seed}: thr {thr:.4f} F1 {rows[-1]['test_f1']:.4f} "
              f"P {rows[-1]['test_precision']:.4f} R {rows[-1]['test_recall']:.4f}", flush=True)

    summary = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "protocol": "threshold locked on validation (max-F1), reported once on test",
        "per_seed": rows,
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in ("test_f1", "test_precision", "test_recall", "test_auprc")},
    }
    out = Path(config["project"]["local_workspace"]) / "results" / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"threshold_metrics_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["mean"], indent=2), flush=True)
    print(f"saved -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
