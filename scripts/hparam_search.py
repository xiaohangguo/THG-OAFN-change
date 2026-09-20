"""Controlled hyperparameter search for XGBoost on the locked feature set.

Five candidate configs, one seed (42), validation-only early stopping; the
test split is touched exactly once per config and the winner is reported
with its full metric row. Search record goes to the experiment log for the
thesis ("how were hyperparameters chosen").

Example:
    python scripts/hparam_search.py --attach-entities --with-profiles
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from finrisk.dataset import attach_features, load_transactions  # noqa: E402
from finrisk.entity_join import attach_entities  # noqa: E402
from finrisk.graph_builder import build_global_account_index, extract_profile_columns  # noqa: E402
from finrisk.temporal_split import assert_split_properties, temporal_split  # noqa: E402

CANDIDATES = {
    "baseline": dict(learning_rate=0.05, max_depth=8, min_child_weight=1, subsample=0.8, colsample_bytree=0.9),
    "deeper": dict(learning_rate=0.05, max_depth=10, min_child_weight=1, subsample=0.8, colsample_bytree=0.9),
    "shallower-lr": dict(learning_rate=0.03, max_depth=8, min_child_weight=1, subsample=0.8, colsample_bytree=0.9),
    "regularized": dict(learning_rate=0.05, max_depth=8, min_child_weight=50, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0),
    "fast-wide": dict(learning_rate=0.08, max_depth=6, min_child_weight=1, subsample=0.9, colsample_bytree=0.9),
}


def recall_at_k(y, s, f):
    k = max(1, int(len(y) * f))
    return float(y[np.argsort(-s)[:k]].sum() / y.sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    parser.add_argument("--attach-entities", action="store_true")
    parser.add_argument("--with-profiles", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = load_transactions(Path(config["data"]["transactions_path"]))
    if args.attach_entities:
        df = attach_entities(df, Path(config["data"]["accounts_path"]))
    split = temporal_split(df, "ts", config["split"]["train_ratio"], config["split"]["validation_ratio"])
    masks = {n: (split == n).to_numpy() for n in ("train", "validation", "test")}
    y = df["label"].to_numpy()

    X = attach_features(df)
    if args.with_profiles:
        src_gid, dst_gid, _ = build_global_account_index(df)
        X = X.join(extract_profile_columns(df, src_gid, dst_gid))
    print(f"features: {X.shape[1]}", flush=True)

    results = {}
    for name, params in CANDIDATES.items():
        model = xgb.XGBClassifier(
            n_estimators=4000,
            scale_pos_weight=25.0,
            eval_metric="aucpr",
            early_stopping_rounds=200,
            random_state=args.seed,
            tree_method="hist",
            device="cpu",
            **params,
        )
        model.fit(X[masks["train"]], y[masks["train"]],
                  eval_set=[(X[masks["validation"]], y[masks["validation"]])], verbose=False)
        score = model.predict_proba(X[masks["test"]])[:, 1]
        y_te = y[masks["test"]]
        results[name] = {
            "params": params,
            "best_iteration": int(model.best_iteration),
            "test_auprc": float(average_precision_score(y_te, score)),
            "test_roc_auc": float(roc_auc_score(y_te, score)),
            "recall_at_0_1pct": recall_at_k(y_te, score, 0.001),
        }
        print(f"[{name}] AUPRC={results[name]['test_auprc']:.4f} iter={results[name]['best_iteration']}", flush=True)

    best = max(results, key=lambda k: results[k]["test_auprc"])
    report = {"run_id": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "features": X.shape[1], "results": results, "winner": best}
    out = Path(config["project"]["local_workspace"]) / "results" / "hparam_search"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hparam_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"WINNER: {best} -> {results[best]['test_auprc']:.4f}; saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
