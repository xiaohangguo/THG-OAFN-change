"""Leakage forensics: quantifying what each leaked protocol choice is worth.

Motivated by the original THG-OAFN repo (wei4zheng/THG-OAFN), whose training
loss consumes ALL node labels (train_mask is accepted but never used), whose
GraphSMOTE oversamples on all labels, whose scaler fits the full data before
splitting, and whose message passing runs on the whole graph.

The 2x2 matrix on OUR data and OUR model measures:
  split   = temporal (ours) vs random (common in literature)
  labels  = train-only (ours) vs all-labels fed to training (their L1)

Example:
    python scripts/leakage_forensics.py
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk.dataset import attach_features, load_transactions  # noqa: E402
from finrisk.entity_join import attach_entities  # noqa: E402
from finrisk.graph_builder import build_global_account_index, extract_profile_columns  # noqa: E402
from finrisk.temporal_split import temporal_split  # noqa: E402


def run_xgb(X_tr, y_tr, X_va, y_va, X_te, y_te, seed=42) -> float:
    model = xgb.XGBClassifier(
        n_estimators=4000, learning_rate=0.05, max_depth=8, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25.0,
        eval_metric="aucpr", early_stopping_rounds=200, random_state=seed,
        tree_method="hist", device="cpu",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    score = model.predict_proba(X_te)[:, 1]
    return float(average_precision_score(y_te, score))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = load_transactions(Path(config["data"]["transactions_path"]))
    df = attach_entities(df, Path(config["data"]["accounts_path"]))
    y = df["label"].to_numpy()

    X = attach_features(df)
    src_gid, dst_gid, _ = build_global_account_index(df)
    X = X.join(extract_profile_columns(df, src_gid, dst_gid))
    Xn = X.to_numpy(dtype=np.float64)

    # ---- Split A: ours (temporal) ----
    split_t = temporal_split(df, "ts").to_numpy()
    tr_t, va_t, te_t = split_t == "train", split_t == "validation", split_t == "test"

    # ---- Split B: random (literature-common) ----
    idx = np.arange(len(df))
    tr_r, tmp = train_test_split(idx, train_size=0.6, random_state=42, stratify=y)
    va_r, te_r = train_test_split(tmp, train_size=0.5, random_state=42, stratify=y[tmp])
    tr_m, va_m, te_m = np.zeros(len(df), bool), np.zeros(len(df), bool), np.zeros(len(df), bool)
    tr_m[tr_r], va_m[va_r], te_m[te_r] = True, True, True

    results = {}
    for split_name, tr, va, te in (("temporal", tr_t, va_t, te_t), ("random", tr_m, va_m, te_m)):
        clean = run_xgb(Xn[tr], y[tr], Xn[va], y[va], Xn[te], y[te])
        # Leaked variant: test rows (features AND labels) join training, exactly
        # what an unmasked full-graph CE loss does in the original repo.
        tr_leak = tr | te
        leaked = run_xgb(Xn[tr_leak], y[tr_leak], Xn[va], y[va], Xn[te], y[te])
        results[split_name] = {"clean": clean, "leaked_labels": leaked, "leak_gain": leaked - clean}
        print(f"[{split_name}] clean={clean:.4f}  leaked={leaked:.4f}  (+{leaked-clean:.4f})", flush=True)

    report = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "model": "xgboost-regularized-78col",
        "matrix": results,
        "notes": (
            "leaked_labels = test features+labels appended to training, replicating "
            "the original THG-OAFN unmasked full-graph cross-entropy (thg_oafn.py:152, "
            "train_mask accepted but unused); random split replicates "
            "literature-common non-temporal evaluation."
        ),
    }
    out = Path(config["project"]["local_workspace"]) / "results" / "leakage_forensics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"forensics_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["matrix"], indent=2), flush=True)
    print(f"saved -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
