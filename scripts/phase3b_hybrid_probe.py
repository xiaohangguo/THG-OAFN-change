"""Hybrid probe: does the snapshot graph carry signal BEYOND the 60 causal features?

Feeds the per-band account profiles (the GNN's exact input) to XGBoost as
extra columns. Three outcomes, all informative:
  - AUPRC > 0.489: window profiles add signal -> graph input side has headroom
  - AUPRC ~ 0.489: profiles are already covered by handcrafted features
  - AUPRC < 0.489: profiles are noise for the tree
Same protocol as phase2: seed 42, chronological split, val-only early stop.

Example:
    python scripts/phase3b_hybrid_probe.py --attach-entities
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
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from finrisk.dataset import attach_features, load_transactions  # noqa: E402
from finrisk.entity_join import attach_entities  # noqa: E402
from finrisk.graph_builder import build_banded_snapshots, build_global_account_index  # noqa: E402
from finrisk.temporal_split import assert_split_properties, temporal_split  # noqa: E402

DEVICE = "cuda" if xgb.build_info().get("USE_CUDA") else "cpu"


def _to_local(node_ids: np.ndarray, gids: np.ndarray) -> np.ndarray:
    if len(node_ids) == 0:
        return np.full(len(gids), -1, dtype=np.int64)
    pos = np.searchsorted(node_ids, gids)
    pos_c = np.minimum(pos, len(node_ids) - 1)
    ok = (pos < len(node_ids)) & (node_ids[pos_c] == gids)
    return np.where(ok, pos, -1).astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph-profile probe for XGBoost.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    parser.add_argument("--attach-entities", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--band-hours", type=int, default=12)
    parser.add_argument("--window-days", type=float, default=7.0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = load_transactions(Path(config["data"]["transactions_path"]))
    if args.attach_entities:
        df = attach_entities(df, Path(config["data"]["accounts_path"]))

    split = temporal_split(df, "ts", config["split"]["train_ratio"], config["split"]["validation_ratio"])
    assert_split_properties(df, "ts", split)
    masks = {name: (split == name).to_numpy() for name in ("train", "validation", "test")}
    y = df["label"].to_numpy()

    X = attach_features(df)
    print(f"base txn features: {X.shape[1]}", flush=True)

    print("building snapshots for profile columns...", flush=True)
    src_gid, dst_gid, _ = build_global_account_index(df)
    snapshots = build_banded_snapshots(
        df, src_gid, dst_gid,
        band=pd.Timedelta(hours=args.band_hours),
        window=pd.Timedelta(days=args.window_days),
    )

    # Per-transaction profile columns: src profile (8) + dst profile (8) + seen flags (2).
    n = len(df)
    prof = np.zeros((n, 18), dtype=np.float32)
    for snap in snapshots:
        l_src = _to_local(snap.node_ids, snap.txn_src)
        l_dst = _to_local(snap.node_ids, snap.txn_dst)
        rows = snap.txn_rows
        for li, side in ((l_src, 0), (l_dst, 1)):
            hit = li >= 0
            prof[rows[hit], side * 8 : side * 8 + 8] = snap.node_feat[li[hit]]
        prof[rows, 16] = (l_src >= 0).astype(np.float32)
        prof[rows, 17] = (l_dst >= 0).astype(np.float32)

    prof_cols = [f"prof_src_{i}" for i in range(8)] + [f"prof_dst_{i}" for i in range(8)] + ["prof_src_seen", "prof_dst_seen"]

    out_dir = Path(config["project"]["local_workspace"]) / "results" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    results = {}
    for tag, mat in (("xgb_base", X.to_numpy(dtype=np.float64)),
                     ("xgb_plus_profiles", np.hstack([X.to_numpy(dtype=np.float64), prof.astype(np.float64)]))):
        model = xgb.XGBClassifier(
            n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8,
            colsample_bytree=0.9, scale_pos_weight=25.0, eval_metric="aucpr",
            early_stopping_rounds=200, random_state=args.seed, device=DEVICE, tree_method="hist",
        )
        model.fit(mat[masks["train"]], y[masks["train"]],
                  eval_set=[(mat[masks["validation"]], y[masks["validation"]])], verbose=False)
        score = model.predict_proba(mat[masks["test"]])[:, 1]
        auprc = average_precision_score(y[masks["test"]], score)
        results[tag] = {"test_auprc": float(auprc), "best_iteration": int(model.best_iteration)}
        print(f"[{tag}] test AUPRC={auprc:.4f} (best_iter={model.best_iteration})", flush=True)

    delta = results["xgb_plus_profiles"]["test_auprc"] - results["xgb_base"]["test_auprc"]
    results["profile_delta"] = float(delta)
    results["verdict"] = (
        "profiles add signal beyond handcrafted features" if delta > 0.005
        else "profiles add nothing beyond handcrafted features" if delta > -0.005
        else "profiles are noise for the tree"
    )
    path = out_dir / f"hybrid_probe_{run_id}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
