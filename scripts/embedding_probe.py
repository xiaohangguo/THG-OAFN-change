"""Embedding probe: do trained GNN embeddings add signal beyond the 78 features?

Trains the strongest GNN form inline (gatv2 + strong profiles + warmup),
extracts per-transaction frozen embeddings (src 64 + dst 64 = 128 dims), and
runs the XGBoost A/B: 78 cols vs 78+128. Trees ignore useless columns, so
the floor is parity; any gain means the embeddings carry signal.

Example:
    python scripts/embedding_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from finrisk.dataset import attach_features, load_transactions  # noqa: E402
from finrisk.entity_join import attach_entities  # noqa: E402
from finrisk.graph_builder import build_banded_snapshots, build_global_account_index, extract_profile_columns  # noqa: E402
from finrisk.temporal_split import assert_split_properties, temporal_split  # noqa: E402
from phase3_graph_baseline import DEVICE, BandData, GraphEncoder, TxnScorer, forward_band  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
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
    print(f"tabular features: {X.shape[1]}", flush=True)

    scaler = StandardScaler().fit(Xn[masks["train"]])
    Xs = np.nan_to_num(scaler.transform(Xn), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_gpu = torch.from_numpy(Xs).to(DEVICE)

    snapshots = build_banded_snapshots(df, src_gid, dst_gid, strong_profiles=True)
    bands = [BandData(s, X_gpu) for s in snapshots]

    # ---- Train the GNN inline (gatv2 + strong profiles + scorer warmup) ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    encoder = GraphEncoder(bands[0].x.shape[1], 64, 2, conv="gatv2").to(DEVICE)
    scorer = TxnScorer(encoder.out_dim, X_gpu.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(scorer.parameters()),
                                  lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    pos_weight = torch.tensor([25.0], device=DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    y_t = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
    split_labels = split.to_numpy()

    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        encoder.train(); scorer.train()
        graph_on = epoch >= 5  # scorer warmup
        for band in bands:
            idx_all = np.flatnonzero(split_labels[band.rows] == "train")
            n_tr = len(idx_all)
            if n_tr == 0:
                continue
            np.random.shuffle(idx_all)
            chunk = max(1, n_tr // 4)
            for b in range(0, n_tr, chunk):
                idx = torch.from_numpy(idx_all[b : b + chunk]).to(DEVICE)
                optimizer.zero_grad()
                logits = forward_band(encoder, scorer, band, graph_on, True)[idx]
                loss_fn(logits, y_t[band.rows][idx]).backward()
                optimizer.step()
        scheduler.step()
        encoder.eval(); scorer.eval()
        with torch.no_grad():
            sc, rr = [], []
            for band in bands:
                m = split_labels[band.rows] == "validation"
                if m.sum() == 0:
                    continue
                lg = forward_band(encoder, scorer, band, graph_on, True)
                sc.append(torch.sigmoid(lg).cpu().numpy()[m]); rr.append(band.rows[m])
            val = average_precision_score(y[np.concatenate(rr)], np.concatenate(sc))
        if val > best:
            best = val
            best_state = {k: t.detach().clone() for k, t in encoder.state_dict().items()}
        if epoch % 10 == 0:
            print(f"epoch {epoch} val {val:.4f}", flush=True)
    encoder.load_state_dict(best_state)
    print(f"GNN trained, best val {best:.4f}", flush=True)

    # ---- Extract frozen per-transaction embeddings --------------------------
    emb = np.zeros((len(df), 128), dtype=np.float32)
    encoder.eval()
    with torch.no_grad():
        for band in bands:
            if band.edge_index.shape[1] == 0:
                continue
            h = encoder(band.x, band.edge_index)
            zero = torch.zeros(1, h.shape[1], device=DEVICE)
            h_src = torch.where((band.l_src >= 0).unsqueeze(1), h[band.l_src.clamp(min=0)], zero)
            h_dst = torch.where((band.l_dst >= 0).unsqueeze(1), h[band.l_dst.clamp(min=0)], zero)
            emb[band.rows] = torch.cat([h_src, h_dst], dim=1).cpu().numpy()
    print("embeddings extracted", flush=True)

    # ---- XGBoost A/B ---------------------------------------------------------
    results = {}
    for tag, mat in (("xgb78", Xn), ("xgb78_plus_emb", np.hstack([Xn, emb.astype(np.float64)]))):
        model = xgb.XGBClassifier(
            n_estimators=4000, learning_rate=0.05, max_depth=8, min_child_weight=50,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25.0,
            eval_metric="aucpr", early_stopping_rounds=200, random_state=args.seed,
            tree_method="hist", device="cpu",
        )
        model.fit(mat[masks["train"]], y[masks["train"]],
                  eval_set=[(mat[masks["validation"]], y[masks["validation"]])], verbose=False)
        s = model.predict_proba(mat[masks["test"]])[:, 1]
        results[tag] = float(average_precision_score(y[masks["test"]], s))
        print(f"[{tag}] test AUPRC={results[tag]:.4f}", flush=True)

    results["emb_delta"] = results["xgb78_plus_emb"] - results["xgb78"]
    results["verdict"] = (
        "embeddings add signal" if results["emb_delta"] > 0.005
        else "embeddings add nothing" if results["emb_delta"] > -0.005
        else "embeddings are noise"
    )
    out = Path(config["project"]["local_workspace"]) / "results" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"embedding_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
