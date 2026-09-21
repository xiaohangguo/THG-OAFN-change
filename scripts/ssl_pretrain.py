"""Route A: self-supervised temporal graph pretraining + embedding A/B.

Zero-label pretraining (masked profile reconstruction, GraphMAE-style) on
snapshot graphs STRICTLY BEFORE the test boundary — no labels touched, so no
leakage; test-band embeddings come from a frozen encoder that has never seen
test-period data. The A/B then answers: do SSL embeddings add anything over
the 78 handcrafted causal features?

Example:
    python scripts/ssl_pretrain.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
from finrisk.temporal_split import temporal_split  # noqa: E402
from phase3_graph_baseline import DEVICE, BandData, GraphEncoder, _to_local  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssl-epochs", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
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

    scaler = StandardScaler().fit(Xn[masks["train"]])
    Xs = np.nan_to_num(scaler.transform(Xn), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_gpu = torch.from_numpy(Xs).to(DEVICE)

    snapshots = build_banded_snapshots(df, src_gid, dst_gid, strong_profiles=True)
    bands = [BandData(s, X_gpu) for s in snapshots]
    split_labels = split.to_numpy()

    # SSL bands: everything before the first test transaction (strict cutoff).
    test_start_row = int(np.flatnonzero(masks["test"])[0])
    ssl_bands = [b for b in bands if b.rows.max() < test_start_row]
    print(f"SSL bands: {len(ssl_bands)} of {len(bands)} (pre-test only)", flush=True)

    # ---- Self-supervised pretraining: masked profile reconstruction --------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    feat_dim = bands[0].x.shape[1]
    encoder = GraphEncoder(feat_dim, 64, 2, conv="gatv2").to(DEVICE)
    decoder = nn.Linear(encoder.out_dim, feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                                  lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ssl_epochs)

    for epoch in range(args.ssl_epochs):
        encoder.train(); decoder.train()
        losses = []
        for band in ssl_bands:
            if band.edge_index.shape[1] == 0:
                continue
            x = band.x
            n = x.shape[0]
            perm = torch.randperm(n, device=DEVICE)
            n_mask = max(1, int(n * args.mask_ratio))
            mask_idx = perm[:n_mask]
            x_in = x.clone()
            x_in[mask_idx] = 0.0
            h = encoder(x_in, band.edge_index)
            recon = decoder(h)
            loss = nn.functional.mse_loss(recon[mask_idx], x[mask_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        scheduler.step()
        if epoch % 10 == 0:
            print(f"[ssl] epoch {epoch} recon-mse {np.mean(losses):.4f}", flush=True)

    # ---- Extract frozen embeddings for EVERY band (encoder never saw test) --
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
    print("SSL embeddings extracted", flush=True)

    # ---- XGBoost A/B ---------------------------------------------------------
    results = {}
    for tag, mat in (("xgb78", Xn), ("xgb78_plus_ssl", np.hstack([Xn, emb.astype(np.float64)]))):
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

    results["ssl_delta"] = results["xgb78_plus_ssl"] - results["xgb78"]
    results["verdict"] = (
        "SSL embeddings add signal" if results["ssl_delta"] > 0.005
        else "SSL embeddings add nothing" if results["ssl_delta"] > -0.005
        else "SSL embeddings hurt")
    results["ssl_config"] = {"epochs": args.ssl_epochs, "mask_ratio": args.mask_ratio, "conv": "gatv2"}

    out = Path(config["project"]["local_workspace"]) / "results" / "routeA_ssl"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ssl_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.save(str(out / "ssl_embeddings.npy"), emb)
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
