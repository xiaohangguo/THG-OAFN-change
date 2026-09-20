"""Phase 3: banded-snapshot GraphSAGE baseline over the account graph.

Every time band is scored from a strictly-past window graph: the snapshot for
band B contains only edges with timestamp < B.start, so message passing can
never see the transaction being scored, its band, or anything later. The
transaction scorer consumes [src_emb, dst_emb, seen_flags, causal txn feats]
— the same feature matrix as the Phase 2 table baselines.

Ablations (honest, same training loop):
  --no-graph      graph embeddings zeroed  -> pure txn-feature MLP control
  --no-txn-feats  scorer sees embeddings only
  --window 7d/3d  snapshot lookback window
Reports mean +/- std over seeds on the test split only; validation is used
for early stopping exactly like the table baselines.

Example:
    python scripts/phase3_graph_baseline.py --attach-entities
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
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from finrisk.dataset import attach_features, load_transactions  # noqa: E402
from finrisk.entity_join import attach_entities  # noqa: E402
from finrisk.graph_builder import build_banded_snapshots, build_global_account_index  # noqa: E402
from finrisk.temporal_split import assert_split_properties, temporal_split  # noqa: E402
from phase2_baseline import aggregate, evaluate  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int = 2) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x


class TxnScorer(nn.Module):
    def __init__(self, emb_dim: int, txn_dim: int, hidden: int = 256) -> None:
        super().__init__()
        in_dim = 2 * emb_dim + 2 + txn_dim  # +2 seen flags
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h_src, h_dst, seen, txn):
        return self.net(torch.cat([h_src, h_dst, seen, txn], dim=-1)).squeeze(-1)


def _to_local(node_ids: np.ndarray, gids: np.ndarray) -> np.ndarray:
    """Vectorized global->local mapping over sorted node ids (-1 if absent)."""
    if len(node_ids) == 0:
        return np.full(len(gids), -1, dtype=np.int64)
    pos = np.searchsorted(node_ids, gids)
    pos_c = np.minimum(pos, len(node_ids) - 1)
    ok = (pos < len(node_ids)) & (node_ids[pos_c] == gids)
    return np.where(ok, pos, -1).astype(np.int64)


class BandData:
    """Per-band precomputed tensors for training/inference."""

    def __init__(self, snap, X_gpu: torch.Tensor):
        self.band_id = snap.band_id
        self.rows = snap.txn_rows
        self.x = torch.from_numpy(snap.node_feat).to(DEVICE)
        self.edge_index = torch.from_numpy(snap.edge_index).to(DEVICE)
        l_src = _to_local(snap.node_ids, snap.txn_src)
        l_dst = _to_local(snap.node_ids, snap.txn_dst)
        self.l_src = torch.from_numpy(l_src).to(DEVICE)
        self.l_dst = torch.from_numpy(l_dst).to(DEVICE)
        self.seen = torch.from_numpy(
            np.stack([(l_src >= 0).astype(np.float32), (l_dst >= 0).astype(np.float32)], axis=1)
        ).to(DEVICE)
        self.txn = X_gpu  # full matrix; caller indexes with band rows


def forward_band(encoder, scorer, band: BandData, use_graph: bool, use_txn: bool):
    if use_graph and band.edge_index.shape[1] > 0:
        h = encoder(band.x, band.edge_index)
        zero = torch.zeros(1, h.shape[1], device=DEVICE)
        h_src = torch.where((band.l_src >= 0).unsqueeze(1), h[band.l_src.clamp(min=0)], zero)
        h_dst = torch.where((band.l_dst >= 0).unsqueeze(1), h[band.l_dst.clamp(min=0)], zero)
    else:
        emb_dim = next(encoder.convs.children().__iter__()).out_channels if use_graph else 64
        h_src = torch.zeros(len(band.rows), emb_dim, device=DEVICE)
        h_dst = torch.zeros(len(band.rows), emb_dim, device=DEVICE)
    txn = band.txn[band.rows] if use_txn else torch.zeros(len(band.rows), 0, device=DEVICE)
    return scorer(h_src, h_dst, band.seen, txn)


def run_graph_model(bands, split_labels, y, X_gpu, seed, args, capacities):
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = bands[0].x.shape[1]
    encoder = GraphEncoder(in_dim, args.hidden, args.layers).to(DEVICE)
    emb_dim = args.hidden
    scorer = TxnScorer(emb_dim, X_gpu.shape[1] if not args.no_txn_feats else 0).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(scorer.parameters()), lr=1e-3, weight_decay=1e-4
    )
    pos_weight = torch.tensor([25.0], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def split_of(band: BandData) -> np.ndarray:
        return split_labels[band.rows]

    y_t = torch.from_numpy(y.astype(np.float32)).to(DEVICE)

    best_auprc, best_state, patience = -1.0, None, 0
    for epoch in range(args.max_epochs):
        encoder.train(); scorer.train()
        order = np.arange(len(bands))
        for i in order:
            band = bands[i]
            mask = split_of(band) == "train"
            if mask.sum() == 0:
                continue
            idx = torch.from_numpy(np.flatnonzero(mask)).to(DEVICE)
            optimizer.zero_grad()
            logits_all = forward_band(encoder, scorer, band, not args.no_graph, not args.no_txn_feats)
            logits = logits_all[idx]
            loss = loss_fn(logits, y_t[band.rows][idx])
            loss.backward()
            optimizer.step()

        encoder.eval(); scorer.eval()
        with torch.no_grad():
            scores = []
            for band in bands:
                mask = split_of(band) == "validation"
                if mask.sum() == 0:
                    continue
                logits = forward_band(encoder, scorer, band, not args.no_graph, not args.no_txn_feats)
                scores.append((band.rows[mask], torch.sigmoid(logits).cpu().numpy()[mask]))
            if not scores:
                break
            rows = np.concatenate([s[0] for s in scores])
            preds = np.concatenate([s[1] for s in scores])
            auprc = average_precision_score(y[rows], preds)
        if auprc > best_auprc + 1e-5:
            best_auprc, patience = auprc, 0
            best_state = (
                {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                {k: v.detach().clone() for k, v in scorer.state_dict().items()},
            )
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state[0]); scorer.load_state_dict(best_state[1])
    encoder.eval(); scorer.eval()
    with torch.no_grad():
        test_scores = []
        for band in bands:
            mask = split_of(band) == "test"
            if mask.sum() == 0:
                continue
            logits = forward_band(encoder, scorer, band, not args.no_graph, not args.no_txn_feats)
            test_scores.append((band.rows[mask], torch.sigmoid(logits).cpu().numpy()[mask]))
        rows = np.concatenate([s[0] for s in test_scores])
        preds = np.concatenate([s[1] for s in test_scores])

    order = np.argsort(rows)
    return rows[order], preds[order], best_auprc


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 graph baseline (banded snapshots).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    parser.add_argument("--transactions", type=Path, default=None)
    parser.add_argument("--attach-entities", action="store_true")
    parser.add_argument("--band-hours", type=int, default=12)
    parser.add_argument("--window-days", type=float, default=7.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--no-graph", action="store_true", help="ablation: zero graph embeddings")
    parser.add_argument("--no-txn-feats", action="store_true", help="ablation: embeddings only")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--tag", default="graphsage")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tx_path = args.transactions or Path(config["data"]["transactions_path"])
    df = load_transactions(tx_path)
    if args.attach_entities:
        ac_path = config["data"].get("accounts_path")
        df = attach_entities(df, Path(ac_path))

    split = temporal_split(df, "ts", config["split"]["train_ratio"], config["split"]["validation_ratio"])
    assert_split_properties(df, "ts", split)
    split_labels = split.to_numpy()
    y = df["label"].to_numpy()

    print("building causal features...", flush=True)
    X = attach_features(df)
    print(f"txn features: {X.shape[1]}", flush=True)

    # Standardize on TRAIN rows only (fit never sees validation/test).
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X.to_numpy(dtype=np.float64)[split_labels == "train"])
    X_scaled = scaler.transform(X.to_numpy(dtype=np.float64)).astype(np.float32)
    X_np = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print("building banded snapshots...", flush=True)
    src_gid, dst_gid, _ = build_global_account_index(df)
    snapshots = build_banded_snapshots(
        df, src_gid, dst_gid,
        band=pd.Timedelta(hours=args.band_hours),
        window=pd.Timedelta(days=args.window_days),
    )
    X_gpu = torch.from_numpy(X_np).to(DEVICE)
    bands = [BandData(snap, X_gpu) for snap in snapshots]
    n_empty = sum(1 for b in bands if b.edge_index.shape[1] == 0)
    # Anti-regression: the graph path must actually see the accounts.
    seen_rates = [float(b.seen[:, 0].mean().item()) for b in bands if len(b.rows) > 0 and b.edge_index.shape[1] > 0]
    print(f"bands: {len(bands)} (empty-window bands: {n_empty}); "
          f"src-seen rate across graphed bands: min={min(seen_rates):.3f} mean={np.mean(seen_rates):.3f}", flush=True)
    if np.mean(seen_rates) < 0.5:
        raise RuntimeError("src-seen rate < 50%: graph embeddings would be mostly zero — mapping bug")

    seeds = args.seeds or config["project"]["seed_list"]
    capacities = config["evaluation"]["alert_capacities"]
    out_dir = Path(config["project"]["local_workspace"]) / "results" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    per_seed = []
    for seed in seeds:
        rows, preds, best_val = run_graph_model(bands, split_labels, y, X_gpu, seed, args, capacities)
        metrics = evaluate(y[rows], preds, capacities)
        metrics["best_val_auprc"] = best_val
        per_seed.append(metrics)
        print(f"[{args.tag}] seed {seed} test AUPRC={metrics['auprc']:.4f} (val {best_val:.4f})", flush=True)
        pd.DataFrame({"row": rows, "y_true": y[rows], "y_score": preds}).to_parquet(
            out_dir / f"{args.tag}_{run_id}_seed{seed}_test_predictions.parquet", index=False
        )

    report = {
        "run_id": run_id,
        "tag": args.tag,
        "config": {
            "band_hours": args.band_hours, "window_days": args.window_days,
            "hidden": args.hidden, "layers": args.layers,
            "no_graph": args.no_graph, "no_txn_feats": args.no_txn_feats,
            "attach_entities": args.attach_entities, "txn_features": X.shape[1],
        },
        "seeds": seeds,
        "test": aggregate(per_seed),
    }
    path = out_dir / f"{args.tag}_{run_id}_metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["test"], indent=2), flush=True)
    print(f"saved -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
