"""Phase 2 baselines: causal features -> temporal split -> LightGBM/XGBoost/MLP.

All models share one strictly-causal feature matrix and the chronological
60/20/20 split; every model runs the full seed list from the config; metrics
report mean +/- std across seeds plus per-seed raw values. Thresholds for
Recall@K / Precision@K are ranking-based (no threshold leakage).

Example:
    python scripts/phase2_baseline.py --config configs/ibm_aml_hi_small.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk.causal_features import build_causal_features, causal_entity_features
from finrisk.data_contract import laundering_mask, resolve_transaction_columns
from finrisk.entity_join import attach_entities
from finrisk.temporal_split import assert_split_properties, temporal_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mapping = resolve_transaction_columns(df.columns)
    df = df.rename(
        columns={
            mapping["timestamp"]: "ts",
            mapping["source_account"]: "src_account",
            mapping["destination_account"]: "dst_account",
            mapping["source_bank"]: "src_bank",
            mapping["destination_bank"]: "dst_bank",
            mapping["label"]: "label",
        }
    )
    df["ts"] = pd.to_datetime(df["ts"])
    df["label"] = laundering_mask(df["label"]).astype(np.int8)
    return df


def attach_features(df: pd.DataFrame) -> pd.DataFrame:
    usd = np.where(df["Payment Currency"].astype(str).str.strip().str.lower() == "us dollar", df["Amount Paid"], 0.0)
    df["usd_amount"] = usd
    causal = build_causal_features(df, "ts", "src_account", "dst_account", "usd_amount")

    base = pd.DataFrame(index=df.index)
    base["amount_paid"] = df["Amount Paid"].astype(float)
    base["hour"] = df["ts"].dt.hour
    base["dow"] = df["ts"].dt.dayofweek
    for col in ("Payment Format", "Payment Currency", "Receiving Currency"):
        base[col] = df[col].astype("category").cat.codes

    banks = pd.concat([df["src_bank"].astype(str), df["dst_bank"].astype(str)], axis=1)
    base["same_bank"] = (banks.iloc[:, 0] == banks.iloc[:, 1]).astype(np.int8)
    # Self-loop: source account == destination account (rare, low laundering rate).
    base["is_self_loop"] = (df["src_account"].astype(str) == df["dst_account"].astype(str)).astype(np.int8)

    out = pd.concat([base, causal], axis=1)
    out.columns = [c.replace(" ", "_").lower() for c in out.columns]

    if {"src_entity", "dst_entity"}.issubset(df.columns):
        entity = causal_entity_features(df, "ts", "src_account", "dst_account",
                                        "src_entity", "dst_entity", "usd_amount")
        out = pd.concat([out, entity], axis=1)
    return out


def recall_at_k(y_true: np.ndarray, score: np.ndarray, fraction: float) -> float:
    k = max(1, int(len(y_true) * fraction))
    top = np.argsort(-score)[:k]
    return float(y_true[top].sum() / y_true.sum())


def precision_at_k(y_true: np.ndarray, score: np.ndarray, fraction: float) -> float:
    k = max(1, int(len(y_true) * fraction))
    top = np.argsort(-score)[:k]
    return float(y_true[top].sum() / k)


def evaluate(y_true: np.ndarray, score: np.ndarray, capacities: list[float]) -> dict:
    metrics = {
        "auprc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }
    for cap in capacities:
        tag = str(cap).replace(".", "_")
        metrics[f"recall_at_{tag}"] = recall_at_k(y_true, score, cap)
        metrics[f"precision_at_{tag}"] = precision_at_k(y_true, score, cap)
    return metrics


def run_lightgbm(X_tr, y_tr, X_va, y_va, X_te, seed) -> np.ndarray:
    def auprc_metric(y_true, y_pred):
        return "auprc", average_precision_score(y_true, y_pred), True

    # LightGBM 4.7 在大规模稀疏 DataFrame 上偶现访问违例；使用独立副本 + 显式 dtype 规避。
    X_tr_ = X_tr.astype("float32", copy=True)
    X_va_ = X_va.astype("float32", copy=True)
    X_te_ = X_te.astype("float32", copy=True)
    y_tr_ = y_tr.astype(np.int32)
    y_va_ = y_va.astype(np.int32)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=200,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        scale_pos_weight=25.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr_, y_tr_,
        eval_set=[(X_va_, y_va_)],
        eval_metric=auprc_metric,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    return model.predict_proba(X_te_)[:, 1]


def run_xgboost(X_tr, y_tr, X_va, y_va, X_te, seed) -> np.ndarray:
    model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.9,
        scale_pos_weight=25.0,
        eval_metric="aucpr",
        early_stopping_rounds=200,
        random_state=seed,
        device=DEVICE,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model.predict_proba(X_te)[:, 1]


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_mlp(X_tr, y_tr, X_va, y_va, X_te, seed) -> np.ndarray:
    scaler = StandardScaler().fit(X_tr)
    arrays = [scaler.transform(part).astype(np.float32) for part in (X_tr, X_va, X_te)]
    x_tr, x_va, x_te = [torch.from_numpy(a).to(DEVICE) for a in arrays]
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(DEVICE)

    torch.manual_seed(seed)
    model = MLP(x_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([25.0], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auprc, best_state, patience = -1.0, None, 0
    batch = 65536
    n = len(x_tr)
    for epoch in range(30):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(model(x_va)).cpu().numpy()
        auprc = average_precision_score(y_va, val_score)
        if auprc > best_auprc:
            best_auprc, patience = auprc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x_te)).cpu().numpy()


MODELS = {"lightgbm": run_lightgbm, "xgboost": run_xgboost, "mlp": run_mlp}


def aggregate(per_seed: list[dict]) -> dict:
    keys = per_seed[0].keys()
    out = {}
    for key in keys:
        values = np.array([run[key] for run in per_seed])
        out[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 multi-model baselines.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    parser.add_argument("--transactions", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--attach-entities", action="store_true",
                        help="join Entity IDs from the accounts CSV and add entity-level causal features")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tx_path = args.transactions or Path(config["data"]["transactions_path"])
    if not tx_path.is_file():
        parser.error(f"transaction file not found: {tx_path}")

    df = load_transactions(tx_path)

    ac_path = config["data"].get("accounts_path")
    if args.attach_entities:
        if not ac_path or not Path(ac_path).is_file():
            parser.error(f"accounts path not found for entity features: {ac_path}")
        print("attaching Entity IDs (strict past, per-entity)...", flush=True)
        df = attach_entities(df, Path(ac_path))
    print(f"transactions: {len(df):,}  positives: {df['label'].sum():,} ({df['label'].mean():.4%})", flush=True)

    split = temporal_split(df, "ts", config["split"]["train_ratio"], config["split"]["validation_ratio"])
    assert_split_properties(df, "ts", split)
    masks = {name: (split == name).to_numpy() for name in ("train", "validation", "test")}
    y = df["label"].to_numpy()
    for name in ("train", "validation", "test"):
        part = df[split == name]
        print(f"{name:>10}: {len(part):,} rows  positives {part['label'].sum():,}", flush=True)

    print("building causal features (single pass, shared by all models)...", flush=True)
    X = attach_features(df)
    feature_cols = list(X.columns)
    print(f"features: {len(feature_cols)}", flush=True)

    capacities = config["evaluation"]["alert_capacities"]
    seeds = config["project"]["seed_list"]
    out_dir = Path(config["project"]["local_workspace"]) / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report: dict = {"run_id": run_id, "features": len(feature_cols), "seeds": seeds, "models": {}}
    for model_name in args.models:
        runner = MODELS[model_name]
        per_seed_test = []
        for seed in seeds:
            # Validation is consumed by early stopping (model selection), so
            # only test metrics are reported to avoid optimistic bias.
            test_score = runner(X.loc[masks["train"], feature_cols], y[masks["train"]],
                                X.loc[masks["validation"], feature_cols], y[masks["validation"]],
                                X.loc[masks["test"], feature_cols], seed)
            per_seed_test.append(evaluate(y[masks["test"]], test_score, capacities))
            print(f"[{model_name}] seed {seed} test AUPRC={per_seed_test[-1]['auprc']:.4f}", flush=True)
            pd.DataFrame({"y_true": y[masks["test"]], "y_score": test_score}).to_parquet(
                out_dir / f"{model_name}_{run_id}_seed{seed}_test_predictions.parquet", index=False
            )
        report["models"][model_name] = {"test": aggregate(per_seed_test)}

    (out_dir / f"multi_model_{run_id}_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
