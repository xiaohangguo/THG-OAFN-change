"""Phase 2 baseline: causal features -> temporal split -> LightGBM/MLP.

Produces the first trustworthy metrics on IBM AML HI-Small under the
experiment protocol: chronological 60/20/20 split, strictly causal account
features, thresholds locked on validation, AUPRC + Recall@K + Precision@K,
per-transaction predictions saved to the local workspace (never committed).

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
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk.causal_features import build_causal_features
from finrisk.data_contract import laundering_mask, resolve_transaction_columns
from finrisk.temporal_split import assert_split_properties, temporal_split


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

    banks = pd.concat(
        [df["src_bank"].astype(str), df["dst_bank"].astype(str)], axis=1
    )
    base["same_bank"] = (banks.iloc[:, 0] == banks.iloc[:, 1]).astype(np.int8)

    out = pd.concat([base, causal], axis=1)
    out.columns = [c.replace(" ", "_").lower() for c in out.columns]
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


def run_lightgbm(X_tr, y_tr, X_va, y_va, X_te, seed) -> dict:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        scale_pos_weight=float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return {
        "validation": model.predict_proba(X_va)[:, 1],
        "test": model.predict_proba(X_te)[:, 1],
        "best_iteration": int(model.best_iteration_ or 600),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 causal-feature baselines.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ibm_aml_hi_small.yaml")
    parser.add_argument("--transactions", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tx_path = args.transactions or Path(config["data"]["transactions_path"])
    if not tx_path.is_file():
        parser.error(f"transaction file not found: {tx_path} (download + audit first)")

    df = load_transactions(tx_path)
    print(f"transactions: {len(df):,}  positives: {df['label'].sum():,} "
          f"({df['label'].mean():.4%})", flush=True)

    split = temporal_split(df, "ts", config["split"]["train_ratio"], config["split"]["validation_ratio"])
    assert_split_properties(df, "ts", split)
    for name in ("train", "validation", "test"):
        part = df[split == name]
        print(f"{name:>10}: {len(part):,} rows  positives {part['label'].sum():,}  "
              f"[{part['ts'].min()} .. {part['ts'].max()}]", flush=True)

    X = attach_features(df)
    feature_cols = [c for c in X.columns if c not in ("label",)]
    print(f"features: {len(feature_cols)}", flush=True)

    masks = {name: (split == name).to_numpy() for name in ("train", "validation", "test")}
    y = df["label"].to_numpy()

    seed = int(config["project"]["seed_list"][0])
    result = run_lightgbm(
        X.loc[masks["train"], feature_cols], y[masks["train"]],
        X.loc[masks["validation"], feature_cols], y[masks["validation"]],
        X.loc[masks["test"], feature_cols], y[masks["test"]],
        seed,
    )
    capacities = config["evaluation"]["alert_capacities"]
    metrics = {
        "validation": evaluate(y[masks["validation"]], result["validation"], capacities),
        "test": evaluate(y[masks["test"]], result["test"], capacities),
    }
    metrics["best_iteration"] = result["best_iteration"]

    out_dir = Path(config["project"]["local_workspace"]) / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"lightgbm_{run_id}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    preds = pd.DataFrame({"split": split.to_numpy(), "y_true": y})
    scores = np.full(len(df), np.nan)
    scores[masks["validation"]] = result["validation"]
    scores[masks["test"]] = result["test"]
    preds["y_score"] = scores
    preds.to_parquet(out_dir / f"lightgbm_{run_id}_predictions.parquet", index=False)

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"saved -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
