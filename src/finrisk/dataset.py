"""Shared dataset loading and causal feature assembly for IBM AML HI-Small.

Single source of truth for the feature matrix used by every model (table and
graph): transaction base columns + strictly-causal account windows, plus
entity-window features when src/dst Entity IDs have been attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finrisk.causal_features import build_causal_features, causal_entity_features
from finrisk.data_contract import laundering_mask, resolve_transaction_columns


def load_transactions(path) -> pd.DataFrame:
    """Load raw CSV, map contract columns to canonical names, parse label."""
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
    """Feature matrix: transaction base + strictly-causal account windows.

    Adds entity-window columns when ``src_entity``/``dst_entity`` exist.
    Also writes ``df['usd_amount']`` in place (USD-or-zero amounts).
    """
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
    base["is_self_loop"] = (df["src_account"].astype(str) == df["dst_account"].astype(str)).astype(np.int8)

    out = pd.concat([base, causal], axis=1)
    out.columns = [c.replace(" ", "_").lower() for c in out.columns]

    if {"src_entity", "dst_entity"}.issubset(df.columns):
        entity = causal_entity_features(df, "ts", "src_account", "dst_account",
                                        "src_entity", "dst_entity", "usd_amount")
        out = pd.concat([out, entity], axis=1)
    return out
