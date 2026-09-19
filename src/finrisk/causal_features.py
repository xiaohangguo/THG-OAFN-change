"""Strictly causal account-history features for transaction risk scoring.

Visibility rule: a transaction at time t only sees history with
timestamp < t on the same account. Concurrent transactions (equal timestamps)
are invisible to each other, which is the defensible ordering when the CSV
carries no intra-timestamp sequence.

Windows are half-open [t - w, t). Statistics per account side (sent /
received): count, usd_count, usd_sum, usd_mean, usd_std. Amount statistics
are computed on the USD-denominated subset only, because mixing currencies
without an auditable FX source would corrupt the semantics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {
    "w1h": pd.Timedelta(hours=1),
    "w24h": pd.Timedelta(hours=24),
    "w7d": pd.Timedelta(days=7),
}


def _window_stats_one_group(
    ts: np.ndarray, usd: np.ndarray, delta: pd.Timedelta
) -> dict[str, np.ndarray]:
    """Prefix-sum window statistics for one account's ascending transactions.

    ts: int64 nanoseconds ascending; usd: USD amount per transaction (0 when
    the row is not USD-denominated, see usd_amount() in the baseline script).
    """
    cum_sum = np.concatenate([[0.0], np.cumsum(usd)])
    cum_sq = np.concatenate([[0.0], np.cumsum(usd * usd)])

    # Right boundary: first index with ts >= t_i (strict past visibility).
    right = np.searchsorted(ts, ts, side="left")
    left = np.searchsorted(ts, ts - np.int64(delta.value), side="left")

    count = (right - left).astype(np.float64)
    s = cum_sum[right] - cum_sum[left]
    q = cum_sq[right] - cum_sq[left]
    c = count
    mean = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
    var = np.divide(q - s * s / np.maximum(c, 1), np.maximum(c - 1, 1), out=np.zeros_like(q), where=c > 1)
    std = np.sqrt(np.maximum(var, 0.0))
    return {"count": count, "sum": s, "mean": mean, "std": std}


def causal_account_features(
    df: pd.DataFrame,
    timestamp_col: str,
    account_col: str,
    usd_amount_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Build causal window features for one account role (sent or received)."""
    ts_ns = pd.to_datetime(df[timestamp_col]).astype("int64").to_numpy()
    usd = df[usd_amount_col].to_numpy(dtype="float64")

    order = np.lexsort((ts_ns, df[account_col].to_numpy()))
    ts_sorted = ts_ns[order]
    acc_sorted = df[account_col].to_numpy()[order]
    usd_sorted = usd[order]

    stats_names = ("count", "sum", "mean", "std")
    features = {
        f"{prefix}_{wname}_{stat}": np.zeros(len(df))
        for wname in WINDOWS
        for stat in stats_names
    }
    # Whole-history degree up to (strictly before) each transaction: sensitive
    # to fan-in / fan-out laundering shapes.
    features[f"{prefix}_past_count"] = np.zeros(len(df))

    n = len(df)
    # Iterate over account groups [start, end) in the (account, time) sort order.
    boundaries = np.flatnonzero(acc_sorted[1:] != acc_sorted[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    for start, end in zip(starts, ends):
        g_ts = ts_sorted[start:end]
        g_usd = usd_sorted[start:end]
        # Strict-past degree: index of the first row sharing this timestamp.
        past = np.searchsorted(g_ts, g_ts, side="left")
        features[f"{prefix}_past_count"][order[start:end]] = past
        for wname, delta in WINDOWS.items():
            stats = _window_stats_one_group(g_ts, g_usd, delta)
            for stat, values in stats.items():
                features[f"{prefix}_{wname}_{stat}"][order[start:end]] = values

    return pd.DataFrame(features, index=df.index)


def build_causal_features(
    df: pd.DataFrame,
    timestamp_col: str,
    source_account_col: str,
    destination_account_col: str,
    usd_amount_col: str,
) -> pd.DataFrame:
    """Combine sent-side and received-side causal features."""
    sent = causal_account_features(df, timestamp_col, source_account_col, usd_amount_col, prefix="src")
    received = causal_account_features(df, timestamp_col, destination_account_col, usd_amount_col, prefix="dst")
    pair = causal_pair_features(df, timestamp_col, source_account_col, destination_account_col, usd_amount_col)
    return pd.concat([sent, received, pair], axis=1)


def causal_pair_features(
    df: pd.DataFrame,
    timestamp_col: str,
    source_account_col: str,
    destination_account_col: str,
    usd_amount_col: str,
) -> pd.DataFrame:
    """Strictly-causal account-pair features: past interaction strength between the specific source→destination pair."""

    ts_ns = pd.to_datetime(df[timestamp_col]).astype("int64").to_numpy()
    src = df[source_account_col].astype(str).to_numpy()
    dst = df[destination_account_col].astype(str).to_numpy()
    usd = df[usd_amount_col].to_numpy(dtype="float64")

    order = np.lexsort((ts_ns, src, dst))
    ts_sorted = ts_ns[order]
    src_sorted = src[order]
    dst_sorted = dst[order]
    usd_sorted = usd[order]

    # build pair key as bytes for fast compare
    n = len(df)
    key = np.empty(n, dtype=object)
    for i in range(n):
        key[i] = src_sorted[i] + "\x1f" + dst_sorted[i]

    stats_names = ("count", "sum", "mean", "std")
    features = {f"pair_{w}_{stat}": np.zeros(n)
                for w in WINDOWS for stat in stats_names}
    features["pair_past_count"] = np.zeros(n)

    # Iterate groups by pair in (pair, time) order.
    boundaries = np.flatnonzero(key[1:] != key[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    for start, end in zip(starts, ends):
        g_ts = ts_sorted[start:end]
        g_usd = usd_sorted[start:end]
        past = np.searchsorted(g_ts, g_ts, side="left")
        features["pair_past_count"][order[start:end]] = past
        for wname, delta in WINDOWS.items():
            stats = _window_stats_one_group(g_ts, g_usd, delta)
            for stat, values in stats.items():
                features[f"pair_{wname}_{stat}"][order[start:end]] = values

    return pd.DataFrame(features, index=df.index)


def causal_entity_features(
    df: pd.DataFrame,
    timestamp_col: str,
    source_account_col: str,
    destination_account_col: str,
    source_entity_col: str,
    destination_entity_col: str,
    usd_amount_col: str,
) -> pd.DataFrame:
    """Strictly-causal *brother-account* entity features (difference form).

    Each transaction is labelled by the legal ``Entity ID`` of its source and
    destination account. Naively aggregating entity history was found to *hurt*
    (mid-size XGBoost AUPRC 0.0863 -> 0.0783) because ~62% of entities own a
    single account, making entity stats a perfect duplicate of account stats
    (pure noise). The value of entities in AML layering is a legal entity
    shuttling value across *multiple* of its own accounts.

    So we return only the *difference* between entity-level history and
    account-level history: window statistics of the entity's *other* (brother)
    accounts. It is naturally 0 for single-account entities (no brothers), adding
    no noise there, and exposes layering where multi-account shuttling exists.
    Visibility is the same strict-past rule as account features.
    """
    # same-entity transfer flag (intra-entity layering).
    ent_src = df[source_entity_col].astype(str).to_numpy()
    ent_dst = df[destination_entity_col].astype(str).to_numpy()
    out = {"same_entity_transfer": (ent_src == ent_dst).astype(np.int8)}

    # window names shared between account & entity stats
    windows = ["w1h", "w24h", "w7d"]
    for side, acc_col, ent_col in (
        ("src", source_account_col, source_entity_col),
        ("dst", destination_account_col, destination_entity_col),
    ):
        acc_stats = causal_account_features(
            df, timestamp_col, acc_col, usd_amount_col, prefix=f"{side}_account"
        )
        ent_stats = causal_account_features(
            df, timestamp_col, ent_col, usd_amount_col, prefix=f"{side}_entity"
        )
        # Brother contribution = entity_* - account_* per window (saturate at 0).
        for w in windows:
            c = f"{side}_bro_{w}_count"
            s = f"{side}_bro_{w}_sum"
            out[c] = np.maximum(0.0, ent_stats[f"{side}_entity_{w}_count"].to_numpy()
                                - acc_stats[f"{side}_account_{w}_count"].to_numpy())
            out[s] = np.maximum(0.0, ent_stats[f"{side}_entity_{w}_sum"].to_numpy()
                                - acc_stats[f"{side}_account_{w}_sum"].to_numpy())
    return pd.DataFrame(out, index=df.index)
