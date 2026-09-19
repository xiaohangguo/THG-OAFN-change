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

    n = len(df)
    # Iterate over account groups [start, end) in the (account, time) sort order.
    boundaries = np.flatnonzero(acc_sorted[1:] != acc_sorted[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    for start, end in zip(starts, ends):
        g_ts = ts_sorted[start:end]
        g_usd = usd_sorted[start:end]
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
    return pd.concat([sent, received], axis=1)
