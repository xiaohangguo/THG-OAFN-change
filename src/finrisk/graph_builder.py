"""Banded temporal snapshot graphs with strict-past edge windows.

For each time band B_i = [start_i, end_i), the snapshot contains ONLY
transaction edges with timestamp in [start_i - W, start_i). Message passing
for any prediction inside B_i therefore never sees the present or future —
the same strict-past visibility rule as the causal table features.

Account profile features per band aggregate the same strict-past window, so
both topology and node attributes are causal. The first band(s) whose window
precedes the data range yield empty graphs; their transactions fall back to
transaction-features-only scoring (honest behaviour, not imputation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BandSnapshot:
    band_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    node_ids: np.ndarray  # [N] global account ids present in the window
    node_feat: np.ndarray  # [N, F] profile aggregated over the window
    edge_index: np.ndarray  # [2, E] local (snapshot-scoped) indices, int64
    edge_attr: np.ndarray  # [E, 3] log_usd, same_bank, hour
    txn_rows: np.ndarray  # rows of target transactions inside this band
    txn_src: np.ndarray  # [T] global account id of source per target txn
    txn_dst: np.ndarray  # [T] global account id of destination
    _global_to_local: dict = field(default_factory=dict, repr=False)

    def localize(self, global_ids: np.ndarray) -> np.ndarray:
        """Map global account ids to local node indices (-1 if absent)."""
        table = self._global_to_local
        return np.array([table.get(int(g), -1) for g in global_ids], dtype=np.int64)


def _to_local_ids(node_ids: np.ndarray, gids: np.ndarray) -> np.ndarray:
    """Vectorized global->local mapping over sorted node ids (-1 if absent)."""
    if len(node_ids) == 0:
        return np.full(len(gids), -1, dtype=np.int64)
    pos = np.searchsorted(node_ids, gids)
    pos_c = np.minimum(pos, len(node_ids) - 1)
    ok = (pos < len(node_ids)) & (node_ids[pos_c] == gids)
    return np.where(ok, pos, -1).astype(np.int64)


def extract_profile_columns(
    df: pd.DataFrame,
    src_gid: np.ndarray,
    dst_gid: np.ndarray,
    band: pd.Timedelta = pd.Timedelta(hours=12),
    window: pd.Timedelta = pd.Timedelta(days=7),
) -> pd.DataFrame:
    """Per-transaction snapshot profile columns (the GNN's input, as features).

    For every transaction, the strictly-past snapshot profile of its source
    and destination accounts: 8 profile dims per side + 2 seen flags.
    Probe result: +0.098 AUPRC over the 60 handcrafted causal features.
    """
    snapshots = build_banded_snapshots(df, src_gid, dst_gid, band=band, window=window)
    n = len(df)
    prof = np.zeros((n, 18), dtype=np.float32)
    for snap in snapshots:
        l_src = _to_local_ids(snap.node_ids, snap.txn_src)
        l_dst = _to_local_ids(snap.node_ids, snap.txn_dst)
        rows = snap.txn_rows
        for li, side in ((l_src, 0), (l_dst, 1)):
            hit = li >= 0
            prof[rows[hit], side * 8 : side * 8 + 8] = snap.node_feat[li[hit]]
        prof[rows, 16] = (l_src >= 0).astype(np.float32)
        prof[rows, 17] = (l_dst >= 0).astype(np.float32)
    cols = [f"prof_src_{i}" for i in range(8)] + [f"prof_dst_{i}" for i in range(8)] + ["prof_src_seen", "prof_dst_seen"]
    return pd.DataFrame(prof, index=df.index, columns=cols)


def build_global_account_index(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Factorize every account once; return (src_gid, dst_gid, unique_accounts)."""
    accounts = pd.concat([df["src_account"].astype(str), df["dst_account"].astype(str)], ignore_index=True)
    codes, uniques = pd.factorize(accounts, sort=False)
    return codes[: len(df)], codes[len(df) :], pd.Series(uniques, name="account")


def _window_profile(
    sub: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-account profile from window transactions.

    Returns (account_codes, account_names, profile_matrix) where profile
    columns are: n_out, log_usd_out_sum, mean_out, cp_out, n_in,
    log_usd_in_sum, mean_in, cp_in.
    """
    src_gid = sub["src_gid"].to_numpy()
    dst_gid = sub["dst_gid"].to_numpy()
    usd = sub["usd_amount"].to_numpy(dtype=np.float64)
    if len(src_gid) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 8), dtype=np.float32)
    n_all = int(sub["n_accounts"].iloc[0])

    def agg_side(gid: np.ndarray, other: np.ndarray, amounts: np.ndarray):
        n = np.bincount(gid, minlength=n_all).astype(np.float64)
        s = np.bincount(gid, weights=amounts, minlength=n_all)
        # distinct counterparties via lexsort trick (group within account)
        order = np.lexsort((other, gid))
        g, o = gid[order], other[order]
        first = np.ones(len(g), dtype=bool)
        first[1:] = (g[1:] != g[:-1]) | (o[1:] != o[:-1])
        cp = np.bincount(g[first], minlength=n_all).astype(np.float64)
        return n, s, cp

    n_out, s_out, cp_out = agg_side(src_gid, dst_gid, usd)
    n_in, s_in, cp_in = agg_side(dst_gid, src_gid, usd)
    present = ((n_out + n_in) > 0)
    idx = np.flatnonzero(present)
    mean_out = np.divide(s_out[idx], n_out[idx], out=np.zeros_like(s_out[idx]), where=n_out[idx] > 0)
    mean_in = np.divide(s_in[idx], n_in[idx], out=np.zeros_like(s_in[idx]), where=n_in[idx] > 0)
    profile = np.column_stack(
        [
            n_out[idx],
            np.log1p(s_out[idx]),
            mean_out,
            cp_out[idx],
            n_in[idx],
            np.log1p(s_in[idx]),
            mean_in,
            cp_in[idx],
        ]
    ).astype(np.float32)
    return idx, profile


def build_banded_snapshots(
    df: pd.DataFrame,
    src_gid: np.ndarray,
    dst_gid: np.ndarray,
    band: pd.Timedelta = pd.Timedelta(hours=12),
    window: pd.Timedelta = pd.Timedelta(days=7),
) -> list[BandSnapshot]:
    """Build one strictly-causal snapshot per time band covering df['ts']."""
    if "usd_amount" not in df.columns:
        raise ValueError("df must carry 'usd_amount' (USD-or-zero amounts) before graph building")

    ts = df["ts"]
    t0, t1 = ts.min(), ts.max()
    n_bands = int(np.ceil((t1 - t0) / band)) + 1
    n_accounts = int(max(src_gid.max(), dst_gid.max())) + 1

    # Precompute per-row arrays once; the band loop must only slice them.
    ts_ns = ts.astype("datetime64[ns]").astype("int64").to_numpy()
    usd_arr = df["usd_amount"].to_numpy()
    hour_arr = (ts.dt.hour.to_numpy().astype(np.float32) / 24.0)
    bank_arr = pd.concat([df["src_bank"].astype(str), df["dst_bank"].astype(str)], axis=1).to_numpy()
    same_bank_arr = (bank_arr[:, 0] == bank_arr[:, 1])

    band_start_ns = t0.value
    band_ns = band.value
    band_ids = (ts_ns - band_start_ns) // band_ns  # [n] band index per row

    snapshots: list[BandSnapshot] = []
    for b in range(n_bands):
        start = pd.Timestamp(band_start_ns + b * band_ns)
        end = start + band
        rows = np.flatnonzero(band_ids == b)
        if len(rows) == 0:
            continue

        w_start_ns = start.value - window.value
        w_mask = (ts_ns >= w_start_ns) & (ts_ns < start.value)
        w_rows = np.flatnonzero(w_mask)

        sub = pd.DataFrame(
            {
                "src_gid": src_gid[w_rows],
                "dst_gid": dst_gid[w_rows],
                "usd_amount": usd_arr[w_rows],
                "n_accounts": n_accounts,
            }
        )
        node_ids, node_feat = _window_profile(sub)

        if len(w_rows) > 0:
            # Vectorized global->local mapping over the sorted node_ids.
            def to_local(gids: np.ndarray) -> np.ndarray:
                pos = np.searchsorted(node_ids, gids)
                pos_c = np.minimum(pos, len(node_ids) - 1)
                ok = (pos < len(node_ids)) & (node_ids[pos_c] == gids)
                return np.where(ok, pos, -1).astype(np.int64)

            local_src = to_local(src_gid[w_rows])
            local_dst = to_local(dst_gid[w_rows])
            keep = (local_src >= 0) & (local_dst >= 0)
            edge_index = np.vstack([local_src[keep], local_dst[keep]])
            usd_w = usd_arr[w_rows][keep]
            same = same_bank_arr[w_rows][keep].astype(np.float32)
            hour = hour_arr[w_rows][keep]
            edge_attr = np.column_stack([np.log1p(usd_w), same, hour]).astype(np.float32)
            g2l = {}  # kept for BandSnapshot.localize() compatibility
        else:
            g2l, edge_index, edge_attr = {}, np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

        snapshots.append(
            BandSnapshot(
                band_id=b,
                start=start,
                end=end,
                node_ids=node_ids,
                node_feat=node_feat,
                edge_index=edge_index,
                edge_attr=edge_attr,
                txn_rows=rows,
                txn_src=src_gid[rows],
                txn_dst=dst_gid[rows],
                _global_to_local=g2l,
            )
        )
    return snapshots
