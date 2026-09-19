import numpy as np
import pandas as pd

from finrisk.causal_features import build_causal_features


def _txn_frame(rows):
    return pd.DataFrame(
        rows,
        columns=["Timestamp", "Account", "Account.1", "USD"],
    )


def test_hand_computed_window_statistics():
    df = _txn_frame(
        [
            ["2026-01-01 10:00", "A", "X", 100.0],
            ["2026-01-01 10:30", "A", "Y", 50.0],
            ["2026-01-01 11:00", "A", "Z", 200.0],
        ]
    )
    feats = build_causal_features(df, "Timestamp", "Account", "Account.1", "USD")
    # Row 3 at 11:00, 1h window [10:00, 11:00): sees rows 1 and 2 only.
    assert feats.loc[2, "src_w1h_count"] == 2
    assert feats.loc[2, "src_w1h_sum"] == 150.0
    # Row 2 at 10:30 sees only row 1.
    assert feats.loc[1, "src_w1h_count"] == 1
    assert feats.loc[1, "src_w1h_sum"] == 100.0
    # Row 1 sees nothing.
    assert feats.loc[0, "src_w1h_count"] == 0
    assert feats.loc[0, "src_w1h_sum"] == 0.0


def test_concurrent_transactions_are_invisible_to_each_other():
    df = _txn_frame(
        [
            ["2026-01-01 10:00", "A", "X", 100.0],
            ["2026-01-01 10:00", "A", "Y", 500.0],
            ["2026-01-01 12:00", "A", "Z", 1.0],
        ]
    )
    feats = build_causal_features(df, "Timestamp", "Account", "Account.1", "USD")
    # Both 10:00 rows must see an empty 1h history.
    assert feats.loc[0, "src_w1h_count"] == 0
    assert feats.loc[1, "src_w1h_count"] == 0
    # The 12:00 row: 1h window is [11:00, 12:00) -> empty; 24h sees both.
    assert feats.loc[2, "src_w1h_count"] == 0
    assert feats.loc[2, "src_w24h_count"] == 2
    assert feats.loc[2, "src_w24h_sum"] == 600.0


def test_future_mutation_never_changes_past_features():
    rows = [
        ["2026-01-01 10:00", "A", "X", 100.0],
        ["2026-01-01 10:30", "A", "Y", 50.0],
        ["2026-01-01 11:00", "A", "Z", 200.0],
        ["2026-01-02 09:00", "B", "A", 999.0],
    ]
    before = build_causal_features(_txn_frame(rows), "Timestamp", "Account", "Account.1", "USD")

    # Corrupt the future of rows 0/1: mutate row 2's amount and insert a
    # transaction at 10:45 (future for rows 0-1, past for row 2).
    rows_mutated = [r[:] for r in rows]
    rows_mutated[2][3] = 77777.0
    rows_mutated.append(["2026-01-01 10:45", "A", "Q", 424242.0])
    after = build_causal_features(_txn_frame(rows_mutated), "Timestamp", "Account", "Account.1", "USD")

    # Row 0 is unchanged in both frames; row 1 must not see the 10:15 insert
    # either (that is a future transaction relative to nothing — it sits
    # between row 1 and row 2, so only row 2+ may change).
    pd.testing.assert_series_equal(before.loc[0], after.loc[0], check_names=False)
    pd.testing.assert_series_equal(before.loc[1], after.loc[1], check_names=False)
    # And the mutation did flow through somewhere (sanity, not vacuous).
    assert after.loc[2, "src_w1h_sum"] != before.loc[2, "src_w1h_sum"]


def test_accounts_are_isolated():
    df = _txn_frame(
        [
            ["2026-01-01 10:00", "A", "X", 100.0],
            ["2026-01-01 10:10", "B", "Y", 5000.0],
            ["2026-01-01 10:20", "A", "Z", 10.0],
        ]
    )
    feats = build_causal_features(df, "Timestamp", "Account", "Account.1", "USD")
    # B's large transfer must not leak into A's history windows.
    assert feats.loc[2, "src_w1h_sum"] == 100.0
    # Received side of account A: row 3 is received by A? No — destination is
    # Z. X receives row 1 only; verify destination-side windows independently.
    assert feats.loc[0, "dst_w1h_count"] == 0
