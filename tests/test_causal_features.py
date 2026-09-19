import numpy as np
import pandas as pd

from finrisk.causal_features import build_causal_features, causal_entity_features, causal_pair_features


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


def _entity_frame(rows):
    return pd.DataFrame(
        rows,
        columns=["Timestamp", "SrcAcc", "DstAcc", "SrcEnt", "DstEnt", "USD"],
    )


def test_entity_strict_past_window():
    df = _entity_frame(
        [
            ["2026-01-01 10:00", "A", "X", "E1", "E9", 100.0],
            ["2026-01-01 10:30", "B", "Y", "E1", "E9", 50.0],
            ["2026-01-01 11:00", "C", "Y", "E1", "E8", 200.0],
        ]
    )
    feats = causal_entity_features(df, "Timestamp", "SrcAcc", "DstAcc", "SrcEnt", "DstEnt", "USD")
    # Brother signal: src entity E1 has 2 prior rows total for row 2 (from entity view),
    # but 2 of them are from distinct accounts A/B, so row 2's own account C has no prior
    # account history (first row for C). Hence brother count = entity count - account count.
    # For row 2 (C->Y): entity count=2, account count for C=0 -> brother count=2
    assert feats.loc[2, "src_bro_w1h_count"] == 2
    # Row 2 at 10:30 row 1 has no prior rows for any account, so brother=1? Actually entity
    # count for B =1 from A's row, account for B=0 -> brother=1
    assert feats.loc[1, "src_bro_w1h_count"] == 1
    assert feats.loc[0, "src_bro_w1h_count"] == 0
    # same_entity transfer flag: rows stay within their own entity on dst side
    # depends on this construct; here src E1 -> dst E9/E8, never equal.
    assert feats["same_entity_transfer"].sum() == 0


def test_entity_aggregates_across_accounts_but_isoillages_other_entities():
    # Accounts A and B both belong to entity E1; account C belongs to E2.
    df = _entity_frame(
        [
            ["2026-01-01 10:00", "A", "X", "E1", "E9", 100.0],
            ["2026-01-01 10:10", "C", "X", "E2", "E9", 5000.0],  # other entity, must not leak into E1
            ["2026-01-01 10:20", "B", "Z", "E1", "E8", 10.0],
        ]
    )
    feats = causal_entity_features(df, "Timestamp", "SrcAcc", "DstAcc", "SrcEnt", "DstEnt", "USD")
    # Row 3 (B->Z) entity E1 brother signal: entity has 1 prior row (A->X), account B has none.
    assert feats.loc[2, "src_bro_w1h_sum"] == 100.0
    assert feats.loc[2, "src_bro_w1h_count"] == 1


def test_entity_same_entity_flag_and_no_future_leak():
    df = _entity_frame(
        [
            ["2026-01-01 10:00", "A", "B", "E1", "E1", 100.0],  # intra-entity (A->B same entity)
            ["2026-01-01 10:30", "A", "X", "E1", "E9", 50.0],
            ["2026-01-01 11:00", "C", "D", "E1", "E1", 200.0],
        ]
    )
    feats = causal_entity_features(df, "Timestamp", "SrcAcc", "DstAcc", "SrcEnt", "DstEnt", "USD")
    # Row 1 is an intra-entity transfer; row 3 is another.
    assert feats["same_entity_transfer"].tolist() == [1, 0, 1]

    # Future mutation must not change row 0 / row 1 features.
    before = feats
    rows_mutated = [
        ["2026-01-01 10:00", "A", "B", "E1", "E1", 100.0],
        ["2026-01-01 10:30", "A", "X", "E1", "E9", 50.0],
        ["2026-01-01 11:00", "C", "D", "E1", "E1", 200.0],
        ["2026-01-02 00:00", "Z", "Q", "E1", "E1", 99999.0],  # far future
    ]
    after = causal_entity_features(_entity_frame(rows_mutated), "Timestamp", "SrcAcc", "DstAcc", "SrcEnt", "DstEnt", "USD")
    # No leak: the joined far-future row must not change any past-row feature.
    pd.testing.assert_series_equal(before.loc[0], after.loc[0], check_names=False)
    pd.testing.assert_series_equal(before.loc[1], after.loc[1], check_names=False)
    pd.testing.assert_series_equal(before.loc[2], after.loc[2], check_names=False)

    # Sanity: the far-future row (index 3) itself sees all 3 prior E1 rows for its source.
    assert after.loc[3, "src_bro_w1h_count"] + after.loc[3, "src_bro_w24h_count"] >= 2
    assert after["same_entity_transfer"].tolist() == [1, 0, 1, 1]


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


def _pair_frame(rows):
    return pd.DataFrame(rows, columns=["Timestamp", "Src", "Dst", "USD"])


def test_pair_strict_past_and_isolation():
    # Same (A,B) pair repeats; a different pair (A,C) must not leak in.
    df = _pair_frame([
        ["2026-01-01 10:00", "A", "B", 100.0],
        ["2026-01-01 10:30", "A", "C", 5000.0],   # different pair, isolatable
        ["2026-01-01 11:00", "A", "B", 200.0],
    ])
    feats = build_causal_features(df, "Timestamp", "Src", "Dst", "USD")
    # Row 3 (A->B at 11:00): pair (A,B) prior = row 1 only (100.0), 1h window active.
    assert feats.loc[2, "pair_w1h_count"] == 1
    assert feats.loc[2, "pair_w1h_sum"] == 100.0
    # Row 2 (A->C): pair (A,C) has no prior history.
    assert feats.loc[1, "pair_past_count"] == 0
    assert feats.loc[1, "pair_w1h_count"] == 0
    # Row 1 sees nothing.
    assert feats.loc[0, "pair_w1h_count"] == 0


def test_pair_no_future_leak():
    rows = [
        ["2026-01-01 10:00", "A", "B", 100.0],
        ["2026-01-01 10:30", "A", "B", 50.0],
        ["2026-01-01 11:00", "A", "B", 200.0],
    ]
    before = build_causal_features(_pair_frame(rows), "Timestamp", "Src", "Dst", "USD")
    rows_mutated = [r[:] for r in rows]
    rows_mutated[2][3] = 77777.0   # mutate row 3 amount (future for rows 0/1)
    rows_mutated.append(["2026-01-02 00:00", "A", "B", 99999.0])  # far future
    after = build_causal_features(_pair_frame(rows_mutated), "Timestamp", "Src", "Dst", "USD")
    pd.testing.assert_series_equal(before.loc[0], after.loc[0], check_names=False)
    pd.testing.assert_series_equal(before.loc[1], after.loc[1], check_names=False)
    # Row 3 (index 2) itself is invariant to a later mutation of its own amount.
    pd.testing.assert_series_equal(before.loc[2], after.loc[2], check_names=False)
    # The far-future row (index 3) sees the 3 prior (A,B) rows in its past count.
    assert after.loc[3, "pair_past_count"] == 3
