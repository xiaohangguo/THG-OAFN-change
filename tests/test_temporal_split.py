import pandas as pd
import pytest

from finrisk.temporal_split import assert_split_properties, temporal_split


def _frame(times):
    return pd.DataFrame({"Timestamp": pd.to_datetime(times)})


def test_split_ratios_approximate_target():
    times = pd.date_range("2026-01-01", periods=1000, freq="h")
    df = _frame(times)
    split = temporal_split(df, "Timestamp")
    shares = split.value_counts(normalize=True)
    assert abs(shares["train"] - 0.60) < 0.02
    assert abs(shares["validation"] - 0.20) < 0.02
    assert abs(shares["test"] - 0.20) < 0.02


def test_same_timestamp_never_spans_subsets():
    base = list(pd.date_range("2026-01-01", periods=300, freq="h"))
    duplicated = base + [base[179]] * 5  # five extra rows on one busy timestamp
    df = _frame(duplicated)
    split = temporal_split(df, "Timestamp")
    assert_split_properties(df, "Timestamp", split)


def test_boundaries_move_with_whole_timestamp():
    df = _frame(pd.date_range("2026-01-01", periods=100, freq="h"))
    split = temporal_split(df, "Timestamp")
    assert_split_properties(df, "Timestamp", split)
    assert (split == "train").sum() > 0
    assert (split == "validation").sum() > 0
    assert (split == "test").sum() > 0


def test_invalid_ratios_rejected():
    df = _frame(pd.date_range("2026-01-01", periods=10, freq="h"))
    with pytest.raises(ValueError):
        temporal_split(df, "Timestamp", train_ratio=0.0, validation_ratio=0.2)
    with pytest.raises(ValueError):
        temporal_split(df, "Timestamp", train_ratio=0.8, validation_ratio=0.3)
