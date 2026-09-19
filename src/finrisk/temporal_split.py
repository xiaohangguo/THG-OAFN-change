"""Chronological train/validation/test split with whole-timestamp boundaries.

The split is defined by transaction time only: rows sharing one timestamp can
never land in different subsets, and every subset occupies a strictly earlier
time range than the next. Random seeds must never move this split.
"""

from __future__ import annotations

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
) -> pd.Series:
    """Return a Series of 'train'/'validation'/'test' aligned to df.index.

    Boundary timestamps are chosen so cumulative row counts reach the target
    ratios; all rows of a boundary timestamp move together with the earlier
    subset, keeping subsets disjoint in time.
    """
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("ratios must satisfy 0 < train, validation and train+validation < 1")

    times = pd.to_datetime(df[timestamp_col], errors="raise")
    counts = times.value_counts().sort_index()
    cumulative = counts.cumsum()
    total = int(cumulative.iloc[-1])

    train_cut = counts.index[cumulative >= total * train_ratio]
    train_boundary = train_cut[0] if len(train_cut) else counts.index[-1]
    validation_cut = counts.index[cumulative >= total * (train_ratio + validation_ratio)]
    validation_boundary = validation_cut[0] if len(validation_cut) else counts.index[-1]

    split = pd.Series("train", index=df.index, dtype="string")
    split[times > train_boundary] = "validation"
    split[times > validation_boundary] = "test"
    return split


def assert_split_properties(df: pd.DataFrame, timestamp_col: str, split: pd.Series) -> None:
    """Fail loudly if the split violates chronology or timestamp integrity."""
    times = pd.to_datetime(df[timestamp_col])
    parts = {}
    for name in ("train", "validation", "test"):
        mask = split == name
        if not mask.any():
            raise AssertionError(f"empty split: {name}")
        parts[name] = (times[mask].min(), times[mask].max())
    if not (parts["train"][1] < parts["validation"][0] <= parts["validation"][1] < parts["test"][0]):
        raise AssertionError(f"overlapping split ranges: {parts}")
    # Every timestamp must belong to exactly one subset.
    grouped = pd.DataFrame({"t": times.values, "s": split.values}).groupby("t")["s"].nunique()
    if (grouped > 1).any():
        raise AssertionError("a timestamp appears in more than one subset")
