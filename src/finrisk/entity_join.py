"""Join per-account legal-entity (Entity ID) from the accounts CSV.

The accounts file maps ``Account Number -> Entity ID``. A transaction's source
and destination accounts therefore inherit the legal entity that owns them.
Account numbers are normalized the same way on both sides (strip whitespace,
drop leading zeros) so the join is exact; an account absent from the accounts
file gets an empty entity (treated as an isolated node with no entity owner).

This module centralises the join so the table baselines and the graph builder
share one definition. It mutates ``df`` in place by adding two columns:
``src_entity`` and ``dst_entity``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _normalized(s: pd.Series) -> pd.Series:
    """Strip whitespace and drop leading zeros on account-number strings."""
    return s.astype(str).str.strip().str.lstrip("0").replace("", "0")


def load_entity_map(accounts_path: Path) -> dict[str, str]:
    """Return {normalized_account_number: Entity ID} for every account."""
    acc = pd.read_csv(accounts_path, usecols=["Account Number", "Entity ID"])
    acc["_acc"] = _normalized(acc["Account Number"])
    return dict(zip(acc["_acc"], acc["Entity ID"]))


def attach_entities(df: pd.DataFrame, accounts_path: Path) -> pd.DataFrame:
    """Add ``src_entity``/``dst_entity`` columns to ``df`` and return it.

    ``df`` must already carry ``src_account`` and ``dst_account`` columns.
    A copy is returned; the input frame is left untouched.
    """
    mapping = load_entity_map(accounts_path)
    out = df.copy()
    out["src_entity"] = _normalized(out["src_account"]).map(mapping).fillna("").astype(str)
    out["dst_entity"] = _normalized(out["dst_account"]).map(mapping).fillna("").astype(str)
    return out