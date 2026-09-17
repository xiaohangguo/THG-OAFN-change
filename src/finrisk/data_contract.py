"""Schema contract for IBM AML transaction data.

The contract is deliberately strict: a missing time, source, destination or label
column means the experiment must stop before any graph or feature is generated.
"""

from __future__ import annotations

from collections.abc import Iterable


CANONICAL_ALIASES = {
    "timestamp": ("Timestamp", "timestamp", "time", "datetime"),
    "source_bank": ("From Bank", "from_bank", "source_bank"),
    "source_account": ("Account", "from_account", "source_account"),
    "destination_bank": ("To Bank", "to_bank", "destination_bank"),
    "destination_account": ("Account.1", "to_account", "destination_account"),
    "label": ("Is Laundering", "is_laundering", "label", "target"),
}


class SchemaError(ValueError):
    """Raised when a required, auditable transaction field is absent."""


def resolve_transaction_columns(columns: Iterable[str]) -> dict[str, str]:
    """Resolve required canonical names to source CSV column names."""
    available = {str(column).strip(): str(column).strip() for column in columns}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in CANONICAL_ALIASES.items():
        source_name = next((alias for alias in aliases if alias in available), None)
        if source_name is None:
            missing.append(f"{canonical} ({', '.join(aliases)})")
        else:
            resolved[canonical] = source_name
    if missing:
        raise SchemaError("Missing required transaction columns: " + "; ".join(missing))
    return resolved


def laundering_mask(values):
    """Return a boolean mask while accepting common 0/1 and true/false encodings."""
    normalized = values.astype(str).str.strip().str.lower()
    valid = {"0", "1", "false", "true", "no", "yes"}
    unknown = sorted(set(normalized.unique()) - valid)
    if unknown:
        raise SchemaError(f"Unsupported laundering label values: {unknown[:10]}")
    return normalized.isin({"1", "true", "yes"})
