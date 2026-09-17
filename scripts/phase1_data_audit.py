"""Stream-audit IBM AML transaction files without building a graph or loading all rows.

Example:
    python scripts/phase1_data_audit.py --transactions D:\\...\\HI-Small_Trans.csv \
        --accounts D:\\...\\HI-Small_accounts.csv --output D:\\...\\audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk.data_contract import SchemaError, laundering_mask, resolve_transaction_columns


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_transactions(path: Path, chunksize: int) -> dict:
    header = pd.read_csv(path, nrows=0)
    mapping = resolve_transaction_columns(header.columns)
    totals = {"rows": 0, "positive_rows": 0, "invalid_timestamp_rows": 0}
    first_timestamp = None
    last_timestamp = None
    accounts: set[str] = set()
    banks: set[str] = set()
    payment_formats: set[str] = set()

    for chunk in pd.read_csv(path, chunksize=chunksize):
        totals["rows"] += len(chunk)
        labels = laundering_mask(chunk[mapping["label"]])
        totals["positive_rows"] += int(labels.sum())

        timestamps = pd.to_datetime(chunk[mapping["timestamp"]], errors="coerce")
        totals["invalid_timestamp_rows"] += int(timestamps.isna().sum())
        valid_timestamps = timestamps.dropna()
        if not valid_timestamps.empty:
            chunk_min, chunk_max = valid_timestamps.min(), valid_timestamps.max()
            first_timestamp = chunk_min if first_timestamp is None else min(first_timestamp, chunk_min)
            last_timestamp = chunk_max if last_timestamp is None else max(last_timestamp, chunk_max)

        accounts.update(chunk[mapping["source_account"]].dropna().astype(str))
        accounts.update(chunk[mapping["destination_account"]].dropna().astype(str))
        banks.update(chunk[mapping["source_bank"]].dropna().astype(str))
        banks.update(chunk[mapping["destination_bank"]].dropna().astype(str))
        if "Payment Format" in chunk.columns:
            payment_formats.update(chunk["Payment Format"].dropna().astype(str))

    if totals["rows"] == 0:
        raise SchemaError("The transaction file has no data rows.")
    if totals["invalid_timestamp_rows"]:
        raise SchemaError(
            f"Found {totals['invalid_timestamp_rows']} invalid timestamps; do not continue with a temporal split."
        )

    return {
        "column_mapping": mapping,
        "rows": totals["rows"],
        "positive_rows": totals["positive_rows"],
        "positive_rate": totals["positive_rows"] / totals["rows"],
        "timestamp_min": first_timestamp.isoformat(),
        "timestamp_max": last_timestamp.isoformat(),
        "unique_accounts": len(accounts),
        "unique_banks": len(banks),
        "payment_formats": sorted(payment_formats),
    }


def file_metadata(path: Path, include_hash: bool) -> dict:
    stat = path.stat()
    metadata = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if include_hash:
        metadata["sha256"] = sha256(path)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an IBM AML transaction CSV in chunks.")
    parser.add_argument("--transactions", type=Path, required=True)
    parser.add_argument("--accounts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--hash", action="store_true", help="Compute a full SHA-256; this reads the file once more.")
    args = parser.parse_args()

    if not args.transactions.is_file():
        parser.error(f"Transaction file not found: {args.transactions}")
    if args.accounts and not args.accounts.is_file():
        parser.error(f"Account file not found: {args.accounts}")
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")

    transaction_audit = audit_transactions(args.transactions, args.chunksize)
    output = {
        "audit_version": "phase1-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transactions": {**file_metadata(args.transactions, args.hash), **transaction_audit},
        "accounts": file_metadata(args.accounts, args.hash) if args.accounts else None,
        "temporal_readiness": "pass",
        "next_gate": "Phase 2 may begin only after this audit and raw-data licence record are reviewed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SchemaError as error:
        print(f"AUDIT FAILED: {error}", file=sys.stderr)
        raise SystemExit(2)
