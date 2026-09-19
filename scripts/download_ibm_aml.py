"""Download the three IBM AML HI-Small files into the local D-drive workspace.

Downloads only the HI-Small whitelist (~510 MB total); never the ~42 GB
HI-Large package. Kaggle credentials come from %USERPROFILE%\\.kaggle\\kaggle.json
or the KAGGLE_USERNAME / KAGGLE_KEY environment variables.

Example:
    python scripts/download_ibm_aml.py
    python scripts/download_ibm_aml.py --output-dir D:/THG-OAFN-Financial-Experiment/data_cache/ibm_aml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub

DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"

# Whitelist: anything else in the remote dataset (notably HI-Large, ~42 GB) is
# intentionally out of reach for this script.
REQUIRED_FILES = (
    "HI-Small_Trans.csv",
    "HI-Small_accounts.csv",
    "HI-Small_Patterns.txt",
)

MIN_EXPECTED_BYTES = {
    "HI-Small_Trans.csv": 400_000_000,      # ~476 MB
    "HI-Small_accounts.csv": 10_000_000,    # ~34 MB
    "HI-Small_Patterns.txt": 1,             # small text file
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download IBM AML HI-Small files to the D-drive workspace.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("D:/THG-OAFN-Financial-Experiment/data_cache/ibm_aml"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in REQUIRED_FILES:
        target = args.output_dir / file_name
        if target.is_file() and target.stat().st_size >= MIN_EXPECTED_BYTES[file_name]:
            print(f"[skip] {file_name} already present ({target.stat().st_size:,} bytes)")
            continue
        print(f"[get ] {file_name} ...")
        saved = kagglehub.dataset_download(DATASET, path=file_name, output_dir=str(args.output_dir))
        saved_path = Path(saved)
        if not saved_path.is_file() or saved_path.stat().st_size < MIN_EXPECTED_BYTES[file_name]:
            print(f"[fail] {file_name}: unexpected download result ({saved})")
            return 1
        print(f"[done] {file_name} -> {saved_path} ({saved_path.stat().st_size:,} bytes)")

    print("\nAll HI-Small files are in place. Next step:")
    print("  python scripts/phase1_data_audit.py "
          "--transactions D:/THG-OAFN-Financial-Experiment/data_cache/ibm_aml/HI-Small_Trans.csv "
          "--accounts D:/THG-OAFN-Financial-Experiment/data_cache/ibm_aml/HI-Small_accounts.csv "
          "--output D:/THG-OAFN-Financial-Experiment/metadata/ibm_hi_small_audit.json --hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
