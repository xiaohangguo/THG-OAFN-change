"""Extract only the three HI-Small files from the Kaggle bundle zip.

The bundle zip is the immutable snapshot of Kaggle dataset version; this
script lists all zip members (for the data card), then extracts the HI-Small
whitelist without unpacking the multi-GB HI-Medium/Large members.

Example:
    python scripts/extract_hi_small.py \
        --zip "E:/THG-OAFN-archive/kaggle_ibm_aml_v12411329.zip" \
        --out "D:/THG-OAFN-Financial-Experiment/data_cache/ibm_aml"
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

WHITELIST = ("HI-Small_Trans.csv", "HI-Small_accounts.csv", "HI-Small_Patterns.txt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract HI-Small whitelist files from the Kaggle bundle zip.")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("D:/THG-OAFN-Financial-Experiment/data_cache/ibm_aml"))
    parser.add_argument("--manifest", type=Path, default=Path("D:/THG-OAFN-Financial-Experiment/metadata/bundle_manifest.json"))
    args = parser.parse_args()

    if not args.zip.is_file():
        parser.error(f"Zip not found: {args.zip}")
    args.out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as bundle:
        members = bundle.namelist()

        # Full member listing goes to the manifest: evidence of exactly what
        # Kaggle version snapshot contains, without unpacking anything.
        manifest = {
            "zip_path": str(args.zip.resolve()),
            "zip_bytes": args.zip.stat().st_size,
            "listed_utc": datetime.now(timezone.utc).isoformat(),
            "members": [{"name": name, "bytes": info.file_size} for name, info in zip(members, bundle.infolist())],
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[manifest] {len(members)} members listed -> {args.manifest}")

        for wanted in WHITELIST:
            # Tolerate a folder prefix inside the zip (e.g. "archive/HI-Small_...").
            matches = [name for name in members if name.endswith(wanted) and not name.endswith("/")]
            if not matches:
                print(f"[fail] {wanted} not found in zip; members were:")
                for name in members:
                    print(f"        {name}")
                return 1
            source = matches[0]
            target = args.out / wanted
            if target.is_file() and target.stat().st_size > 0:
                print(f"[skip] {wanted} already present ({target.stat().st_size:,} bytes)")
                continue
            bundle.extract(source, args.out)
            if source != wanted:
                (args.out / source).rename(target)
            print(f"[done] {source} -> {target} ({target.stat().st_size:,} bytes)")

    print("\nNext step: run scripts/phase1_data_audit.py with --hash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
