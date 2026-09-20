#!/bin/bash
# Fingerprint all cloud parts, then rebuild bundle in the correct order.
cd /root/autodl-tmp/parts
echo "=== cloud fingerprints ==="
for f in part_0 part_1 part_2 part_3 part_4 part_5; do
  head_hex=$(head -c4 "$f" | od -An -tx1 | tr -d ' \n')
  fp=$(head -c1048576 "$f" | md5sum | cut -c1-16)
  echo "$f | $head_hex | $fp"
done

# Correct order resolved from local fingerprints:
# local part_0 F8316F83541FC98C, part_1 09741803D8604670, part_2 C58183857F533D26,
# part_3 1F77150133843DAA, part_4 CCA8855B04A9D61B, part_5 8639ED701B6BE7F4
declare -A ORDER
for f in part_0 part_1 part_2 part_3 part_4 part_5; do
  fp=$(head -c1048576 "$f" | md5sum | cut -c1-16 | tr "a-f" "A-F")
  case "$fp" in
    F8316F83541FC98C) ORDER[0]=$f ;;
    09741803D8604670) ORDER[1]=$f ;;
    C58183857F533D26) ORDER[2]=$f ;;
    1F77150133843DAA) ORDER[3]=$f ;;
    CCA8855B04A9D61B) ORDER[4]=$f ;;
    8639ED701B6BE7F4) ORDER[5]=$f ;;
    *) echo "UNKNOWN fingerprint $fp for $f"; exit 1 ;;
  esac
done
echo "resolved order: ${ORDER[0]} ${ORDER[1]} ${ORDER[2]} ${ORDER[3]} ${ORDER[4]} ${ORDER[5]}"
cat "${ORDER[0]}" "${ORDER[1]}" "${ORDER[2]}" "${ORDER[3]}" "${ORDER[4]}" "${ORDER[5]}" > /root/autodl-tmp/bundle.zip
TOTAL=$(stat -c%s /root/autodl-tmp/bundle.zip)
echo "bundle rebuilt: $TOTAL"
[ "$TOTAL" = "8176169418" ] || { echo "SIZE MISMATCH"; exit 1; }

cd /root/autodl-tmp/THG-OAFN-change
/root/miniconda3/bin/python scripts/extract_hi_small.py \
  --zip /root/autodl-tmp/bundle.zip \
  --out /root/autodl-tmp/data_cache/ibm_aml \
  --manifest /root/autodl-tmp/metadata/bundle_manifest.json
echo "=== EXTRACT OK ==="
