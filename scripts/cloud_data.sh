#!/bin/bash
# Cloud data pipeline: download 6 parts from Tsinghua cloud, join, extract, configure.
set -e
cd /root/autodl-tmp
mkdir -p parts data_cache/ibm_aml metadata results

cd parts
declare -A U=(
  [part_0]="https://cloud.tsinghua.edu.cn/f/4745dc27a6294621a345/?dl=1"
  [part_1]="https://cloud.tsinghua.edu.cn/f/9b4362e56f5547d4beff/?dl=1"
  [part_2]="https://cloud.tsinghua.edu.cn/f/54051b0530c44cb19a8c/?dl=1"
  [part_3]="https://cloud.tsinghua.edu.cn/f/2d625a9ce8c14506a60e/?dl=1"
  [part_4]="https://cloud.tsinghua.edu.cn/f/3e5afa311a1644819adc/?dl=1"
  [part_5]="https://cloud.tsinghua.edu.cn/f/6d9af39b2a9a4a79bf5c/?dl=1"
)
for p in part_0 part_1 part_2 part_3 part_4 part_5; do
  if [ ! -f "$p.done" ]; then
    echo "[get] $p"
    wget -c -q --show-progress -O "$p" "${U[$p]}" && touch "$p.done"
  fi
done
echo "--- sizes ---"
ls -l part_* | awk '{print $9, $5}'

cat part_0 part_1 part_2 part_3 part_4 part_5 > bundle.zip
TOTAL=$(stat -c%s bundle.zip)
echo "bundle: $TOTAL (expect 8176169418)"
if [ "$TOTAL" != "8176169418" ]; then echo "SIZE MISMATCH"; exit 1; fi

cd /root/autodl-tmp/THG-OAFN-change
python scripts/extract_hi_small.py --zip /root/autodl-tmp/bundle.zip --out /root/autodl-tmp/data_cache/ibm_aml --manifest /root/autodl-tmp/metadata/bundle_manifest.json

sed -i 's|D:/THG-OAFN-Financial-Experiment|/root/autodl-tmp|g' configs/ibm_aml_hi_small.yaml
echo "--- config paths now ---"
grep -E 'path|workspace' configs/ibm_aml_hi_small.yaml

python scripts/phase1_data_audit.py \
  --transactions /root/autodl-tmp/data_cache/ibm_aml/HI-Small_Trans.csv \
  --accounts /root/autodl-tmp/data_cache/ibm_aml/HI-Small_accounts.csv \
  --output /root/autodl-tmp/metadata/ibm_hi_small_audit.json 2>&1 | head -20
echo "=== PIPELINE DONE ==="
