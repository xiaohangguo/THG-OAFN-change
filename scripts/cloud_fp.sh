#!/bin/bash
set -e
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
/root/miniconda3/bin/python - <<'PYEOF'
import pandas
print("cloud pandas:", pandas.__version__)
PYEOF
nohup /root/miniconda3/bin/python scripts/feature_fingerprint.py > /root/fp_cloud2.txt 2>&1 &
echo fingerprint-bg-started
