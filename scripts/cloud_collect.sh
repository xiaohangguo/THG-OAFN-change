#!/bin/bash
echo "=== gatv2-strongfeat 终审 ==="
grep -E "seed 42" /root/exp_gatv2_strong.log
python3 - <<'PYEOF' 2>/dev/null || /root/miniconda3/bin/python - <<'PYEOF2'
PYEOF
import json, glob
p = sorted(glob.glob("/root/autodl-tmp/results/phase3/gatv2-strongfeat_*_metrics.json"))[-1]
d = json.load(open(p))
t = d["test"]
for k in ("auprc", "roc_auc", "recall_at_0_001", "recall_at_0_01"):
    print(k, round(t[k]["mean"], 4))
PYEOF2
echo "=== 5-seed 逐种子 ==="
grep -E "\[xgboost\] seed" /root/exp_xgb5.log
echo "=== 5-seed 聚合 ==="
/root/miniconda3/bin/python - <<'PYEOF3'
import json, glob
p = sorted(glob.glob("/root/autodl-tmp/results/phase2/multi_model_*_metrics.json"))[-1]
d = json.load(open(p))
for m, v in d["models"].items():
    print(m, "auprc", round(v["test"]["auprc"]["mean"], 4), "+/-", round(v["test"]["auprc"]["std"], 4))
PYEOF3
