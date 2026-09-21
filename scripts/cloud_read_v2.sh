#!/bin/bash
/root/miniconda3/bin/python - <<'PYEOF'
import json, glob
p = sorted(glob.glob('/root/autodl-tmp/results/routeB_counterfactual/counterfactual_v2_*.json'))[-1]
d = json.load(open(p))
print('=== 组级干预摘要（20 个真阳性案例，分数从 ~0.999 起） ===')
for g, v in sorted(d['group_interventions_summary'].items(), key=lambda kv: -kv[1]['mean_drop']):
    print(f"  {g:<18} 平均降幅 {v['mean_drop']:.3f}   最大 {v['max_drop']:.3f}")
c = d['cases'][0]
print('case0 local SHAP top5:', [(f, round(v,2)) for f, v in c['local_shap_top5']])
PYEOF
