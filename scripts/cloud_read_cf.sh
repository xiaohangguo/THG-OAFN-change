#!/bin/bash
/root/miniconda3/bin/python - <<'PYEOF'
import json
d = json.load(open('/root/autodl-tmp/results/routeB_counterfactual/counterfactual_20260921T045213Z.json'))
print('surrogate:', d['surrogate_quality'])
print('cf_fidelity:', d['counterfactual_fidelity'])
print('drop:', d['score_drop_when_drivers_zeroed'])
c = d['cases'][0]
print('case0 base score:', round(c['score'], 4), '| drivers:', c['waterfall'][0]['drivers'][:5])
for w in c['waterfall']:
    print('  scale', w['scale'], '-> real_xgb', round(w['real_xgb_score'], 4), '| surrogate', round(w['surrogate_score'], 4))
PYEOF
