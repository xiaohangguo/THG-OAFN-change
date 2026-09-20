#!/bin/bash
/root/miniconda3/bin/python - <<'PYEOF'
import json, glob

# SHAP attribution summary (78-col winner model)
p = sorted(glob.glob("/root/autodl-tmp/results/attribution/xgb_shap_*.json"))[-1]
d = json.load(open(p))
print("=== SHAP 78列归因 top12 ===")
for r in d["global_ranking_top20"][:12]:
    print(f"{r['feature']:<28} {r['mean_abs_shap']:.3f}")
f = d["fidelity"]
print("fidelity: before", round(f["flagged_score_before"], 3),
      "after", f["flagged_score_after"], "drop", f"{f['score_drop_ratio']:.5f}")
PYEOF
