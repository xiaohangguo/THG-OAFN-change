#!/bin/bash
# CPU battery rerun: winner 5-seed + same-feature controls + SHAP attribution.
cd /root/autodl-tmp/THG-OAFN-change
P=/root/miniconda3/bin/python
nohup bash -c "
  $P scripts/phase2_baseline.py --attach-entities --with-profiles --models xgboost --seeds 42 123 456 789 2024 > /root/exp_xgb5.log 2>&1
  $P scripts/phase2_baseline.py --attach-entities --with-profiles --models lightgbm mlp --seeds 42 > /root/exp_lgbm_mlp.log 2>&1
  $P scripts/attribution_report.py --attach-entities --with-profiles > /root/exp_shap.log 2>&1
  echo DONE > /root/cpu_queue.done
" > /dev/null 2>&1 &
echo cpu-battery-relaunched
