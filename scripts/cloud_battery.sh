#!/bin/bash
# Full-blast parallel battery:
#   GPU: GATv2 full training (fixed loop, 60 epochs, minibatches, cosine lr)
#   CPU queue A: regularized XGBoost 5-seed final + LightGBM/MLP on 78 cols
#   CPU queue B (after A): SHAP attribution on winner feature set
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
P=/root/miniconda3/bin/python

# GPU lane
nohup $P scripts/phase3_graph_baseline.py \
  --attach-entities --conv gatv2 --seeds 42 --max-epochs 60 --patience 5 \
  --band-batches 4 --lr 3e-3 --hidden 64 --tag gatv2-full \
  > /root/exp_gatv2_full.log 2>&1 &

# CPU lane (sequential inside)
nohup bash -c "
  $P scripts/phase2_baseline.py --attach-entities --with-profiles --models xgboost --seeds 42 123 456 789 2024 > /root/exp_xgb5.log 2>&1
  $P scripts/phase2_baseline.py --attach-entities --with-profiles --models lightgbm mlp --seeds 42 > /root/exp_lgbm_mlp.log 2>&1
  $P scripts/attribution_report.py --attach-entities --with-profiles > /root/exp_shap.log 2>&1
  echo ALL-CPU-DONE > /root/cpu_queue.done
" > /dev/null 2>&1 &

echo "GPU lane: gatv2-full launched"
echo "CPU lane: 5-seed + lgbm/mlp + shap queued"
