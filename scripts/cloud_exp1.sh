#!/bin/bash
# Sprint experiment 1: XGBoost reproduction on cloud (seed 42, 78 features).
cd /root/autodl-tmp/THG-OAFN-change
/root/miniconda3/bin/python scripts/phase2_baseline.py \
  --attach-entities --with-profiles --models xgboost --seeds 42 \
  > /root/exp_xgb_repro.log 2>&1
echo "exit=$?" >> /root/exp_xgb_repro.log
