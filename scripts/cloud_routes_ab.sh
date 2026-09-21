#!/bin/bash
# Routes A + B in parallel: GPU (SSL pretrain) + CPU (counterfactual engine).
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
P=/root/miniconda3/bin/python

# Route A: SSL pretrain (GPU lane)
nohup $P scripts/ssl_pretrain.py > /root/routeA_ssl.log 2>&1 &

# Route B: counterfactual attribution (CPU lane, xgboost training inside)
nohup $P scripts/counterfactual_attribution.py > /root/routeB_cf.log 2>&1 &

echo "A(ssl, GPU) + B(counterfactual, CPU) launched"
