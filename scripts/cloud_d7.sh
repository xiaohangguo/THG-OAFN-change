#!/bin/bash
# D7: embedding probe — frozen GNN embeddings vs 78-col XGBoost A/B.
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
nohup /root/miniconda3/bin/python scripts/embedding_probe.py > /root/d7_emb_probe.log 2>&1 &
echo D7-launched
