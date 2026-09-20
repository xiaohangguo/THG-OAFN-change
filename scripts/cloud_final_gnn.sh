#!/bin/bash
# The final GNN shot: literature form (strong 20-dim behavioural node profiles
# + GATv2 attention), full training loop. GPU-exclusive lane.
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
nohup /root/miniconda3/bin/python scripts/phase3_graph_baseline.py \
  --attach-entities --conv gatv2 --strong-profiles \
  --seeds 42 --max-epochs 60 --patience 5 --band-batches 4 --lr 3e-3 \
  --hidden 64 --tag gatv2-strongfeat \
  > /root/exp_gatv2_strong.log 2>&1 &
echo final-gnn-shot-launched
