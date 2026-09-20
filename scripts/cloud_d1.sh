#!/bin/bash
# D1: the decisive control — no-graph + identical new training loop.
# If this also lands ~0.02, the training loop (not the graph) broke it.
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
P=/root/miniconda3/bin/python
nohup $P scripts/phase3_graph_baseline.py \
  --attach-entities --conv gatv2 --no-graph \
  --seeds 42 --max-epochs 60 --patience 5 --band-batches 4 --lr 3e-3 \
  --tag diag-nograph-newloop \
  > /root/d1_nograph_newloop.log 2>&1 &
echo D1-launched
