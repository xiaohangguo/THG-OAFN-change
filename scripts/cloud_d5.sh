#!/bin/bash
# D5: GATv2 strong profiles + scorer warmup (graph joins after 5 epochs).
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
P=/root/miniconda3/bin/python
nohup $P scripts/phase3_graph_baseline.py \
  --attach-entities --conv gatv2 --strong-profiles \
  --seeds 42 --max-epochs 100 --patience 8 --band-batches 4 --lr 1e-3 \
  --warmup-graph-epochs 5 \
  --tag d5-gatv2-warmup \
  > /root/d5_gatv2_warmup.log 2>&1 &
echo D5-launched
