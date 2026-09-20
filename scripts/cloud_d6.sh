#!/bin/bash
# D6: clean warmup — eval mode matches training mode during warmup.
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
P=/root/miniconda3/bin/python
nohup $P scripts/phase3_graph_baseline.py \
  --attach-entities --conv gatv2 --strong-profiles \
  --seeds 42 --max-epochs 100 --patience 8 --band-batches 4 --lr 1e-3 \
  --warmup-graph-epochs 5 \
  --tag d6-gatv2-cleanwarmup \
  > /root/d6_gatv2_cleanwarmup.log 2>&1 &
echo D6-launched
