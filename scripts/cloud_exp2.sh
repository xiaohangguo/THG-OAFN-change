#!/bin/bash
# Hyperparameter search on cloud CPU.
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
nohup /root/miniconda3/bin/python scripts/hparam_search.py --attach-entities --with-profiles \
  > /root/exp_hparam.log 2>&1 &
echo hparam-started
