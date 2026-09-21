#!/bin/bash
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
nohup /root/miniconda3/bin/python scripts/counterfactual_v2.py > /root/routeB_v2.log 2>&1 &
echo B-v2-launched
