#!/bin/bash
# Leakage forensics 2x2 on cloud CPU.
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
nohup /root/miniconda3/bin/python scripts/leakage_forensics.py > /root/forensics.log 2>&1 &
echo forensics-launched
