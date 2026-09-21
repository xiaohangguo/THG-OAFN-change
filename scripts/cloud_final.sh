#!/bin/bash
# Final metric (threshold-locked F1), then collect everything, then shutdown.
cd /root/autodl-tmp/THG-OAFN-change
git pull -q
/root/miniconda3/bin/python scripts/threshold_metrics.py > /root/threshold.log 2>&1
echo "exit=$?" >> /root/threshold.log
# Bundle every result for local archive
cd /root/autodl-tmp && tar czf /root/results_final.tar.gz results metadata
ls -la /root/results_final.tar.gz
