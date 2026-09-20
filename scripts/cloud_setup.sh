#!/bin/bash
# Cloud setup: clone repo + install deps + verify.
source /etc/network_turbo
cd /root/autodl-tmp
if [ ! -d THG-OAFN-change ]; then
  git clone -q https://github.com/xiaohangguo/THG-OAFN-change.git
fi
cd THG-OAFN-change
git pull -q
pip install -q pandas scikit-learn lightgbm xgboost shap pyyaml pyarrow torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -1
python - <<'PYEOF'
import torch, torch_geometric, xgboost, shap, lightgbm, pandas, sklearn
print("deps ok | torch", torch.__version__, "| pyg", torch_geometric.__version__, "| xgb", xgboost.__version__, "| shap", shap.__version__)
print("gpu:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PYEOF
