"""Feature-matrix fingerprint for cross-machine consistency debugging."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrisk.dataset import attach_features, load_transactions
from finrisk.entity_join import attach_entities
from finrisk.graph_builder import build_global_account_index, extract_profile_columns

config = yaml.safe_load((ROOT / "configs/ibm_aml_hi_small.yaml").read_text(encoding="utf-8"))
df = load_transactions(Path(config["data"]["transactions_path"]))
df = attach_entities(df, Path(config["data"]["accounts_path"]))
X = attach_features(df)
src_gid, dst_gid, _ = build_global_account_index(df)
X = X.join(extract_profile_columns(df, src_gid, dst_gid))

arr = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
print("shape:", arr.shape)
print("md5:", hashlib.md5(arr).hexdigest())
print("sum:", repr(float(arr.sum())))
print("nan_count:", int(np.isnan(arr).sum()))
for i in (0, 1, 2, arr.shape[0] // 2, arr.shape[0] - 1):
    print(f"row{i} md5:", hashlib.md5(arr[i].tobytes()).hexdigest())
