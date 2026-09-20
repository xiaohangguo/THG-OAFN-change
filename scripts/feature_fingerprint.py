"""Per-column fingerprints to localize cross-machine feature divergence."""

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

arr = X.to_numpy(dtype=np.float64)
for j, col in enumerate(X.columns):
    col_arr = np.ascontiguousarray(arr[:, j])
    print(f"{col}\t{hashlib.md5(col_arr.tobytes()).hexdigest()[:10]}\t{col_arr.sum():.6e}")
