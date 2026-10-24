# Issue #1 更正评论（建议以 comment 形式发布，保留原帖历史）

> 用法：贴到 https://github.com/wei4zheng/THG-OAFN/issues/1 评论区。选择 comment 而非编辑原帖，是为了保留原始陈述的历史——更正本身就是严谨性的展示。行号已对 git 快照逐条验证（utils/data_loader.py 经 GitHub 提交历史确认自 2025-10-28 后未再修改）。

---

**Line-number correction (verified against the repository snapshot)**

Re-verifying every citation against the actual files, three of the five locations need corrected line numbers. The substance of every claim is unchanged — the corrected locations make each one verifiable:

- `train.py:17` — correct as stated (`train_epoch` accepts `train_mask`, never uses it). The training loop at `train.py:153-156` confirms full labels are passed every epoch.
- `models/thg_oafn.py:152` — correct as stated (`F.cross_entropy(logits, labels_processed)` over all nodes).
- `models/thg_oafn.py:101-107` → the full block is **101-108** (line 102 counts fraud nodes over all labels; line 105 passes all labels into GraphSMOTE).
- `data_loader.py:257,281` → the scaler citation is **`utils/data_loader.py:281`** (the `normalize_features` call inside `load_real_dataset`, executed before any split), with the definition at **168-185** (full-data `fit_transform`). Line 257 was incorrect.
- `data_loader.py:265` → the no-timestamp citation is **`utils/data_loader.py:289-292`** (timestamps stay `None` without a timestamp column) combined with **line 150** (`split_data` falls back to raw node order). Consequently the temporal-split verification block at `train.py:109-126` never executes on the Amazon dataset. Line 265 was incorrect.

A fully verified evidence excerpt with the exact code spans is available here: [code-evidence.md](https://github.com/xiaohangguo/THG-OAFN-change/blob/main/docs/letters/evidence/THG-OAFN-plosone-evidence/code-evidence.md)
