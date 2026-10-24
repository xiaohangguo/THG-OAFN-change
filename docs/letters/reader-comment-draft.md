# Reader Comment 草稿（PLOS 论文页评论区用）

> 使用前提：注册 PLOS 账号（实名+地理位置，邮箱私密）。位置：论文页（https://doi.org/10.1371/journal.pone.0337208）顶部 **Comments** 标签 → **Post a new comment** → 填标题与正文，competing interests 声明必填（填 "No competing interests"）。
>
> 措辞纪律：评论区不得暗示不当行为（PLOS Comments 政策明文），因此本稿以"技术性数据疑问"口吻呈现，全部 evidence-based；完整性关注的主渠道是邮件（pub-ethics@plos.org）。两轨内容有意不同强度，这是 PLOS 政策设计的正确用法。

---

**Title:** Question about the train/val/test split implementation in the released code

**Body:**

I have been studying the public code release accompanying this article (https://github.com/wei4zheng/THG-OAFN) while reproducing the experiments, and I would appreciate the authors' clarification on the evaluation protocol.

In the released code, the training loss appears to be computed over all nodes rather than over the training split:

- `train.py:17-39` — the `train_epoch` function receives `train_mask` as an argument, but the function body does not apply it; full `labels` are forwarded to the model, and the training loop (lines 153-156) repeats this each epoch.
- `models/thg_oafn.py:152` — `F.cross_entropy(logits, labels_processed)` is computed over all nodes; `labels_processed` derives either from the oversampling path (lines 101-108, which samples from all labels) or directly from full `labels`.
- `utils/data_loader.py:281` — features are normalized on the full dataset inside `load_real_dataset`, before `split_data` is called; and `utils/data_loader.py:289-292,150` — when the node file carries no timestamp column, the split falls back to raw node order, so the temporal-split verification block (`train.py:109-126`) does not execute on the Amazon dataset.

As a point of reference for how sensitive such pipelines are to this family of issues: in a leakage-free evaluation we ran on a public AML benchmark, appending evaluation rows (with labels) to training moved AUPRC from 0.6276 to 0.9340 under a temporal split, and from 0.5873 to 0.9588 under a random split. Details and a line-verified evidence excerpt are available here: https://github.com/wei4zheng/THG-OAFN/issues/1

Could the authors clarify whether the released code corresponds to the version used for the published metrics, and if so, how the reported generalization numbers should be interpreted under a train-mask-connected loss? If a corrected version of the code exists, a pointer to it would fully resolve my question.

**Competing interests:** No competing interests.
