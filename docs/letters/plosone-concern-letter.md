# 给 PLOS One 编辑部的关注信（Post-Publication Concern）

> 用途说明：正式投诉信。英文版为发送主件，中文版供你逐句核对内容。发送前完成三个占位符（署名、单位、联系方式）。
>
> 发送渠道（按优先级）：
> 1. PLOS 官方联系表单/邮箱，从 https://www.plos.org/contact/ 现查（避免用过期邮箱）
> 2. 主题行务必含 DOI，分拣快
> 3. 同时在 PLOS One 论文页提交 Reader Comment（公开记录，双轨）
>
> 建议实名并署导师（PLOS 对匿名举报处理权重低；你们的泄漏法医分析已进学位论文第 4.6 节，是时间戳证据）。

---

## 英文正式信（主件）

**Subject: Post-publication concern regarding 10.1371/journal.pone.0337208 — evaluation-set leakage in the released code invalidates all reported metrics**

Dear Editor-in-Chief and Editorial Office of PLOS ONE,

I am writing to raise a post-publication concern about the following article:

Wei S., Lee S. (2025). Internet fraud transaction detection based on temporal-aware heterogeneous graph oversampling and attention fusion network. *PLOS ONE*, 20(12): e0337208. https://doi.org/10.1371/journal.pone.0337208

I am an independent researcher who reproduced the experiments described in the article using the authors' public code repository (https://github.com/wei4zheng/THG-OAFN). My reproduction shows that the released training pipeline computes its loss on the labels of **all nodes, including the validation and test sets**. The train/validation/test split exists in name only: the `train_mask` argument is accepted by the training function but is never connected to the loss. Consequently, every gradient step optimizes directly on test-set answers, and every metric reported from this pipeline measures memorization of the evaluation set rather than fraud detection.

**Evidence (all verifiable by file and line number in the public repository):**

1. `train.py:17-39` — `train_epoch(..., train_mask, ...)` accepts the mask; the function body never uses it. Full labels enter the model, and the training loop (`train.py:153-156`) repeats this every epoch.
2. `models/thg_oafn.py:152` — `F.cross_entropy(logits, labels_processed)` is computed over all nodes, including test nodes. Functionally this is identical to `X_train = concat(X_train, X_test); y_train = concat(y_train, y_test)`.
3. `models/thg_oafn.py:101-108` — the GraphSMOTE oversampling step counts and samples minority nodes from all labels; test-set fraud nodes are therefore oversampled into training.
4. `utils/data_loader.py:281` (with the definition at 168-185) — `normalize_features` performs a full-data `fit_transform` inside `load_real_dataset`, before any split occurs.
5. `utils/data_loader.py:289-292 and 150` — timestamps remain `None` unless the node file carries a timestamp column; `split_data` then falls back to raw node order, and the "temporal split verification" block in `train.py:109-126` can never execute on the flagship Amazon dataset. No temporal split exists.

**Quantified impact.** To measure the size of this artifact, we reproduced the same leakage mechanism (appending evaluation rows and labels to training) under a clean evaluation protocol on a public AML benchmark (IBM HI-Small, 5.07M transactions). Under a temporal split, the leak inflates AUPRC from 0.6276 to 0.9340 (+0.31); under a random split — which is what the repository's node-order splitting effectively is — from 0.5873 to 0.9588 (+0.37). The performance levels reported in the article and its repository README (AUC 96.56%, Recall 95.21%, F1 94.72% on the Amazon dataset) fall squarely within the magnitude this artifact produces. Our full forensic experiment (script, per-seed results, JSON artifacts) is publicly available and was lodged in a public GitHub issue on 2026-09-21: https://github.com/wei4zheng/THG-OAFN/issues/1. As of the date of this letter, the issue has received no response from the authors.

I wish to state clearly that I make no claim about intent; the concern is strictly about the validity of the evaluation. However, because the defect affects every reported metric of the article, I respectfully request that the journal initiate an assessment under COPE guidelines — including contacting the authors for an explanation — and consider an Expression of Concern or Retraction, as the reported results cannot be substantiated by the released code.

I declare no competing interests. This analysis is part of my graduate thesis research (a leakage-free re-evaluation protocol for anti-money-laundering models), and I am happy to provide the complete reproduction package, raw run logs, and per-seed artifacts upon request.

Sincerely,

[姓名 / Full name]
[单位与身份 / Affiliation and status, e.g., Master's student, School of Finance, University]
[联系邮箱 / Email]
[ORCID（如有）]

**Attachments (suggested):**
1. Reproduction package: leakage forensic script and JSON artifacts (temporal/random × clean/leaked, with configuration snapshot) — bundled as `THG-OAFN-plosone-evidence.zip`
2. Screenshot/record of GitHub issue #1 (filed 2026-09-21)
3. Diff summary of the code locations cited above (verified line numbers, included in the bundle as `code-evidence.md`)

---

## 中文对照（供核对，勿发送）

**主题：关于 10.1371/journal.pone.0337208 的发表后关注——发布代码中的评估集泄漏使全部报告指标失效**

尊敬的 PLOS ONE 主编与编辑部：

现就下述文章提出发表后关注：

Wei S., Lee S. (2025). Internet fraud transaction detection based on temporal-aware heterogeneous graph oversampling and attention fusion network. *PLOS ONE*, 20(12): e0337208.

本人为独立研究者，使用作者公开代码库（github.com/wei4zheng/THG-OAFN）复现了论文实验。复现表明：发布的训练流程在**全部节点（含验证集与测试集）的标签**上计算损失。数据切分名存实亡——训练函数接受 `train_mask` 参数但从未将其接入损失。因此每个梯度步都在直接优化测试集答案，该流程产生的全部指标衡量的是对评估集的记忆而非欺诈检测能力。

**证据（全部可通过公开仓库的文件与行号验证）**：① `train.py:17-39` 接受掩码但函数体从未使用，且训练循环（`train.py:153-156`）每轮都以全量标签调用；② `thg_oafn.py:152` 在含测试节点的全量标签上计算交叉熵（等价于把测试集拼接进训练集）；③ `thg_oafn.py:101-108` GraphSMOTE 从全量标签中挑选少数类节点，测试集欺诈节点被过采样进训练；④ `utils/data_loader.py:281`（定义在 168-185）在 `load_real_dataset` 内、切分发生之前对全量数据 `fit_transform`；⑤ `utils/data_loader.py:289-292 与 150` 节点文件无时间戳列时 timestamps 恒为 None，`split_data` 退化为按节点顺序切分，`train.py:109-126` 的"时间切分验证"在旗舰 Amazon 数据集上永不触发——不存在时间切分。

**量化影响**：为测量该伪影的量级，我们在公开 AML 基准（IBM HI-Small，507 万笔）上、以干净评估协议复现同一泄漏机制（将评估行与标签附加进训练）：时序切分下泄漏使 AUPRC 由 0.6276 升至 0.9340（+0.31）；随机切分（即该仓库按节点序切分的实际形态）下由 0.5873 升至 0.9588（+0.37）。论文及仓库 README 报告的性能（Amazon 数据集 AUC 96.56%、Recall 95.21%、F1 94.72%）恰好落在该伪影可产生的量级之内。完整法医实验（脚本、逐种子结果、JSON 工件）已公开，并于 2026-09-21 在公开 GitHub issue 提交（链接见上）。截至本信日期，该 issue 未获作者回应。

本人郑重声明：不对动机作任何主张，关注点仅在于评估有效性。但由于该缺陷影响论文全部报告指标，谨请求期刊依 COPE 准则启动评估（包括联系作者说明），并考虑发布关注声明（Expression of Concern）或撤稿（Retraction），因发布代码无法支撑所报告的结果。

本人声明无利益冲突。该分析系本人硕士论文研究（反洗钱模型的无泄漏重评估协议）的一部分，可应要求提供完整复现包、原始运行日志与逐种子工件。

此致
敬礼

[署名占位]

---

## 发送前检查清单

- [ ] 论文页与 DOI 再核对一遍（PLOS 站内检索 e0337208）
- [ ] 三个占位符（姓名/单位/邮箱，ORCID 可选）填完
- [ ] 决定是否署导师（建议署，增加处理权重）
- [ ] 附件 1：`scripts/leakage_forensics.py` + `experiments/results/leakage_forensics.json` 打包
- [ ] 附件 2：GitHub issue #1 存档（截图或 PDF 打印）
- [ ] 中文版仅供你自己核对，勿发送（PLOS 用英文受理）
- [ ] 信中所有行号在发送前对最新版仓库再验一次（作者若已 push 修复，行号会漂移——若已修复，信的诉求改为"对已发表结果的更正"）
