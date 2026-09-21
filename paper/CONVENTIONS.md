# 写作规范（CONVENTIONS）

来源：research-writing-skill（zLanqing/codex-claude-academic-skills）内化 + 中文核心期刊计算机类格式。

## 语言与证据纪律

- 中文学术表达为默认；英文术语首次出现给中文译名并括注原文，如"平均精度（Average Precision, AP）"。
- 保留：公式、变量名、软件名（XGBoost、SHAP）、数据集名（IBM AML HI-Small、Elliptic++）、英文文献标题。
- 禁用空词：显著、先进、有效、鲁棒、极大地——替换为测量条件与对照基线（"在相同切分下 AUPRC 由 X 升至 Y"）。
- 每个定量论断可溯源到 `experiments/results/` 中的 JSON；不发明数据、DOI、期刊细节。
- 段落式行文，禁止口号式列表；每段推进一个论点。
- 结果讨论的段内顺序：论断 → 证据 → 机制解释 → 局限。

## 结构模板（《计算机应用》体，章节 move）

- 摘要：问题 → 方法 → 实验 → 关键结果 → 贡献（300 字左右，中图分类号 TP391，关键词 5 个）。
- 引言：背景 → 未解决的缺口 → 为何重要 → 本文方法 → 三贡献 → 结构安排。
- 相关工作：按技术主题组织（AML 机器学习 / 图方法 / 反事实可解释性 / GBDT-GNN 之争），比较假设与局限，禁止逐篇罗列。
- 方法：假设与变量 → 框架总览 → 各模块（含可复现细节）→ 与最近工作的区别。
- 实验：数据与审计 → 协议设置 → 主结果 → 消融 → 归因 → 前沿 → 负结果分析。
- 结论：回答研究问题 → 证据小结 → 局限与下一步。

## 术语表（全文统一，首次出现定锚）

| 统一用词 | 英文/代码名 | 备注 |
|---|---|---|
| 反洗钱 | anti-money laundering, AML | |
| 因果行为画像框架 | Causal Behavioural Profiling, CBP | 方法主名 CBP-GCA |
| 组级反事实归因 | group-level counterfactual attribution | 干预=置回人群中位数 |
| 精度-可归因性前沿 | accuracy-attributability frontier | |
| 无泄漏时序协议 | leakage-free temporal protocol | 60/20/20，边界吸附 |
| 行为族 | behaviour group | 8 族，如 pair_frequency |
| 平均精度 | AUPRC | 主指标 |
| 审核容量 | review budget @K | K=0.1%/1% |

## 图表规范

- 图题表题中英双语（《计算机应用》要求中文为主）。
- 所有图出自脚本（matplotlib，脚本入 `scripts/figures/`），图内文字中文，无 AI 装饰。
- 表格三线表；数值保留位数全文一致（AUPRC 4 位、百分比 1 位）。
- 核心图表预算：主表 1、消融 1、前沿 1、反事实瀑布 1、SHAP 1、GNN 诊断链 1。

## 参考文献格式

- GB/T 7714-2015 顺序编码制：[J]期刊 [C]会议 [EB/OL]在线。
- 数字对象标识符（DOI）不确定的条目标注 TODO，禁止编造。

## 当前占位说明

- seed 2024 的 F1/P/R 单行待下次开机补（现用 4 种子均值并注明）。
