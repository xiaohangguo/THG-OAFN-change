# 论文总规划（PAPER PLAN）

更新：2026-09-21 · 云机已关机 · 全部实验资产已在库，进入写作阶段

## 一、实验资产全景（写作可用弹药库）

| # | 资产 | 关键数字 | 位置 |
|---|---|---|---|
| 1 | 主表（5 种子，78 列，零泄漏时序切分） | XGBoost AUPRC **0.6214±0.0040**、ROC 0.9904、P@0.1%=0.8337、R@1%=0.8427；MLP 0.2803、LightGBM 0.2619、随机 0.0018 | `experiments/results/phase2/` |
| 2 | 特征消融链（每级增量可解释） | 33 列 0.412 → 60 列 0.489 → 78 列 0.585 → 超参 0.628 | EXPERIMENT_REPORT.md §3 |
| 3 | 超参搜索（5 候选存档） | regularized 胜出（mcw50/λ5/colsample0.7，433 轮） | `hparam_search/` |
| 4 | GNN 七发诊断链（全负结果） | SAGE 0.030 → GATv2 0.012 → +强画像 0.0125 → 冻结注入 0.5922（<0.6276） | EXPERIMENT_REPORT.md §5 |
| 5 | 探针反转实验 | GNN 输入画像 18 列直接给 XGBoost：0.487→0.585（+0.098） | PROJECT_SUMMARY.md |
| 6 | SHAP 全局归因 | top-5 全部 AML 业务语义（pair_w7d_count 居首） | `attribution/` |
| 7 | Fidelity 双版本 | 60 列版遮 top-10 坍缩 99.998%；78 列版 70.7% | EXPERIMENT_REPORT.md §6 |
| 8 | 组级反事实归因（v2，20 案例） | pair_frequency 组置回普通人：均降 **0.927**（max 0.993）；entity_diff 0.879；其余组≈0 | `routeB_counterfactual_v2.json` |
| 9 | 精度-可归因性帕累托前沿 | 78f 0.618/0.892 → 30f 0.537/0.968 → 15f 0.381/0.857 → 8f 0.234/0.570 | 同上 |
| 10 | 阈值锁定指标（协议承诺项） | 4 种子 F1≈0.588、P≈0.886、R≈0.440；AUPRC 0.6214 复现一致 | `phase2/threshold_metrics_recovered.json` |
| 11 | 泄漏法医 2×2 | 泄漏配置 vs 无泄漏配置的 +0.31 差距定量 | `leakage_forensics.json` |
| 12 | SSL 信息上限论证 | 探针判决：嵌入无监督信息量不足 | `routeA_ssl_probe.json` |
| 13 | 数据审计与复现 | SHA-256 锁定、5,078,345 笔/0.102% 正样本、跨机一致（特征 MD5 相同、AUPRC 差 0.006） | `metadata/` |

## 二、叙事主线（对开题的诚实升级）

**一句话**：在严格无泄漏的时序协议下，把交易图的关系信息以因果行为特征的形式注入 GBDT，配合组级反事实归因协议，同时拿到检测精度（AUPRC 0.6214）与监管级可归因性（单案例反事实瀑布 0.93/0.88）。

### GNN 执念的转化（换角度而非认输）

七发诊断链 + 探针反转合起来讲一个完整的科学故事：
1. 端到端 GNN 三个出口（端到端训练、warmup 调度、冻结注入）被系统性排除；
2. 但同样的图信息以因果聚合特征进入 XGBoost 立即 +0.098；
3. 结论：**图信息的正确出口是因果聚合特征，不是端到端嵌入**。
4. 反手服务归因主线：特征形式进入 = SHAP/反事实可作用（可归因）；端到端嵌入 = 黑箱（不可归因）。GNN 的死是"可归因优先"方法论的正面论据，与 Rudin (2019) 立场同构，与 Grinsztajn et al. (NeurIPS 2022) 表格数据 GBDT 统治结论互证。

### 三个贡献（论文的骨头）

- **C1 因果行为特征框架**：三层递进（基础窗口 → pair/entity 结构 → 快照画像），每层增量单独可测（0.412→0.489→0.585→0.628）。
- **C2 组级反事实归因协议**：干预单位从特征升级为行为族，干预语义从"归零"改为"置回人群中位数（回到普通行为）"；输出 SAR 式案例解释（"主因：与对手 X 的 7 天高频互转，贡献 0.93"）。
- **C3 精度-可归因性帕累托前沿**：开题命题"精准性与可解释性权衡"的第一次可测量实证；附带无泄漏协议 + 泄漏量化法医 + GNN 系统性负结果。

### 金融学生身份的叙事红利

- 业务指标语言：审核容量 0.1%/1% 的 P/R、阈值锁定工作点的 P=0.886（~89% 告警是真阳性，合规友好）；
- 特征族 ↔ FATF 洗钱三阶段映射：pair_frequency=分层（layering）快速互转，entity_diff=整合（integration）跨账户归集，burst=放置（placement）突发高频；
- 案例解释即 SAR 报告语言，归因章天然是"监管可接受性"论证。

## 三、投稿定位

- 主投：**《计算机应用》**（北大核心，应用型，AI+金融交叉友好，双栏模板）
- 备选：《计算机工程与应用》（篇幅宽容）
- 体量：8-12 页；图表预算：主表 1、消融 1、前沿曲线 1、反事实瀑布 1、SHAP 1、GNN 诊断链 1

## 四、论文骨架（章节 → 弹药映射）

1. **引言**：AML 痛点（极不平衡、监管要求可解释）→ 文献两大病（泄漏普遍、端到端黑箱）→ 三贡献
2. **相关工作**：AML 机器学习 / 图方法 / 反事实可解释性 / GBDT-GNN 之争
3. **方法（CBP-GCA）**：3.1 无泄漏时序协议；3.2 因果行为特征三层框架；3.3 组级反事实归因协议（中位数置换）；3.4 精度-可归因前沿定义
4. **实验**：4.1 数据与设置（表 13 资产）；4.2 主表+消融（资产 1/2/3）；4.3 归因：SHAP+组级瀑布+案例（资产 6/7/8）；4.4 前沿曲线（资产 9）；4.5 GNN 诊断链负结果（资产 4/5/12）；4.6 泄漏法医（资产 11）
5. **结论**

方法名候选：CBP-GCA（Causal Behavioural Profiling with Group-Counterfactual Attribution）。

## 五、待办实验（按优先级）

| 级别 | 实验 | 机器 | 成本 | 说明 |
|---|---|---|---|---|
| P0 | E1 补 seed2024 threshold + 拉回云机 json | 云机 10 分钟 | 下次开机第一件事 | 完整 5 种子 F1/P/R |
| P0 | E5 全套图表（matplotlib） | 本地 | 半天 | 主表/消融/前沿/瀑布/SHAP/GNN 链 |
| P1 | E2 基线补齐：Logistic Regression + 调参 LightGBM | 本地 CPU | 1-2 小时 | 堵"LightGBM 没调参"审稿口；协议最低对照组承诺 |
| P1 | E3 top-30 甜点模型完整评估（P@K/R@K/F1） | 本地 | 1 小时 | 前沿表只差这半边 |
| P2 | E4 Elliptic++ 外部验证 | 云机半天 | 数据 10GB+ | 泛化性终证；不做主线也成立 |
| P2 | E6 组级归因跨种子稳定性 | 云机 1 小时 | 现仅 1 模型 20 案例 | 补方差更硬 |
| P3 | E7 归因方法对照（permutation vs SHAP vs 组级） | 本地 | 2 小时 | 强化 C2 必要性 |

## 六、写作工具链（GitHub skills 调研结论）

中文科研导向（本轮 GitHub API 两轮检索 16 仓库）：

| 仓库 | 星 | 用途 | 结论 |
|---|---|---|---|
| [zLanqing/codex-claude-academic-skills](https://github.com/zLanqing/codex-claude-academic-skills) | 4146 | 中文写作/润色/审稿回复 + 期刊级图表 + PPT/Word | **主力安装** |
| [cangtianhuang/humanizer-academic-zh](https://github.com/cangtianhuang/humanizer-academic-zh) | 60 | 中文学术去 AI 痕迹，轻量省 token | 终稿阶段安装 |
| [houlaisan/deai-academic-zh](https://github.com/houlaisan/deai-academic-zh) | 56 | AIGC 检测对抗改写 | 备选（与上二选一） |
| [lishix520/academic-paper-skills](https://github.com/lishix520/academic-paper-skills) | 1326 | strategist+composer 规划写作（英文向） | 借鉴流程思想 |
| [kael-odin/awesome-academic-research-skills](https://github.com/kael-odin/awesome-academic-research-skills) | 99 | 中文科研 skill 每日榜 | 追踪入口 |

**缺口判断**：现成 skill 无一内建"中文核心期刊计算机类"投稿规范（GB/T 7714 参考文献格式、中图分类号、双栏模板）。方案：装 zLanqing 主力 + 终稿用 humanizer 去 AI 痕迹，另自建一个 `zhcore-journal-fmt` skill 固化《计算机应用》格式规范（结构模板、图表规范、GB/T 7714 引用、AI 痕迹自检清单）。

## 七、写作推进顺序

1. 图表先行（E5）：数字都在，先把 6 张核心图做出来，论文围绕图写；
2. 方法章（第 3 章）最稳：全部素材在库，无实验依赖；
3. 实验章（第 4 章）逐节填装弹药 1-11；
4. 引言/相关工作最后写（需要文献检索，观点随正文定型）；
5. 终稿：格式 skill 过一遍 + 去 AI 痕迹 + 中图分类号/GB/T 7714 合规。
