# TP-THGN 实验设计文档

## Topology-Preserving Temporal Heterogeneous Graph Network for Credit Card Fraud Detection

**论文题目**: 极端非平衡下信用卡交易欺诈风险识别与可解释预警研究——基于拓扑保持时序异构图模型
**模型命名**: TP-THGN (Topology-Preserving Temporal Heterogeneous Graph Network)
**日期**: 2026-06-01

---

## 1. 项目目标

在现有THG-OAFN代码基础上进行模块化重构，产出一篇金融学硕士毕业论文所需的全部实验结果：

- 四个改进模块的独立实现与消融验证
- 与6+基线模型的对比实验
- 跨数据集（Amazon + IEEE-CIS + Kaggle CC）验证
- 三层可解释归因的案例分析与可视化
- 完整的实验结果表格与论文图表

---

## 2. 模型架构

### 2.1 整体流程

```
输入数据 → 异构图构建 → 特征提取 → TD-GRU-GNN → TP-GraphSMOTE → XAttention → 分类器
                                                                        ↓
                                                                  TriExplainer → 归因报告
```

### 2.2 模块1: TD-GRU-GNN (Time-Decay GRU-GNN)

改进点: 在原始GRU-GNN融合中引入时间衰减因子。

核心公式:
```
decay_weight = exp(-λ · Δt)
h_fused = α · (decay_weight ⊙ h_gru) + (1-α) · h_gnn
```

实现要点:
- λ 初始化为 0.1，通过反向传播学习
- Δt 归一化到 [0, 1] 区间
- 无时间戳时退化为原始等权融合

消融验证: 对比有/无时间衰减的F1差异

### 2.3 模块2: TP-GraphSMOTE (Topology-Preserving GraphSMOTE)

改进点: 加入拉普拉斯正则化约束。

核心公式:
```
L_total = L_cls + β · L_laplacian
L_laplacian = Σ_{(i,j)∈E} ||h_i - h_j||² · A_ij
```

实现要点:
- 生成合成节点时约束嵌入与k-hop邻域的拉普拉斯平滑一致
- β 搜索范围: [0.001, 0.01, 0.1, 1.0]
- k-hop 约束: k=1 或 k=2
- 保留原始欺诈簇的局部拓扑结构

消融验证: 对比有/无拉普拉斯约束的F1和拓扑保持度

### 2.4 模块3: XAttention (Explainable Multi-Layer Attention)

改进点: 三层注意力增加权重导出接口。

三层结构:
1. 关系融合层 → 关系重要性权重 w_r
2. 邻域融合层 → 邻居贡献度 α_vu
3. 信息感知层 → 特征维度门控权重 g_h

实现要点:
- 训练时正常前向传播，不增加计算开销
- 推理时通过 explain=True 触发权重收集
- 权重存储为字典结构供TriExplainer消费

### 2.5 模块4: TriExplainer (Three-Level Explainer)

全新模块，提供三层归因:

1. 特征层归因: 门控权重 → Top-K重要特征
2. 边层归因: 邻域注意力 → 关键关联交易
3. 子图层归因: 计算图剪枝 → 欺诈传播路径

输出格式: JSON结构，包含每笔交易的归因解释。

---

## 3. 数据策略

### 3.1 主数据集: Amazon
- 已有 data/Amazon.mat
- 11,944节点，~9.5M边，25维特征，欺诈率6.87%
- 用途: 模型开发、调参、消融实验、主对比实验

### 3.2 验证数据集: IEEE-CIS Fraud Detection
- 来源: Kaggle
- ~590K交易，欺诈率~3.5%
- 构图: 持卡人-商户-设备-地址异构图
- 用途: 跨场景验证、可解释性案例

### 3.3 补充数据集: Kaggle Credit Card
- 284,807交易，28维PCA特征，欺诈率0.172%
- 构图: 时序交易模式构图
- 用途: 极端不平衡场景验证

### 3.4 数据管理
- 原始数据大文件 → .gitignore
- 预处理脚本 → git跟踪
- 实验结果摘要 → git跟踪

---

## 4. 基线模型

| 编号 | 模型 | 类别 |
|------|------|------|
| B1 | Logistic Regression | 传统ML |
| B2 | XGBoost | 集成学习 |
| B3 | GCN | 基础GNN |
| B4 | GAT | 注意力GNN |
| B5 | GraphSAGE | 采样GNN |
| B6 | THG-OAFN (原始) | 本文基础 |
| B7 | SMOTE + GCN | 过采样+GNN |

---

## 5. 评价指标

主指标: F1-score (欺诈类), AUPRC, AUC-ROC
辅助指标: Recall, Precision, G-Mean
可解释性: Fidelity, Sparsity, Case Study

---

## 6. 实验Phase

| Phase | 分支 | 内容 | 产出 |
|-------|------|------|------|
| 0 | experiment/phase0-baseline | 基线复现 | baseline_results.json |
| 1 | experiment/phase1-td-gru-gnn | 时间衰减模块 | phase1_results.json |
| 2 | experiment/phase2-tp-graphsmote | 拓扑保持过采样 | phase2_results.json |
| 3 | experiment/phase3-xattention | 可解释注意力 | phase3_results.json |
| 4 | experiment/phase4-triexplainer | 三层归因 | 归因案例JSON |
| 5 | experiment/phase5-ablation | 消融+对比实验 | 实验表格CSV |
| 6 | experiment/phase6-cross-dataset | 跨数据集+可视化 | 论文图表 |

---

## 7. Git管理

每个Phase完成后:
1. 实验结果摘要commit到对应分支
2. 代码改动merge回main
3. 原始数据/大文件通过.gitignore排除

---

## 8. 文件结构

```
THG-OAFN-change/
├── data/                          # 数据 (大文件gitignore)
├── models/
│   ├── thg_oafn.py               # 原始模型 (保留)
│   ├── tp_thgn.py                # 改进模型 (新增)
│   ├── td_gru_gnn.py            # 时间衰减GRU-GNN
│   ├── tp_graphsmote.py          # 拓扑保持GraphSMOTE
│   ├── xattention.py            # 可解释注意力
│   ├── tri_explainer.py          # 三层归因
│   └── baselines/                # 基线模型
├── utils/
│   ├── data_loader.py            # 扩展支持多数据集
│   ├── metrics.py                # 扩展AUPRC等
│   └── visualization.py          # 可视化工具
├── experiments/                   # 实验脚本与结果
│   ├── run_baseline.py
│   ├── run_ablation.py
│   ├── run_comparison.py
│   └── results/
├── figures/                       # 论文图表
├── train.py                      # 原始训练 (保留)
└── train_tp_thgn.py             # 改进模型训练
```

---

## 9. 论文图表清单

| 编号 | 类型 | 内容 | 章节 |
|------|------|------|------|
| Fig.1 | 架构图 | TP-THGN整体架构 | §3 |
| Fig.2 | 折线图 | 训练收敛曲线 | §4.1 |
| Fig.3 | 柱状图 | 消融实验对比 | §4.2 |
| Fig.4 | 热力图 | 注意力权重可视化 | §5.1 |
| Fig.5 | 网络图 | 欺诈子图归因案例 | §5.1 |
| Fig.6 | 柱状图 | 跨数据集验证 | §4.3 |
| Fig.7 | 折线图 | 不同不平衡率性能 | §4.4 |
| Tab.1 | 表格 | 数据集统计 | §4.1 |
| Tab.2 | 表格 | 超参数设置 | §4.1 |
| Tab.3 | 表格 | 主实验结果 | §4.2 |
| Tab.4 | 表格 | 消融实验 | §4.2 |
| Tab.5 | 表格 | 跨数据集结果 | §4.3 |

---

## 10. 成功标准

- Amazon数据集F1 >= 88% (保守)，争取 >= 92%
- 每个模块消融显示正向贡献 (ΔF1 > 0)
- 超越原始THG-OAFN >= 5pp
- 超越GCN/GAT >= 10pp
- 三层归因通过人工合理性检查
- 跨数据集保持相对排名一致
- 产出全部论文图表
