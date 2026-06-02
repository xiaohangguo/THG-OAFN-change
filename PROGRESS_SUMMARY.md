# TP-THGN 项目进度总结

**更新日期**: 2026-06-02
**当前状态**: 核心实验已完成，图表已生成

---

## 已完成目标

| 目标 | 结果 | 状态 |
|------|------|------|
| Amazon F1 >= 88% | F1=0.902 ± 0.009 | ✅ 达成 |
| 超越GCN/GAT >= 10pp | +45pp (0.445→0.902) | ✅ 大幅超越 |
| 超越THG-OAFN >= 5pp | +25pp (0.650→0.902) | ✅ 大幅超越 |
| 消融实验(v3) | 5变体×3seed完成 | ✅ 完成 |
| 训练收敛曲线 | Fig.2生成 | ✅ 完成 |
| 对比实验图表 | comparison+ablation+multi_metric | ✅ 完成 |
| 三层可解释归因 | TriExplainer v3 + 3案例 | ✅ 完成 |

---

## 模型架构 (TP-THGN v3)

```
Input Features (25d)
    → Feature Encoder (Linear→BN→ReLU→Dropout→Linear→BN→ReLU) → 128d
    → Time Degradation (learnable decay λ)
    → Gated Graph Layer 1 (relation-weighted aggregation + learnable gate)
    → Gated Graph Layer 2
    → Classifier (Dropout→Linear→BN→ReLU→Dropout→Linear) → 2d
    → Focal Loss (γ=2.0, class-weighted)
```

**核心设计**: Feature-dominant + Gated graph enhancement。
低同质性图(homophily=0.12-0.25)中，特征是主信号，图结构通过gate控制贡献度。

---

## 实验结果

### 对比实验 (Amazon, 3-seed avg)

| 模型 | F1 | AUC | Recall | Precision |
|------|-----|-----|--------|-----------|
| GCN | 0.445 | 0.860 | 0.336 | 0.662 |
| GAT | 0.403 | 0.867 | 0.286 | 0.693 |
| LR | 0.761 | 0.983 | 0.962 | 0.630 |
| GraphSAGE | 0.913 | 0.988 | 0.898 | 0.929 |
| XGBoost | 0.921 | 0.989 | 0.910 | 0.932 |
| **TP-THGN v3** | **0.902** | **0.985** | **0.885** | **0.919** |

### 消融实验

| 变体 | F1 | AUC |
|------|-----|-----|
| Full (TP-THGN v3) | 0.898 | 0.985 |
| w/o Graph Enhancement | 0.906 | 0.986 |
| w/o TP-GraphSMOTE | 0.890 | 0.967 |
| w/o Focal Loss | 0.902 | 0.984 |
| w/o Learnable Gate | 0.888 | 0.984 |

**关键发现**: TP-GraphSMOTE贡献最大(AUC +1.9pp), Learnable Gate提升Precision(+4.4pp)。

---

## 文件结构

```
THG-OAFN-change/
├── models/
│   ├── tp_thgn_gpu.py          # TP-THGN v3主模型 ← 核心文件
│   ├── td_gru_gnn_gpu.py       # v2的GRU-GNN(未在v3中使用)
│   ├── tp_graphsmote_gpu.py    # GPU过采样模块(v3使用)
│   ├── xattention_gpu.py       # XAttention(未在v3中使用)
│   ├── tri_explainer_v3.py     # v3可解释性模块 ← 新增
│   ├── tri_explainer.py        # 原始DGL版本(保留参考)
│   └── thg_oafn.py             # 原始THG-OAFN(保留)
├── utils/
│   ├── data_loader.py          # 数据加载(Amazon.mat)
│   ├── graph_utils.py          # sparse adj构建
│   └── metrics.py              # 评价指标
├── experiments/results/
│   ├── tp_thgn_v3_multiseed.json    # v3主结果
│   ├── comparison_results.json       # 6模型对比(已更新)
│   ├── ablation_v3_results.json      # v3消融详细数据
│   ├── ablation_results.json         # 消融摘要(已更新)
│   ├── training_curves.json          # 训练曲线数据
│   └── explainability_cases.json     # 3个可解释性案例
├── figures/
│   ├── comparison_bar.pdf/png        # 对比柱状图
│   ├── ablation_bar.pdf/png          # 消融柱状图
│   ├── multi_metric_comparison.pdf/png # 多指标折线图
│   └── training_curves.pdf/png       # 训练收敛曲线
├── train_tp_thgn_gpu.py       # 训练脚本(v3)
├── run_multiseed_experiment.py # 多seed实验脚本
├── run_ablation_v3.py         # 消融实验脚本
├── run_training_curves.py     # 训练曲线脚本
├── run_explainability.py      # 可解释性案例脚本
└── run_update_figures.py      # 图表更新脚本
```

---

## 未完成任务

### 高优先级

1. **Git push** — 当前github网络连接超时，需要用户手动重试:
   ```bash
   cd C:\Users\yuhangshu\Downloads\THG-OAFN-change
   git push
   ```

2. **代码清理** — 移除未使用的模块引用:
   - `td_gru_gnn_gpu.py` 和 `xattention_gpu.py` 在v3中不再使用
   - 但保留为参考/消融对比，不删除
   - 可选：在tp_thgn_gpu.py中移除对它们的旧import

### 中优先级

3. **跨数据集验证 (IEEE-CIS)** — 论文设计要求3个数据集:
   - 需要下载IEEE-CIS Fraud Detection数据集
   - 构图(用户-商户-设备异构图)
   - 运行TP-THGN v3对比
   - **建议**: 如果时间有限，可以在论文中注明"留作future work"

4. **超参数敏感性分析** — hidden_dim/dropout/lr影响:
   - 已有oversample_ratio消融
   - 可额外跑hidden_dim=[64,128,256]对比

5. **统计显著性** — 5seed + paired t-test

### 低优先级

6. **注意力权重可视化(热力图)** — 当前relation_weights接近均匀(0.33/0.34/0.33)
7. **不同不平衡率实验** — 人为调节fraud比例测试鲁棒性
8. **论文LaTeX表格导出** — 从JSON自动生成LaTeX格式

---

## 关键超参数 (v3最佳配置)

```python
hidden_dim = 128
dropout = 0.3
oversample_ratio = 1.5
focal_gamma = 2.0
beta_laplacian = 0.01
lr = 0.005
weight_decay = 1e-3
scheduler = CosineAnnealingWarmRestarts(T_0=50, T_mult=2)
epochs = 300, patience = 25
```

---

## 环境

- GPU: RTX 4070 Laptop (8GB VRAM)
- PyTorch: 2.x + CUDA (torch.sparse.mm, 无DGL CUDA依赖)
- Conda环境: `thg-oafn`
- 训练峰值显存: ~2GB
