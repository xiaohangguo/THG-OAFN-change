# TP-THGN 实验进度记录

## 项目概述
- **模型**: TP-THGN (Topology-Preserving Temporal Heterogeneous Graph Network)
- **论文**: 极端非平衡下信用卡交易欺诈风险识别与可解释预警研究
- **分支**: worktree-tp-thgn-design
- **环境**: Conda thg-oafn, Python 3.10, PyTorch 2.5+cu124, DGL 2.0, RTX 4070

---

## 已完成阶段

### Phase 0: 基础设施 (2026-06-01)
- [x] 创建 .gitignore（排除大数据文件）
- [x] 更新 requirements.txt（添加 xgboost, networkx）
- [x] 扩展 utils/metrics.py（添加 AUPRC, G-Mean, Specificity）
- [x] 创建 experiments/ 和 figures/ 目录
- [x] 编写 experiments/run_baseline.py（THG-OAFN 基线 5-seed 运行器）

### Phase 1: TD-GRU-GNN 时间衰减模块 (2026-06-01)
- [x] 实现 models/td_gru_gnn.py
- [x] 核心创新: exp(-λ·Δt) 时间衰减因子，λ可学习
- [x] 验证: Amazon数据集(11,944节点)前向/反向传播通过
- [x] 无时间戳时正确退化为标准融合

---

## 进行中

### Phase 2: TP-GraphSMOTE 拓扑保持过采样
- [ ] 实现 models/tp_graphsmote.py
- [ ] 拉普拉斯正则化约束
- [ ] 验证模块

---

## 待完成
- Phase 3: XAttention 可解释注意力
- Phase 4: TriExplainer 三层归因
- Integration: 完整 TP-THGN 模型组装
- Phase 5: 消融实验 + 对比实验
- Phase 6: 跨数据集验证 + 论文图表
