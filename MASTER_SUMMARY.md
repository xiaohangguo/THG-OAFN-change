# 项目总结与工作交接文档 (Master Summary)

**生成时间**: 2026-06-02
**目的**: 供后续 agents 直接读取，无需重新探索项目即可开始工作

---

## 一、项目基本信息

| 项目 | 内容 |
|------|------|
| 论文题目 | 极端非平衡下信用卡交易欺诈风险识别与可解释预警研究——基于拓扑保持时序异构图模型 |
| 学位类别 | 金融硕士 |
| 作者 | 舒宇杭 |
| 学号 | 24980201004 |
| 校内导师 | 邸忆 |
| 行业导师 | 陈宗霞 |
| 模型名称 | TP-THGN v3 (Topology-Preserving Temporal Heterogeneous Graph Network) |
| 核心结果 | Amazon数据集 F1=0.902±0.009, AUC=0.985±0.001 |
| GitHub | https://github.com/xiaohangguo/THG-OAFN-change |
| 本地路径 | C:\Users\yuhangshu\Downloads\THG-OAFN-change |

---

## 二、仓库目录结构与关键文件

```
C:\Users\yuhangshu\Downloads\THG-OAFN-change\
├── models/                          # 模型代码（主分支，原始THG-OAFN）
│   ├── thg_oafn.py                  # 原始基线模型
│   ├── gru_gnn.py
│   ├── graph_smote.py
│   └── attention.py
├── utils/
│   ├── data_loader.py
│   └── metrics.py
├── data/
│   ├── Amazon.mat                   # 主数据集（11,944节点，25维特征，6.87%欺诈率）
│   └── Amazon.zip
├── train.py                         # 原始训练脚本
├── experiments/
│   └── results/                     # ★ 真实实验结果（已全部跑完）
│       ├── tp_thgn_v3_multiseed.json    # TP-THGN 3-seed结果
│       ├── comparison_results.json       # 6个基线模型对比
│       ├── ablation_v3_results.json      # 4项消融实验
│       ├── explainability_cases.json     # 3个可解释性案例（TP/FN/TN）
│       └── training_curves.json          # 300epoch训练曲线数据
├── .claude/
│   ├── worktrees/
│   │   ├── tp-thgn-design/          # 代码实现worktree（Phase 0-6全部完成）
│   │   ├── gpu-implementation/       # GPU实现worktree
│   │   ├── thesis-writing/           # ★ 论文撰写worktree（主要工作区）
│   │   └── agents-ppt/              # PPT制作worktree
│   └── skills/
│       ├── research-paper-writing/   # ML论文写作skill
│       └── academic-paper-skills/    # 学术论文strategist+composer skill
└── README.md
```

### ★ 论文文件位置（最新版）

```
.claude/worktrees/thesis-writing/thesis/
├── abstract.md              # 中英文摘要（~800字中文 + ~500词英文）
├── chapter1.md              # 第1章 绪论（~7,500字）
├── chapter2.md              # 第2章 理论基础与文献综述（~7,700字）
├── chapter3.md              # 第3章 数据构建与模型设计（~6,900字）
├── chapter4.md              # 第4章 实证分析（~6,500字，已加厚）
├── chapter5.md              # 第5章 可解释归因与实践启示（~5,500字，已加厚）
├── chapter6.md              # 第6章 结论与展望（~3,100字）
├── references.md            # 参考文献（60篇，GB/T 7714-2015格式）
├── acknowledgements.md      # 致谢（新增）
├── full_thesis.md           # 全文合并版（1012行）
├── THESIS_PLAN.md           # 论文框架规划
├── REVISION_NOTES.md        # 修订备忘（用户反馈记录）
└── HANDOFF.md               # 工作交接文档
```

---

## 三、模型技术方案摘要

TP-THGN v3 = Feature-dominant架构，核心组件：

1. **深层特征编码器**: 2层MLP (25d→128d)，BatchNorm + Dropout(0.3)
2. **时序衰减模块**: exp(-softplus(λ)·Δt)，可学习衰减系数
3. **TP-GraphSMOTE**: 嵌入空间KNN插值 + 拉普拉斯正则化(β=0.01)，过采样率1.5
4. **门控图增强×2层**: 关系加权聚合(softmax) + 门控融合 G=σ(W[h;h_neigh])
5. **Focal Loss**: γ=2.0 + 自适应类别权重 clamp(√(N_norm/N_fraud), 1.5, 5.0)
6. **TriExplainer**: 特征归因(梯度×输入) + 关系归因(权重向量) + 子图归因(BFS+概率阈值)

关键设计理念: **低同质性(0.12-0.25)的欺诈图中，特征信号主导，图结构辅助增强**

---

## 四、实验结果数据速查

### 主对比实验 (comparison_results.json)

| 模型 | F1 | AUC | Recall | Precision |
|------|-----|-----|--------|-----------|
| LR | 0.761 | 0.983 | 0.962 | 0.630 |
| XGBoost | 0.921 | 0.989 | 0.910 | 0.932 |
| GCN | 0.445±0.014 | 0.860±0.004 | 0.336±0.023 | 0.662±0.022 |
| GAT | 0.403±0.056 | 0.867±0.001 | 0.286±0.049 | 0.693±0.021 |
| GraphSAGE | 0.913±0.004 | 0.988±0.000 | 0.898±0.011 | 0.929±0.009 |
| **TP-THGN** | **0.902±0.009** | **0.985±0.001** | **0.885±0.010** | **0.918±0.012** |

### 消融实验 (ablation_v3_results.json)

| 变体 | F1 | AUC | Precision |
|------|-----|-----|-----------|
| Full (TP-THGN v3) | 0.898 | 0.985 | 0.924 |
| w/o Graph Enhancement | 0.906 | 0.986 | 0.922 |
| w/o TP-GraphSMOTE | 0.890 | 0.966 | 0.934 |
| w/o Focal Loss (CE) | 0.902 | 0.984 | 0.924 |
| w/o Learnable Gate | 0.888 | 0.984 | 0.880 |

### 可解释性案例 (explainability_cases.json)

- Case 1 (TP): Node#11245, P(fraud)=0.849, 8个高风险邻居, 团伙欺诈模式
- Case 2 (FN): Node#9561, P(fraud)=0.456, 孤立型欺诈者, feat_19异常
- Case 3 (TN): Node#9706, P(fraud)=0.019, 正常交易, 邻域干净

---

## 五、已完成工作清单

| 工作项 | 状态 | 说明 |
|--------|------|------|
| 全部代码实现(Phase 0-6) | ✅ | tp-thgn-design worktree，含TD-GRU-GNN/TP-GraphSMOTE/XAttention/TriExplainer/baselines |
| GPU优化实现 | ✅ | gpu-implementation worktree，tp_thgn_gpu.py |
| 全部实验运行 | ✅ | 对比/消融/可解释/训练曲线，结果在experiments/results/ |
| 第1章 绪论 | ✅ | 研究背景(信用卡概念铺垫)、问题与意义、内容与路线、方法、创新点 |
| 第2章 文献综述 | ✅ | 欺诈理论、非平衡方法、GNN/时序图、可解释AI、研究缺口 |
| 第3章 模型设计 | ✅ | 数据描述、图构建、TP-GraphSMOTE、TP-THGN架构、TriExplainer、实验方案 |
| 第4章 实证分析 | ✅ | 主实验、消融、训练曲线、多seed稳健性（段落已加厚） |
| 第5章 可解释性 | ✅ | 权重分析、3案例深度解读、STR辅助、成本分析（段落已加厚） |
| 第6章 结论 | ✅ | 结论、实践启示、不足、展望 |
| 中英文摘要 | ✅ | 约800字中文 + 500词英文 |
| 参考文献 | ✅ | 60篇，GB/T 7714格式，近三年38篇 |
| 致谢 | ✅ | acknowledgements.md |
| 概念铺垫修订 | ✅ | 第1-3章已补充基础概念解释 |
| 逻辑衔接修订 | ✅ | 各章增加引言段落承上启下 |
| 段落加厚 | ✅ | 第4章(结果分析)、第5章(案例解读)已消除碎片化短段落 |
| 全文合并 | ✅ | full_thesis.md (1012行, ~35,000-40,000字) |
| 学术论文skills安装 | ✅ | ~/.claude/skills/ 下三套skill |

---

## 六、待完成工作（优先级排序）

### P0 - 高优先级

1. **Markdown → Word格式转换**
   - 需要学校论文模板（.docx/.dotx），用户尚未提供
   - 建议命令: `pandoc full_thesis.md -o thesis.docx --reference-doc=template.docx`
   - 需处理: 公式(LaTeX→Word公式编辑器)、表格(→三线表)、图片插入
   - 页眉页脚、页码、目录自动生成

2. **公式编号**
   - 第3章约15个公式需添加编号（已在Markdown中标注为 `\tag{3-1}` 等）
   - 转Word后需转为Word公式编辑器格式

### P1 - 中优先级

3. **图表完善**
   - 训练曲线图(training_curves.json → matplotlib → PDF/PNG)已生成
   - 转Word后需确认图片清晰度(300dpi+)和三线表格式
   - 添加规范图注("图4-1 TP-THGN训练曲线")

4. **交叉引用检查**
   - "如表X所示"、"如图X所示"需在Word中建立交叉引用
   - 参考文献[1][2]需转为Word尾注/脚注格式

### P2 - 低优先级

5. **导师审阅修改** - 提交后等反馈
6. **答辩PPT制作** - agents-ppt worktree已有基础
7. **查重降重** - 提交前最后步骤

---

## 七、用户偏好与工作原则

1. **自主决策**: 用户要求agent自行做决策，除非遇到真正无法解决的问题才提问
2. **内容厚度**: 论文段落需要有充实的论证展开，禁止"两行一段"的AI碎片化写法
3. **金融为体**: 技术服务于金融风控目标，每个模块对应金融业务痛点
4. **实证论文风格**: 重实验证据和结果讨论，轻纯技术推导
5. **中文撰写**: 论文正文中文，代码和模型命名英文
6. **引用格式**: GB/T 7714-2015
7. **数据真实性**: 所有实验数据均为真实运行结果，禁止编造或使用"预期值"

---

## 八、技术环境

| 配置项 | 规格 |
|--------|------|
| GPU | NVIDIA RTX 4070 Laptop (8GB VRAM) |
| Python | 3.12 |
| 框架 | PyTorch 2.x + CUDA, torch.sparse |
| Conda环境 | thg-oafn |
| 操作系统 | Windows 11 |
| 训练时长 | ~15-20分钟/完整训练(300 epochs) |
| 峰值显存 | ~2GB |

---

## 九、关键注意事项（给后续agents）

1. **论文正式版在 thesis-writing worktree 中**，不是主分支的thesis/目录（该目录已删除）
2. **实验数据在主分支 experiments/results/ 中**，各worktree中有副本
3. **不要重新跑实验** — 所有结果已经完整，数据是真实的GPU运行结果
4. **不要重写论文章节** — 只需在现有基础上修改/润色/格式化
5. **full_thesis.md 是各章拼接而成** — 修改时应修改对应的chapter文件，然后重新cat合并
6. **XGBoost和GraphSAGE的F1高于TP-THGN** — 论文中已有合理解释(可解释性优势)，不是bug
7. **图增强去除后F1微升** — 这是Feature-dominant设计的核心验证，论文中已充分讨论
8. **3个seed而非5个** — 论文4.5.1节已说明合理性，不需要补跑
