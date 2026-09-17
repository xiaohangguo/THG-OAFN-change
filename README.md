# THG-OAFN Financial Risk Graph Lab

本仓库现用于金融交易反洗钱风险监测的可复现实验，不再将 Amazon 评论图或伪造时间戳作为论文主实验。

## 当前研究主线

**题目暂定：** 面向反洗钱交易风险监测的因果时序异构图神经网络研究。

主数据为 IBM AML HI-Small；Elliptic++ 用于外部金融场景验证。模型只使用在预测时刻及之前可见的交易、账户和关系信息。

项目分阶段计划、当前状态、删除内容及原因见 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)。数据许可、下载和本地存储规范见 [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md)。

## 快速开始

1. 在 D 盘创建的本地工作区 `D:\THG-OAFN-Financial-Experiment` 中保存运行元数据和临时数据缓存。
2. 按数据治理文档下载 IBM AML HI-Small。原始数据不提交到 GitHub。
3. 将交易文件路径传给第一阶段审计器：

```powershell
python scripts/phase1_data_audit.py `
  --transactions "D:\THG-OAFN-Financial-Experiment\data_cache\ibm_aml\HI-Small_Trans.csv" `
  --accounts "D:\THG-OAFN-Financial-Experiment\data_cache\ibm_aml\HI-Small_accounts.csv" `
  --output "D:\THG-OAFN-Financial-Experiment\metadata\ibm_hi_small_audit.json"
```

4. 审计通过后，后续阶段只从该审计清单构建特征、切分和训练集。

## 仓库结构

```text
configs/       固化的数据、切分、训练配置
docs/          数据治理、实验协议、阶段总结与弃用记录
scripts/       可直接运行的审计、构图、训练与评估入口
src/finrisk/   金融时序图数据与模型实现
tests/         可重复的单元与泄漏防护测试
```

## 重要约束

- 原始数据、模型权重、预测概率和大图缓存不得提交 GitHub。
- 任何特征只能由当前时点及其历史交易构建。
- 阈值只能由验证集确定；测试集只用于最终一次报告。
- AUPRC、Recall@K、Precision@K 和成本敏感指标是主指标；F1 是辅助指标。
