# 弃用与删除记录

## 2026-09-17：Amazon 伪时序主线

以下内容在 Phase 1 删除，历史仍可通过 Git commit 查看：

| 删除范围 | 删除原因 | 替代内容 |
|---|---|---|
| `data/Amazon.mat`、`data/Amazon.zip` | 是评论用户图；没有原始事件 ID 和真实时间，不能用于金融交易时序结论 | IBM AML HI-Small 真实带时间交易；Elliptic++ 外部验证 |
| 旧 `models/`、`train*.py`、`experiments/` | 包含伪时序、全图统计/标签路径或不适用于交易事件图的过采样设计 | `src/finrisk/` 下的因果时序异构图实现 |
| 旧 `figures/`、`experiments/results/` | 依赖已撤销的实验设计，不能作为论文证据 | 每阶段以真实保存的预测重新生成图表 |
| 旧汇总文件 | 将 Amazon 评论图写成信用卡交易，并保留未经重新审计的指标 | `PROJECT_SUMMARY.md` 与阶段报告 |

删除不是否定 Git 历史，而是避免旧内容继续被误认为现行论文证据。
