"""Assemble the graduate thesis manuscript (UTF-8 BOM) from chapter files.

Usage: python scripts/assemble_thesis.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TH = PAPER / "thesis"

FRONT = """# 基于因果行为画像与组级反事实归因的反洗钱交易风险检测研究

（封面占位，按研究生处官方模板填充：学校代码 / 学号 / 密级 / 分类号 / 学位类别、学院、专业、研究方向、研究生、指导教师、答辩日期）

## 原创*.*性声明

（占位，按学校官方模板文本粘贴）

## 版权使用授权书

（占位，按学校官方模板文本粘贴）

---

"""

TOC = """## 目　录

- 摘要
- Abstract
- 第1章　绪论
  - 1.1 研究背景与意义（1.1.1 洗钱及其经济危害；1.1.2 反洗钱监管体系；1.1.3 金融机构反洗钱实践的技术痛点；1.1.4 研究意义）
  - 1.2 国内外研究现状（1.2.1 基于机器学习的洗钱交易检测；1.2.2 图神经网络在金融交易图上的应用；1.2.3 可解释性与反事实归因；1.2.4 研究现状简评）
  - 1.3 研究内容与创新点
  - 1.4 论文组织结构
- 第2章　相关理论与技术基础
  - 2.1 反洗钱业务框架（2.1.1 洗钱三阶段的行为特征；2.1.2 以风险为本的监测与可疑活动报告）
  - 2.2 集成学习与梯度提升树（2.2.1 决策树与集成学习；2.2.2 XGBoost 的正则化目标）
  - 2.3 类别不平衡学习与评估指标（2.3.1 不平衡学习策略；2.3.2 平均精度与不平衡场景下的指标选择）
  - 2.4 图神经网络（2.4.1 交易图建模与消息传递；2.4.2 GraphSAGE 与 GATv2）
  - 2.5 可解释机器学习与反事实归因（2.5.1 SHAP 与 TreeSHAP；2.5.2 反事实解释与归因的验收）
  - 2.6 评估协议与信息泄漏
  - 2.7 本章小结
- 第3章　基于因果行为画像与组级反事实归因的检测框架
  - 3.1 问题定义与无泄漏时序协议
  - 3.2 因果行为画像特征框架
  - 3.3 组级反事实归因协议
  - 3.4 精度-可归因性权衡前沿
  - 3.5 与已有工作的区别
- 第4章　实验与结果分析
  - 4.1 数据与实验设置
  - 4.2 主结果：检测精度与审核作业读数
  - 4.3 归因：从特征重要性到反事实瀑布
  - 4.4 精度-可归因性权衡前沿
  - 4.5 端到端图神经网络与自监督表示的系统诊断
  - 4.6 泄漏法医：文献高指标的一种解剖
  - 4.7 稳健性与复现
- 第5章　结论与展望
  - 5.1 主要结论
  - 5.2 局限与展望
- 参考文献
- 致谢
- 攻读硕士学位期间取得的研究成果

---

"""

BACK = """
## 致谢

（初稿占位，请按个人情况改写）本论文的选题与完成，离不开导师XXX教授的悉心指导——从研究问题的聚焦、实验协议的严格性，到论文写作的打磨，老师始终以高标准要求并给予耐心点拨，谨致谢忱。感谢学院诸位任课老师在金融与数据科学交叉知识上打下的基础；感谢同门与好友在实验讨论与论文互评中的无私帮助；感谢家人一贯的理解与支持。最后，感谢开源社区与公开数据集的建设者，本研究建立在他们的工作之上。

## 攻读硕士学位期间取得的研究成果

（占位：学术论文、竞赛、专利等，无则按学校规定处理）
"""


def main() -> None:
    abstract = (TH / "ch0-abstract.md").read_text(encoding="utf-8")
    zh, en = abstract.split("# Abstract")
    zh_body = zh.replace("# 摘要", "").strip()
    en_body = en.strip()

    parts = [FRONT, "## 摘要\n\n" + zh_body + "\n\n---\n\n# Abstract\n\n" + en_body + "\n\n" + TOC]
    for sub in [
        "thesis/ch1-introduction.md",
        "thesis/ch2-theory.md",
        "thesis/ch3-method.md",
        "thesis/ch4-experiments.md",
        "thesis/ch5-conclusion.md",
        "07-references.md",
    ]:
        parts.append((PAPER / sub).read_text(encoding="utf-8").strip() + "\n\n---\n\n")
    parts.append(BACK)

    out = TH / "thesis-manuscript.md"
    out.write_text("".join(parts), encoding="utf-8-sig")
    txt = out.read_text(encoding="utf-8-sig")
    n_chars = len(re.sub(r"\s", "", txt))
    residue = re.findall(r"合并稿|装配：", txt)
    print(f"OK {out} | {out.stat().st_size} bytes | {n_chars} chars | residue: {residue or 'clean'}")


if __name__ == "__main__":
    main()
