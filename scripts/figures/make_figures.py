"""E5: paper figures (all from archived JSONs, no recomputation).

Usage: python scripts/figures/make_figures.py
Output: paper/figures/fig1..fig6 (300 dpi PNG)
Style: journal-grade, Chinese labels (Microsoft YaHei), print-safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

GRAY, BLUE, RED, GOLD = "#404040", "#2b6a99", "#b2182b", "#b8860b"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def fig1_ablation():
    """Feature ablation waterfall: layerwise AUPRC gain."""
    stages = ["基础因果窗口\n(33列)", "+账户对/实体结构\n(60列)", "+对手方画像\n(78列)", "+超参正则化\n(78列)"]
    vals = [0.4119, 0.4888, 0.5846, 0.6276]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.bar(range(4), vals, width=0.58, color=[GRAY, BLUE, BLUE, GOLD])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.4f}", ha="center", fontsize=9)
    for i in range(1, 4):
        ax.annotate(f"+{vals[i]-vals[i-1]:.3f}", xy=(i - 0.5, (vals[i] + vals[i-1]) / 2),
                    ha="center", fontsize=8.5, color=RED)
    ax.set_xticks(range(4), stages, fontsize=9)
    ax.set_ylabel("测试集 AUPRC")
    ax.set_ylim(0.36, 0.68)
    ax.set_title("图1  因果行为特征三层消融（每层增量单独可测）", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_ablation.png")
    plt.close(fig)


def fig2_counterfactual_waterfall():
    """Case-level counterfactual waterfall (case 0) with feature-level contrast."""
    d = load(RES / "routeB_counterfactual_v2.json")
    case = d["cases"][0]
    groups = sorted(case["groups"].items(), key=lambda kv: -kv[1]["drop"])
    names = [k for k, _ in groups]
    drops = [v["drop"] for _, v in groups]
    zh = {"pair_frequency": "账户对高频互转", "entity_diff": "实体内差分", "burst_activity": "突发高频",
          "daily_volume": "日交易量", "weekly_volume": "周交易量", "src_profile": "源账户画像",
          "dst_profile": "对手方画像", "amount": "金额族"}
    labels = [zh.get(n, n) for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(range(len(labels)), drops, width=0.6, color=[RED if v > 0.5 else GRAY for v in drops])
    for i, v in enumerate(drops):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.axhline(0.055, color=BLUE, ls="--", lw=1.2)
    ax.text(len(labels) - 0.4, 0.075, "单特征归零上限 0.055", ha="right", fontsize=8.5, color=BLUE)
    ax.set_xticks(range(len(labels)), labels, rotation=32, ha="right", fontsize=8.5)
    ax.set_ylabel("风险分降幅 (单独干预)")
    ax.set_title(f"图2  被标记案例的反事实瀑布（基础分 {case['base_score']:.3f}，干预=置回普通账户水平）", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_counterfactual_waterfall.png")
    plt.close(fig)


def fig3_frontier():
    """Accuracy-attributability frontier."""
    d = load(RES / "routeB_counterfactual_v2.json")
    pts = d["accuracy_attributability_frontier"]
    k = [p["n_features"] for p in pts]
    auprc = [p["test_auprc"] for p in pts]
    sens = [p["pair_group_mean_drop"] for p in pts]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.6))
    ax1.plot(k, auprc, "o-", color=BLUE, lw=1.8, ms=6, label="AUPRC（检测精度）")
    ax1.set_xlabel("特征数（按全局 SHAP 重要性取前 k 列重训）")
    ax1.set_ylabel("AUPRC", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(k, sens, "s--", color=RED, lw=1.8, ms=6, label="主归因族敏感度（可归因性）")
    ax2.set_ylabel("组级干预降幅", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0.4, 1.05)
    for x, y in zip(k, auprc):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax1.set_title("图3  精度-可归因性权衡前沿", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_frontier.png")
    plt.close(fig)


def fig4_shap():
    """Global SHAP top-12."""
    d = load(next((RES / "attribution").glob("xgb_shap_*.json")))
    top = d["global_ranking_top20"][:12][::-1]
    zh = {"pair_w7d_count": "账户对7天互转频次", "payment_format": "支付方式", "src_w1h_count": "源账户1小时笔数",
          "src_w24h_count": "源账户24小时笔数", "src_w7d_count": "源账户7天笔数", "amount_paid": "交易金额",
          "pair_w1h_count": "账户对1小时频次", "pair_past_count": "账户对历史频次", "dst_bro_w7d_count": "对手跨行7天计数",
          "src_bro_w7d_count": "源跨行7天计数", "dow": "星期几", "prof_src_7": "源画像:不同对手数"}
    names = [zh.get(t["feature"], t["feature"]) for t in top]
    vals = [t["mean_abs_shap"] for t in top]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.barh(range(len(names)), vals, height=0.62, color=[RED if v == max(vals) else BLUE for v in vals])
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("图4  全局特征重要性 Top-12（全部具备反洗钱业务语义）", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_shap.png")
    plt.close(fig)


def fig5_gnn_diagnosis():
    """Seven-config GNN diagnostic chain."""
    labels = ["无图对照\n(同训练循环)", "GraphSAGE\n+8维画像", "GATv2\n注意力", "GATv2\n+20维强画像",
              "评分头warmup\n(污染版)", "干净warmup\n(test)", "冻结嵌入\n注入XGB"]
    vals = [0.0885, 0.0297, 0.0120, 0.0125, 0.0215, 0.0029, 0.5922]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    colors = [GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GOLD]
    ax.bar(range(7), vals, width=0.6, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.4f}" if v < 0.1 else f"{v:.4f}", ha="center", fontsize=8)
    ax.axhline(0.6276, color=RED, ls="--", lw=1.3)
    ax.text(0.02, 0.645, "纯手工因果特征 0.6276", fontsize=8.5, color=RED)
    ax.set_xticks(range(7), labels, fontsize=7.8)
    ax.set_ylabel("测试集 AUPRC")
    ax.set_title("图5  端到端图神经网络七配置诊断链（种子42）", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_gnn_diagnosis.png")
    plt.close(fig)


def fig6_leakage_forensics():
    """Leakage forensics 2x2."""
    d = load(RES / "leakage_forensics.json")
    m = d["matrix"]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    x = [0, 1]
    w = 0.32
    clean = [m["temporal"]["clean"], m["random"]["clean"]]
    leaked = [m["temporal"]["leaked_labels"], m["random"]["leaked_labels"]]
    ax.bar([i - w / 2 for i in x], clean, width=w, color=BLUE, label="干净协议")
    ax.bar([i + w / 2 for i in x], leaked, width=w, color=RED, label="标签泄漏协议")
    for i, (c, l) in enumerate(zip(clean, leaked)):
        ax.text(i - w / 2, c + 0.012, f"{c:.4f}", ha="center", fontsize=8.5)
        ax.text(i + w / 2, l + 0.012, f"{l:.4f}", ha="center", fontsize=8.5)
        ax.annotate(f"+{l-c:.3f}", (i, max(c, l) + 0.06), ha="center", fontsize=9.5, color=RED, fontweight="bold")
    ax.set_xticks(x, ["时序切分\n(本文协议)", "随机切分\n(文献常见)"])
    ax.set_ylabel("测试集 AUPRC")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("图6  泄漏法医 2×2：协议缺陷的量化虚高", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_leakage_forensics.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_ablation()
    fig2_counterfactual_waterfall()
    fig3_frontier()
    fig4_shap()
    fig5_gnn_diagnosis()
    fig6_leakage_forensics()
    for p in sorted(OUT.glob("fig*.png")):
        print(f"{p.name}  {p.stat().st_size/1024:.0f} KB")
