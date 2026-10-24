# -*- coding: utf-8 -*-
"""第01章实践任务：300328 日简单/对数收益率的经验密度 vs 正态密度对比图。"""
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import thsdk
from scipy import stats

logging.disable(logging.CRITICAL)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).parent
CODE = "USZA300328"

with thsdk.THS() as ths:
    resp = ths.klines(CODE, interval="day",
                      start_time=datetime(2025, 10, 24), end_time=datetime(2026, 10, 24))
    if not resp.success:
        raise SystemExit(f"取数失败：{resp.error}")
    df = resp.df.copy()

df["时间"] = pd.to_datetime(df["时间"])
df = df.sort_values("时间").reset_index(drop=True)
df["简单收益率"] = df["收盘价"].pct_change()
df["对数收益率"] = np.log(df["收盘价"]).diff()
ret = df.dropna(subset=["简单收益率", "对数收益率"]).copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
fig.suptitle("300328 宜安科技 · 日收益率经验密度 vs 正态密度（近一年，241 个交易日）",
             fontsize=14, fontweight="bold")

series = [
    ("简单收益率", "R_t = P_t/P_(t-1) − 1", axes[0], "#A32D2D"),
    ("对数收益率", "r_t = ln(P_t/P_(t-1))", axes[1], "#185FA5"),
]
stats_lines = []
for name, formula, ax, color in series:
    x = ret[name].values
    mu, sd = x.mean(), x.std(ddof=1)
    sk = stats.skew(x)
    ku = stats.kurtosis(x)          # 超额峰度
    jb, jb_p = stats.jarque_bera(x)

    # 经验密度：直方图（密度归一）+ 高斯核平滑
    ax.hist(x, bins=40, density=True, color=color, alpha=0.35,
            edgecolor=color, linewidth=0.6, label="经验密度（直方图）")
    kde = stats.gaussian_kde(x)
    xs = np.linspace(x.min() * 1.15, x.max() * 1.15, 400)
    ax.plot(xs, kde(xs), color=color, linewidth=1.6, label="核平滑密度")
    # 同均值同方差正态密度
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color="#444441", linewidth=1.6,
            linestyle="--", label="正态密度 N(μ, σ²)")
    ax.axvline(0, color="#888780", linewidth=0.7)
    ax.set_title(f"{name}   {formula}", fontsize=11)
    ax.set_xlabel("收益率")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    stats_lines.append(f"{name}: 偏度 {sk:+.2f}  超额峰度 {ku:+.2f}  JB={jb:.1f} (p={jb_p:.1e})")

axes[0].set_ylabel("密度")

# 用对数收益标注尾部超出区域（厚尾可视化）
r = ret["对数收益率"].values
mu, sd = r.mean(), r.std(ddof=1)
xs = np.linspace(r.min() * 1.15, r.max() * 1.15, 400)
kde = stats.gaussian_kde(r)
axes[1].fill_between(xs, kde(xs), stats.norm.pdf(xs, mu, sd),
                     where=kde(xs) > stats.norm.pdf(xs, mu, sd),
                     color="#F09595", alpha=0.5, linewidth=0)
axes[1].text(0.03, 0.55, "粉色填充 = 经验密度超出正态的部分\n（尾部更高、中心更尖 → 尖峰厚尾）",
             transform=axes[1].transAxes, fontsize=9, color="#A32D2D", va="top")

fig.text(0.99, 0.005, "   |   ".join(stats_lines), ha="right", fontsize=9, color="#5F5E5A")
fig.tight_layout(rect=(0, 0.035, 1, 0.94))
fig.savefig(OUT_DIR / "300328_收益率密度对比_近一年.png", dpi=150)

# 输出尾部统计：超出 ±2σ、±3σ 的天数比例 vs 正态理论值
for name in ["简单收益率", "对数收益率"]:
    x = ret[name].values
    mu, sd = x.mean(), x.std(ddof=1)
    z = np.abs((x - mu) / sd)
    n = len(x)
    print(f"{name}: 超出±2σ 实际 {(z>2).sum()}/{n} = {(z>2).mean()*100:.1f}% (正态理论 4.6%) | "
          f"±3σ 实际 {(z>3).sum()}/{n} = {(z>3).mean()*100:.1f}% (正态理论 0.27%)")
for line in stats_lines:
    print(line)
