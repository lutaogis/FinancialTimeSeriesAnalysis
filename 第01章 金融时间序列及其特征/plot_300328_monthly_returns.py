# -*- coding: utf-8 -*-
"""第01章实践任务：300328 近 10 年月收益率（简单/对数）时序图 + 经验密度 vs 正态密度。"""
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

# ---- 取 10 年日 K ----
with thsdk.THS() as ths:
    resp = ths.klines(CODE, interval="day",
                      start_time=datetime(2016, 10, 24), end_time=datetime(2026, 10, 24))
    if not resp.success:
        raise SystemExit(f"取数失败：{resp.error}")
    daily = resp.df.copy()

daily["时间"] = pd.to_datetime(daily["时间"])
daily = daily.sort_values("时间").reset_index(drop=True)

# ---- 聚合到月末（取每月最后一个交易日的收盘价）----
monthly = daily.set_index("时间")["收盘价"].resample("ME").last().dropna()
months = monthly.index

# 月简单收益与月对数收益
m_simple = monthly.pct_change().dropna()
m_log = np.log(monthly).diff().dropna()
n = len(m_simple)

summary = pd.DataFrame({"收盘价": monthly, "简单收益率": m_simple, "对数收益率": m_log}).dropna()
summary.to_csv(OUT_DIR / "300328_月收益率_近10年.csv", encoding="utf-8-sig")

# ---- 图 1：月收益率时序（涨红跌绿）----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.4]})
fig.suptitle("300328 宜安科技 · 月收盘价与月收益率（近 10 年）", fontsize=15, fontweight="bold")

ax1.plot(months, monthly, color="#185FA5", linewidth=1.3)
ax1.set_ylabel("月末收盘价（元）")
ax1.grid(alpha=0.25)
ax1.set_title("月末收盘价", fontsize=11, loc="left")

x_pos = summary.index
w = 20  # 柱宽（天）
ax2.bar(x_pos, summary["简单收益率"] * 100, width=w, color="#A32D2D", alpha=0.55, label="月简单收益率")
ax2.bar(x_pos, summary["对数收益率"] * 100, width=w, color="#185FA5", alpha=0.55, label="月对数收益率")
ax2.axhline(0, color="#888780", linewidth=0.8)
ax2.set_ylabel("月收益率（%）")
ax2.grid(alpha=0.25)
ax2.legend(loc="upper right", fontsize=9)
ax2.set_title("月收益率：月频下两种口径的差异已肉眼可见（对数收益在大涨/大跌月系统性偏离）", fontsize=11, loc="left")

diff_max = (summary["简单收益率"] - summary["对数收益率"]).abs().max()
fig.text(0.99, 0.005,
         f"{n} 个月  月均 {m_simple.mean()*100:+.2f}%  月波动 {m_simple.std()*100:.2f}%  "
         f"年化(月×√12) {m_simple.std()*np.sqrt(12)*100:.1f}%  |简单−对数| 最大 {diff_max*100:.1f}pp",
         ha="right", fontsize=9, color="#5F5E5A")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(OUT_DIR / "300328_月收益率_近10年.png", dpi=150)

# ---- 图 2：月收益率经验密度 vs 正态密度 ----
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
fig2.suptitle("300328 宜安科技 · 月收益率经验密度 vs 正态密度（近 10 年）", fontsize=14, fontweight="bold")

lines = []
for name, ax, color in [("简单收益率", axes2[0], "#A32D2D"), ("对数收益率", axes2[1], "#185FA5")]:
    x = summary[name].values
    mu, sd = x.mean(), x.std(ddof=1)
    sk, ku = stats.skew(x), stats.kurtosis(x)
    jb, jb_p = stats.jarque_bera(x)

    ax.hist(x, bins=20, density=True, color=color, alpha=0.35,
            edgecolor=color, linewidth=0.6, label="经验密度（直方图）")
    kde = stats.gaussian_kde(x)
    xs = np.linspace(x.min() * 1.15, x.max() * 1.15, 400)
    ax.plot(xs, kde(xs), color=color, linewidth=1.6, label="核平滑密度")
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color="#444441", linewidth=1.6,
            linestyle="--", label="正态密度 N(μ, σ²)")
    ax.axvline(0, color="#888780", linewidth=0.7)
    ax.set_title(f"月{name}", fontsize=11)
    ax.set_xlabel("收益率")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    z = np.abs((x - mu) / sd)
    lines.append((name, sk, ku, jb, jb_p, (z > 2).mean(), (z > 3).mean(), n))

axes2[0].set_ylabel("密度")
fig2.text(0.99, 0.005,
          "   |   ".join(f"{name}: 偏度 {sk:+.2f}  超额峰度 {ku:+.2f}  JB={jb:.1f}(p={jb_p:.2f})"
                         f"  >±2σ {p2*100:.1f}%  >±3σ {p3*100:.1f}%"
                         for name, sk, ku, jb, jb_p, p2, p3, _ in lines),
          ha="right", fontsize=8.5, color="#5F5E5A")
fig2.tight_layout(rect=(0, 0.035, 1, 0.94))
fig2.savefig(OUT_DIR / "300328_月收益率密度对比_近10年.png", dpi=150)

print(f"月样本 {n} 个")
print(f"简单: 月均 {m_simple.mean()*100:+.3f}%  月波动 {m_simple.std()*100:.2f}%  年化波动 {m_simple.std()*np.sqrt(12)*100:.1f}%")
print(f"对数: 月均 {m_log.mean()*100:+.3f}%  月波动 {m_log.std()*100:.2f}%  年化波动 {m_log.std()*np.sqrt(12)*100:.1f}%")
print(f"|简单-对数| 平均 {(summary['简单收益率']-summary['对数收益率']).abs().mean()*100:.2f}pp, 最大 {diff_max*100:.2f}pp")
for name, sk, ku, jb, jb_p, p2, p3, cnt in lines:
    print(f"{name}: 偏度 {sk:+.3f}  超额峰度 {ku:+.3f}  JB {jb:.2f} (p={jb_p:.3f})  "
          f"±2σ外 {p2*100:.1f}%(正态4.6%)  ±3σ外 {p3*100:.1f}%(正态0.27%)")
