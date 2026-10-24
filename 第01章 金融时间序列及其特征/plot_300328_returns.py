# -*- coding: utf-8 -*-
"""第01章实践任务：抓取 300328 宜安科技近一年日 K，计算日简单收益率并绘图。"""
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import thsdk

logging.disable(logging.CRITICAL)  # 关闭 thsdk 游客账户的日志刷屏

# 中文显示（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).parent
CODE = "USZA300328"  # 前缀规则：6 开头 -> USHA，其余 -> USZA

with thsdk.THS() as ths:
    resp = ths.klines(CODE, interval="day",
                      start_time=datetime(2025, 10, 24), end_time=datetime(2026, 10, 24))
    if not resp.success:
        raise SystemExit(f"取数失败：{resp.error}")
    df = resp.df.copy()

df["时间"] = pd.to_datetime(df["时间"])
df = df.sort_values("时间").reset_index(drop=True)

# 日简单收益率：R_t = P_t / P_{t-1} - 1（用复权收盘价）
df["简单收益率"] = df["收盘价"].pct_change()

ret = df.dropna(subset=["简单收益率"]).copy()
ret[["时间", "收盘价", "简单收益率"]].to_csv(
    OUT_DIR / "300328_日简单收益率_近一年.csv", index=False, encoding="utf-8-sig")

n = len(ret)
mu = ret["简单收益率"].mean()
sigma = ret["简单收益率"].std()
ann_vol = sigma * (252 ** 0.5)
max_up = ret.loc[ret["简单收益率"].idxmax()]
max_dn = ret.loc[ret["简单收益率"].idxmin()]

# 绘图：上价格、下收益率（涨红跌绿，A 股惯例）
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw={"height_ratios": [1.2, 1]},
)
fig.suptitle("300328 宜安科技 · 近一年日行情与简单收益率", fontsize=15, fontweight="bold")

ax1.plot(df["时间"], df["收盘价"], color="#185FA5", linewidth=1.3)
ax1.set_ylabel("收盘价（元）")
ax1.grid(alpha=0.25)
ax1.set_title("收盘价", fontsize=11, loc="left")

colors = ["#A32D2D" if v >= 0 else "#0F6E56" for v in ret["简单收益率"]]
ax2.bar(ret["时间"], ret["简单收益率"] * 100, color=colors, width=1.0)
ax2.axhline(0, color="#888780", linewidth=0.8)
ax2.set_ylabel("日简单收益率（%）")
ax2.grid(alpha=0.25)
ax2.set_title("日简单收益率  R_t = P_t / P_(t-1) − 1", fontsize=11, loc="left")

fig.text(
    0.99, 0.005,
    f"样本 {n} 个交易日  日均 {mu*100:+.3f}%  日波动 {sigma*100:.2f}%  年化波动 {ann_vol*100:.1f}%"
    f"  最大涨 {max_up['简单收益率']*100:+.2f}%({max_up['时间']:%m-%d})  最大跌 {max_dn['简单收益率']*100:+.2f}%({max_dn['时间']:%m-%d})",
    ha="right", fontsize=9, color="#5F5E5A",
)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(OUT_DIR / "300328_日简单收益率_近一年.png", dpi=150)
print(f"样本 {n} 天 | 均值 {mu*100:+.3f}%/日 | 日波动 {sigma*100:.2f}% | 年化波动 {ann_vol*100:.1f}%")
print(f"最大单日涨 {max_up['简单收益率']*100:+.2f}% @ {max_up['时间']:%Y-%m-%d}")
print(f"最大单日跌 {max_dn['简单收益率']*100:+.2f}% @ {max_dn['时间']:%Y-%m-%d}")
