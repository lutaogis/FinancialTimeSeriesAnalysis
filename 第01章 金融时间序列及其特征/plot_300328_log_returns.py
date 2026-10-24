# -*- coding: utf-8 -*-
"""第01章实践任务：300328 宜安科技近一年对数收益率，并与简单收益率对比。"""
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import thsdk

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

# 简单收益 R_t = P_t/P_{t-1} - 1 ；对数收益 r_t = ln(P_t/P_{t-1}) = ln(1+R_t)
df["简单收益率"] = df["收盘价"].pct_change()
df["对数收益率"] = np.log(df["收盘价"]).diff()

ret = df.dropna(subset=["对数收益率"]).copy()
ret[["时间", "收盘价", "简单收益率", "对数收益率"]].to_csv(
    OUT_DIR / "300328_对数收益率_近一年.csv", index=False, encoding="utf-8-sig")

n = len(ret)
r_log = ret["对数收益率"]
mu, sigma = r_log.mean(), r_log.std()
m3 = ((r_log - mu) ** 3).mean()
m4 = ((r_log - mu) ** 4).mean()
skew = m3 / sigma ** 3
kurt = m4 / sigma ** 4
jb = n / 6 * (skew ** 2 + (kurt - 3) ** 2 / 4)
diff = (ret["简单收益率"] - ret["对数收益率"]).abs()
big = ret[ret["简单收益率"].abs() > 0.05]

# 对比图：上=两种收益率叠加，下=差异放大（按 |R| 大小着色）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("300328 宜安科技 · 简单收益率 vs 对数收益率（近一年）", fontsize=15, fontweight="bold")

ax1.plot(ret["时间"], ret["简单收益率"] * 100, color="#854F0B", linewidth=0.9, label="简单收益率 R")
ax1.plot(ret["时间"], ret["对数收益率"] * 100, color="#185FA5", linewidth=0.9, alpha=0.75,
         linestyle="--", label="对数收益率 r = ln(1+R)")
ax1.axhline(0, color="#888780", linewidth=0.8)
ax1.set_ylabel("收益率（%）")
ax1.grid(alpha=0.25)
ax1.legend(loc="upper right", fontsize=9)
ax1.set_title("两条线几乎重合——日频下两者近似相等", fontsize=11, loc="left")

ax2.bar(ret["时间"], diff * 100, color="#993556", width=1.0)
ax2.set_ylabel("|R − r|（百分点）")
ax2.grid(alpha=0.25)
ax2.set_title("差异放大图：只在涨跌幅大的日子才看得出区别，跌幅越大差异越大", fontsize=11, loc="left")

fig.text(
    0.99, 0.005,
    f"样本 {n} 天  对数收益 日均 {mu*100:+.3f}%  日波动 {sigma*100:.2f}%  偏度 {skew:.2f}  超额峰度 {kurt-3:.2f}  JB={jb:.0f}"
    f"  |R−r| 最大 {diff.max()*100:.2f}pp",
    ha="right", fontsize=9, color="#5F5E5A",
)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(OUT_DIR / "300328_对数收益率对比_近一年.png", dpi=150)
print(f"样本 {n} 天")
print(f"对数收益: 日均 {mu*100:+.4f}%  日波动 {sigma*100:.3f}%  年化波动 {sigma*np.sqrt(252)*100:.1f}%")
print(f"偏度 {skew:.3f}  峰度 {kurt:.3f}(超额 {kurt-3:.3f})  Jarque-Bera {jb:.1f} (chi2_2 的 99% 分位=9.21)")
print(f"|简单-对数| 平均 {diff.mean()*100:.4f}pp, 最大 {diff.max()*100:.2f}pp")
print(big[["时间", "简单收益率", "对数收益率"]].assign(
    简单收益率=lambda x: (x["简单收益率"]*100).round(2),
    对数收益率=lambda x: (x["对数收益率"]*100).round(2)).to_string(index=False))
