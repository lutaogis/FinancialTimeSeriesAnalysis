# -*- coding: utf-8 -*-
"""第01章实践任务：三只大盘指数（上证/深成/创业板）日收益率分析，并与 300328 对比。
前缀规则：沪市指数 USHI + 老代码（1A0001）；深市指数 USZI + 6 位代码；个股 6 开头 USHA / 其余 USZA。
"""
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
START, END = datetime(2025, 10, 24), datetime(2026, 10, 24)

ASSETS = [
    ("上证指数", "USHI1A0001", "#185FA5"),
    ("深证成指", "USZI399001", "#534AB7"),
    ("创业板指", "USZI399006", "#0F6E56"),
    ("300328宜安科技", "USZA300328", "#A32D2D"),  # 对照组
]


def fetch(ths, code):
    resp = ths.klines(code, interval="day", start_time=START, end_time=END)
    if not resp.success:
        raise SystemExit(f"{code} 取数失败：{resp.error}")
    df = resp.df.copy()
    df["时间"] = pd.to_datetime(df["时间"])
    return df.sort_values("时间").reset_index(drop=True)


# ---- 逐只抓取、计算、绘图 ----
results = {}
with thsdk.THS() as ths:
    for name, code, color in ASSETS:
        df = fetch(ths, code)
        df["简单收益率"] = df["收盘价"].pct_change()
        df["对数收益率"] = np.log(df["收盘价"]).diff()
        ret = df.dropna(subset=["简单收益率"]).copy()
        results[name] = (df, ret, color)

        # 每只指数单独一张图：上=价格，下=日简单收益率（涨红跌绿）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                       gridspec_kw={"height_ratios": [1.2, 1]})
        fig.suptitle(f"{name} · 近一年日行情与简单收益率", fontsize=15, fontweight="bold")
        ax1.plot(df["时间"], df["收盘价"], color="#185FA5", linewidth=1.3)
        ax1.set_ylabel("收盘点位" if "指数" in name else "收盘价（元）")
        ax1.grid(alpha=0.25)
        ax1.set_title("收盘价", fontsize=11, loc="left")
        bar_colors = ["#A32D2D" if v >= 0 else "#0F6E56" for v in ret["简单收益率"]]
        ax2.bar(ret["时间"], ret["简单收益率"] * 100, color=bar_colors, width=1.0)
        ax2.axhline(0, color="#888780", linewidth=0.8)
        ax2.set_ylabel("日简单收益率（%）")
        ax2.grid(alpha=0.25)
        mu, sd = ret["简单收益率"].mean(), ret["简单收益率"].std()
        fig.text(0.99, 0.005,
                 f"样本 {len(ret)} 个交易日  日均 {mu*100:+.3f}%  日波动 {sd*100:.2f}%  年化波动 {sd*np.sqrt(252)*100:.1f}%",
                 ha="right", fontsize=9, color="#5F5E5A")
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig.savefig(OUT_DIR / f"指数分析_{name}_日收益率_近一年.png", dpi=150)
        plt.close(fig)

# ---- 汇总统计表（对数收益口径）----
rows = []
for name, (df, ret, color) in results.items():
    x = ret["对数收益率"].values
    mu, sd = x.mean(), x.std(ddof=1)
    jb, jb_p = stats.jarque_bera(x)
    z = np.abs((x - mu) / sd)
    rows.append({
        "资产": name, "交易日": len(x),
        "日均bp": round(mu * 1e4, 1), "日波动%": round(sd * 100, 2),
        "年化波动%": round(sd * np.sqrt(252) * 100, 1),
        "偏度": round(stats.skew(x), 2), "超额峰度": round(stats.kurtosis(x), 2),
        "JB": round(jb, 1), "JB_p": f"{jb_p:.1e}",
        ">±3σ占比%": round((z > 3).mean() * 100, 1),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "指数与300328_收益率统计对比.csv", index=False, encoding="utf-8-sig")
print(summary.to_string(index=False))

# ---- 合图：四资产对数收益率密度 vs 正态密度 ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
fig.suptitle("对数收益率经验密度 vs 正态密度（近一年）：三指数 vs 300328", fontsize=15, fontweight="bold")
for ax, (name, (df, ret, color)) in zip(axes.flat, results.items()):
    x = ret["对数收益率"].values
    mu, sd = x.mean(), x.std(ddof=1)
    sk, ku = stats.skew(x), stats.kurtosis(x)
    jb, _ = stats.jarque_bera(x)
    ax.hist(x, bins=40, density=True, color=color, alpha=0.35, edgecolor=color, linewidth=0.6)
    kde = stats.gaussian_kde(x)
    xs = np.linspace(x.min() * 1.2, x.max() * 1.2, 400)
    ax.plot(xs, kde(xs), color=color, linewidth=1.6, label="核平滑密度")
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color="#444441", linewidth=1.6, linestyle="--",
            label="正态密度 N(μ,σ²)")
    ax.axvline(0, color="#888780", linewidth=0.7)
    ax.set_title(f"{name}  偏度 {sk:+.2f}  超额峰度 {ku:+.2f}  JB={jb:.0f}", fontsize=11)
    ax.set_xlabel("日对数收益率")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")
for ax in axes[:, 0]:
    ax.set_ylabel("密度")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT_DIR / "指数与300328_密度对比_近一年.png", dpi=150)
print("图已保存")
