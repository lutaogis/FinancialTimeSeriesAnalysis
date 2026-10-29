# -*- coding: utf-8 -*-
"""第2章实践：计算 300328 宜安科技日简单收益率与对数收益率的样本自相关函数 (ACF)

数据源：第01章已抓取的近一年日收益率 CSV（241 个观测）
输出：
  - 300328/300328_ACF_收益率.csv        两种收益率的样本 ACF 数值表
  - 300328/300328_ACF_对比.png          correlogram 对比图（含 ±2/√T 置信带）
  - 300328/300328_LjungBox_检验.csv     Ljung-Box 检验结果（m=5/10/20）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------- 配置 ----------
ROOT = r"C:\Ai4System\FinancialTimeSeriesAnalysis"
CH1 = os.path.join(ROOT, "第01章 金融时间序列及其特征")
OUT = os.path.join(ROOT, "300328")
CSV = os.path.join(CH1, "300328_对数收益率_近一年.csv")  # 该文件同时含简单收益率与对数收益率两列
NLAGS = 20
LB_M = [5, 10, 20]

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 读取 ----------
df = pd.read_csv(CSV, encoding="utf-8-sig")
df["时间"] = pd.to_datetime(df["时间"])
series = {
    "简单收益率": df["简单收益率"].astype(float),
    "对数收益率": df["对数收益率"].astype(float),
}
T = len(df)
print(f"观测数 T = {T}  ({df['时间'].iloc[0].date()} ~ {df['时间'].iloc[-1].date()})")
print(f"置信带半径 2/sqrt(T) = {2/np.sqrt(T):.4f}\n")

# ---------- ACF ----------
acf_rows = []
for name, r in series.items():
    a = acf(r, nlags=NLAGS, fft=True)
    for l in range(0, NLAGS + 1):
        acf_rows.append({"序列": name, "滞后阶 l": l, "ACF": round(a[l], 6),
                         "显著(|acf|>2/sqrt(T))": abs(a[l]) > 2 / np.sqrt(T)})
acf_df = pd.DataFrame(acf_rows)
acf_path = os.path.join(OUT, "300328_ACF_收益率.csv")
acf_df.to_csv(acf_path, index=False, encoding="utf-8-sig")

print("滞后阶   简单收益率ACF   对数收益率ACF")
for l in range(0, NLAGS + 1):
    s1 = acf_df[(acf_df["序列"] == "简单收益率") & (acf_df["滞后阶 l"] == l)]["ACF"].iloc[0]
    s2 = acf_df[(acf_df["序列"] == "对数收益率") & (acf_df["滞后阶 l"] == l)]["ACF"].iloc[0]
    print(f"{l:4d}   {s1:+.4f}          {s2:+.4f}")

# ---------- Ljung-Box ----------
lb_rows = []
for name, r in series.items():
    lb = acorr_ljungbox(r, lags=LB_M, return_df=True)
    for m in LB_M:
        lb_rows.append({"序列": name, "m": m,
                        "Q(m)": round(lb.loc[m, "lb_stat"], 4),
                        "p值": round(lb.loc[m, "lb_pvalue"], 6),
                        "5%下是否拒绝": lb.loc[m, "lb_pvalue"] < 0.05})
lb_df = pd.DataFrame(lb_rows)
lb_path = os.path.join(OUT, "300328_LjungBox_检验.csv")
lb_df.to_csv(lb_path, index=False, encoding="utf-8-sig")
print("\nLjung-Box 检验：")
print(lb_df.to_string(index=False))

# ---------- 作图 ----------
band = 2 / np.sqrt(T)
x = np.arange(0, NLAGS + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, (name, _) in zip(axes, series.items()):
    a = acf_df[acf_df["序列"] == name]["ACF"].values
    colors = ["#A32D2D" if abs(v) > band else "#185FA5" for v in a]
    ax.bar(x, a, color=colors, width=0.6)
    ax.axhline(0, color="#888780", lw=0.8)
    ax.axhline(band, color="#378ADD", lw=1, ls="--", label=r"$\pm 2/\sqrt{T}$ 置信带")
    ax.axhline(-band, color="#378ADD", lw=1, ls="--")
    ax.set_title(f"300328 宜安科技 日{name} 样本ACF（T={T}）")
    ax.set_xlabel("滞后阶 l（交易日）")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.legend(fontsize=9)
axes[0].set_ylabel("ACF")
fig.tight_layout()
png_path = os.path.join(OUT, "300328_ACF_对比.png")
fig.savefig(png_path, dpi=150)
print(f"\n已输出：\n  {acf_path}\n  {lb_path}\n  {png_path}")
