# -*- coding: utf-8 -*-
"""第2章实践（稳健性补充）：300328 月收益率 ACF 的异常值与子样本检验

目的：判断月 ACF 在 lag=6 / lag=12 的显著正相关是真结构还是少数极端月造成的伪相关。
输出：控制台表格（同时核对极端月份）
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

CSV = r"C:\Ai4System\FinancialTimeSeriesAnalysis\第01章 金融时间序列及其特征\300328_月收益率_近10年.csv"
df = pd.read_csv(CSV, encoding="utf-8-sig")
df["时间"] = pd.to_datetime(df["时间"])
r = df["简单收益率"].astype(float)

print(f"样本 {len(r)} 个月  {df['时间'].iloc[0].date()} ~ {df['时间'].iloc[-1].date()}")
print("最大涨幅:", df.loc[r.idxmax(), "时间"].date(), round(r.max(), 4),
      " 最大跌幅:", df.loc[r.idxmin(), "时间"].date(), round(r.min(), 4))
print(f"|r|>15% 的月份数: {(r.abs() > 0.15).sum()} / {len(r)}\n")

print("[原序列]        acf1-12:", np.round(acf(r, nlags=12, fft=True)[1:], 3))
w = r.clip(r.quantile(.05), r.quantile(.95))
print("[Winsor 5/95%]  acf1-12:", np.round(acf(w, nlags=12, fft=True)[1:], 3), " (压缩极端月后是否保留)")

for lab, sub in [("2016-2023", r[df["时间"] < "2024-01-01"]),
                 ("2024-2026", r[df["时间"] >= "2024-01-01"])]:
    a = acf(sub, nlags=12, fft=True)[1:]
    lb = acorr_ljungbox(sub, lags=[6, 12], return_df=True)
    print(f"[{lab}] n={len(sub):3d} acf1-12:", np.round(a, 3),
          f" LB(6) p={lb.loc[6, 'lb_pvalue']:.4f} LB(12) p={lb.loc[12, 'lb_pvalue']:.4f}")
