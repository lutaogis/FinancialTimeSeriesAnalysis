# -*- coding: utf-8 -*-
"""第2章实践：复现 Tsay 书中 VW 指数的 AR 建模 + 样本外预测流程（A股三指数版）

流程（对齐教材第 2 章）：
  1. 抓取三指数近10年日K → 聚合月末 → 月对数收益率（对应书中 m-vw 月度数据）
  2. ACF/PACF 定阶图 + Ljung-Box 检验
  3. 候选模型 AR(1)/AR(2)/AR(3)/MA(1)/MA(2)/ARMA(1,1)，MLE 估计，AIC/BIC 选阶
  4. 选中模型残差 Ljung-Box 诊断（均值方程是否建干净）
  5. 样本外滚动预测（expanding window，一步向前）：RMSE/MAE vs 样本均值与随机游走基准

输出约定：
  各指数文件夹：{code}_月收益率_近10年.csv / {code}_AR建模_AIC_BIC.csv /
               {code}_AR建模_定阶图.png / {code}_AR建模_样本外预测.png / {code}_样本外预测明细.csv
  第02章文件夹：指数_AR建模_横向对比.csv
"""
import logging
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import thsdk
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"C:\Ai4System\FinancialTimeSeriesAnalysis")
CH2 = ROOT / "第02章 线性时间序列分析及其应用"
START, END = datetime(2016, 10, 24), datetime(2026, 10, 24)
TRAIN_N = 96          # 样本内训练窗（月），其余滚动预测
INDICES = [
    ("000001", "上证指数", "USHI1A0001", "#185FA5"),
    ("399001", "深证成指", "USZI399001", "#534AB7"),
    ("399006", "创业板指", "USZI399006", "#0F6E56"),
]
CANDIDATES = [(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (1, 1)]  # (p,q)：AR/MA/ARMA 候选


def fetch_monthly(ths, code, retries=6, delay=10):
    """取日K聚合成月；游客会话被限流时偶发『未登录』，自动重试换会话。
    结果按 code 缓存到 CSV，重跑不重复抓数。"""
    import time
    cache = ROOT / "第02章 线性时间序列分析及其应用" / f"_cache_{code}_月K.csv"
    if cache.exists():
        m = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"  [{code}] 命中缓存 {len(m)} 个月")
        return m
    last_err = None
    for attempt in range(retries):
        try:
            resp = ths.klines(code, interval="day", start_time=START, end_time=END) if ths else None
            if resp is not None and resp.success:
                daily = resp.df.copy()
                daily["时间"] = pd.to_datetime(daily["时间"])
                monthly = (daily.sort_values("时间").set_index("时间")["收盘价"]
                           .resample("M").last().dropna())
                m = pd.DataFrame({"收盘价": monthly})
                m["简单收益率"] = monthly.pct_change()
                m["对数收益率"] = np.log(monthly).diff()
                m = m.dropna()
                m.to_csv(cache, encoding="utf-8-sig")
                return m
            last_err = getattr(resp, "error", "unknown") if resp is not None else "no session"
            print(f"  [DEBUG] resp 类型: {type(resp)}, error 类型: {type(last_err)}, 内容: {last_err!r}")
            print(f"  [DEBUG] ths 对象: {type(ths)}, klines 存在: {hasattr(ths, 'klines')}")
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            import traceback
            traceback.print_exc()
        print(f"  [{code}] 第{attempt+1}次失败：{last_err}，{delay}s 后换会话重试…")
        time.sleep(delay)
        try:
            ths.close()
        except Exception:  # noqa: BLE001
            pass
        ths = thsdk.THS()
        ths.connect()
    raise SystemExit(f"{code} 重试{retries}次仍失败：{last_err}")


def fit_table(r):
    """对月收益序列拟合候选模型，返回 AIC/BIC 表。"""
    rows = []
    for p, q in CANDIDATES:
        res = ARIMA(r, order=(p, 0, q), trend="c").fit()
        rows.append({"模型": f"ARMA({p},{q})", "参数个数": p + q + 1,
                     "对数似然": round(res.llf, 3),
                     "AIC": round(res.aic, 3), "BIC": round(res.bic, 3),
                     "系数": "  ".join(f"{n}={v:+.4f}" for n, v in
                                       zip(res.param_names, res.params))})
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def oos_backtest(r):
    """expanding window 一步向前滚动预测，对比三基准。"""
    T = len(r)
    actual, f_model, f_mean, f_rw = [], [], [], []
    for t in range(TRAIN_N, T):
        train = r.iloc[:t]
        res = ARIMA(train, order=(BEST_ORDER[0], 0, BEST_ORDER[1]), trend="c").fit()
        actual.append(r.iloc[t])
        f_model.append(float(res.forecast(1).iloc[0]))
        f_mean.append(train.mean())
        f_rw.append(train.iloc[-1])
    out = pd.DataFrame({"实际": actual, "模型预测": f_model,
                        "均值基准": f_mean, "随机游走": f_rw})
    metrics = {}
    for c in ["模型预测", "均值基准", "随机游走"]:
        e = out["实际"] - out[c]
        metrics[c] = {"RMSE": np.sqrt((e**2).mean()), "MAE": e.abs().mean()}
    return out, pd.DataFrame(metrics).T, BEST_ORDER[0]


comparison_rows = []
ths = thsdk.THS()
ths.connect()
try:
    for code, name, tscode, color in INDICES:
        outdir = ROOT / code
        m = fetch_monthly(ths, tscode)
        m.to_csv(outdir / f"{code}_月收益率_近10年.csv", encoding="utf-8-sig")
        r = m["对数收益率"]
        T = len(r)
        band = 2 / np.sqrt(T)

        # ---- 1. ACF/PACF 定阶图 ----
        a = acf(r, nlags=12, fft=True)[1:]
        p_ = pacf(r, nlags=12, method="ywm")[1:]
        lb = acorr_ljungbox(r, lags=[3, 6, 12], return_df=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        x = np.arange(1, 13)
        for ax, vals, ttl in [(ax1, a, "ACF"), (ax2, p_, "PACF")]:
            ax.bar(x, vals, color=["#A32D2D" if abs(v) > band else "#185FA5" for v in vals], width=0.6)
            ax.axhline(0, color="#888780", lw=0.8)
            ax.axhline(band, color="#378ADD", ls="--", lw=1)
            ax.axhline(-band, color="#378ADD", ls="--", lw=1, label=r"$\pm 2/\sqrt{T}$")
            ax.set_title(f"{name} 月对数收益率 {ttl}")
            ax.set_xlabel("滞后阶（月）")
            ax.set_xticks(x)
            ax.legend(fontsize=8)
        ax1.set_ylabel("系数")
        fig.tight_layout()
        fig.savefig(outdir / f"{code}_AR建模_定阶图.png", dpi=150)
        plt.close(fig)

        # ---- 2. AIC/BIC 选阶 ----
        tab = fit_table(r.values)
        tab.to_csv(outdir / f"{code}_AR建模_AIC_BIC.csv", index=False, encoding="utf-8-sig")
        best_aic = tab.iloc[0]
        best_bic = tab.sort_values("BIC").iloc[0]
        p_best = int(best_aic["模型"].split("(")[1].split(",")[0])
        q_best = int(best_aic["模型"].split(",")[1].rstrip(")"))
        BEST_ORDER = (p_best, q_best)

        # ---- 3. 残差诊断（AIC 最优模型）----
        res = ARIMA(r.values, order=(p_best, 0, q_best), trend="c").fit()
        resid = res.resid[1:]  # 去掉初始条件项
        lb_res = acorr_ljungbox(resid, lags=[3, 6, 12], return_df=True)

        # ---- 4. 样本外滚动预测 ----
        oos, metrics, _ = oos_backtest(r)
        oos.index = r.index[TRAIN_N:TRAIN_N + len(oos)]  # 绑定真实月份，避免画图当行号
        oos.to_csv(outdir / f"{code}_样本外预测明细.csv", encoding="utf-8-sig")
        metrics.index.name = "方法"
        metrics.round(6).to_csv(outdir / f"{code}_样本外预测_RMSE_MAE.csv", encoding="utf-8-sig")

        # 预测图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                       gridspec_kw={"height_ratios": [1, 1.2]})
        fig.suptitle(f"{name} · 月对数收益率 ARMA 滚动样本外预测（训练 {TRAIN_N} 月，预测 {T-TRAIN_N} 月）",
                     fontsize=14, fontweight="bold")
        ax1.plot(m.index, r, color=color, lw=1.2)
        ax1.axvline(m.index[TRAIN_N], color="#888780", ls=":", lw=1.2)
        ax1.axhline(r.mean(), color="#378ADD", ls="--", lw=1, label=f"样本均值 {r.mean()*100:+.2f}%")
        ax1.set_ylabel("月对数收益率")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.25)
        ax2.plot(oos.index, oos["实际"] * 100, color="#444441", lw=1.4, label="实际")
        ax2.plot(oos.index, oos["模型预测"] * 100, color="#A32D2D", lw=1.4,
                 label=f"ARMA{BEST_ORDER} 预测")
        ax2.plot(oos.index, oos["随机游走"] * 100, color="#854F0B", lw=1, ls="--",
                 label="随机游走（上月值）")
        ax2.set_ylabel("月对数收益率（%）")
        ax2.set_xlabel("月份")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.25)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(outdir / f"{code}_AR建模_样本外预测.png", dpi=150)
        plt.close(fig)

        comparison_rows.append({
            "指数": name, "月数": T,
            "LB(6) p(原始)": round(lb.loc[6, "lb_pvalue"], 4),
            "AIC最优": best_aic["模型"], "AIC": best_aic["AIC"],
            "BIC最优": best_bic["模型"], "BIC": best_bic["BIC"],
            "残差LB(6) p": round(lb_res.loc[6, "lb_pvalue"], 4),
            "残差LB(12) p": round(lb_res.loc[12, "lb_pvalue"], 4),
            "模型RMSE": round(metrics.loc["模型预测", "RMSE"], 5),
            "均值RMSE": round(metrics.loc["均值基准", "RMSE"], 5),
            "随机游走RMSE": round(metrics.loc["随机游走", "RMSE"], 5),
            "模型/均值RMSE比": round(metrics.loc["模型预测", "RMSE"] / metrics.loc["均值基准", "RMSE"], 3),
        })

finally:
    try:
        ths.close()
    except Exception:  # noqa: BLE001
        pass

cmp_df = pd.DataFrame(comparison_rows)
cmp_df.to_csv(CH2 / "指数_AR建模_横向对比.csv", index=False, encoding="utf-8-sig")
print(cmp_df.to_string(index=False))
print("\n完成。产物已写入各指数文件夹与第02章文件夹。")
