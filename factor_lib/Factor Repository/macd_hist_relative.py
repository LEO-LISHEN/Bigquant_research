# -*- coding: utf-8 -*-
"""相对价格归一化的 MACD 柱状线因子。"""
import time
import numpy as np
import pandas as pd
def _prepare_daily_panel(data, target_dates, as_of_date, required_columns):
    if not isinstance(data, pd.DataFrame): raise TypeError("data 必须是 pandas.DataFrame。")
    columns = ["date", "instrument", *required_columns]; missing = set(columns) - set(data.columns)
    if missing: raise ValueError(f"macd_hist_relative 缺少输入字段：{sorted(missing)}")
    df = data.loc[:, columns].copy(); df.date = pd.to_datetime(df.date, errors="coerce").dt.normalize()
    if df.date.isna().any() or df.instrument.isna().any(): raise ValueError("date 或 instrument 存在缺失或无效值。")
    if df.duplicated(["date", "instrument"], keep=False).any(): raise ValueError("输入存在重复的 date + instrument 记录。")
    for column in required_columns: df[column] = pd.to_numeric(df[column], errors="coerce")
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        if pd.isna(cutoff): raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df.date <= cutoff.normalize()].copy()
    available = pd.DatetimeIndex(df.date.unique()).sort_values()
    if target_dates is None: targets = available
    else:
        values = [target_dates] if isinstance(target_dates, (str, pd.Timestamp)) else list(target_dates)
        targets = pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize().unique().sort_values()
        if not targets.difference(available).empty: raise ValueError("缺少目标日期的原始数据。")
    return df.sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True), targets

OUTPUT_COLUMNS = ["date", "instrument", "macd_hist_relative"]


def _resolve_macd_hist_relative_data_window(params):
    fast, slow, signal, multiplier = (params.get("fast_window", 12), params.get("slow_window", 26), params.get("signal_window", 9), params.get("warmup_multiplier", 5))
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and v >= 1 for v in (fast, slow, signal, multiplier)): raise ValueError("MACD 窗口和 warmup_multiplier 必须是正整数。")
    if fast >= slow: raise ValueError("fast_window 必须小于 slow_window。")
    lookback = int(multiplier) * int(slow) + int(signal) - 2
    return {"lookback_trading_days": lookback, "requires_target_date_data": True, "minimum_history_observations": lookback, "preheating_required": True, "insufficient_window_behavior": "EMA 预热不足或收盘价为零时输出 NaN。"}


def calc_macd_hist_relative(data, target_dates=None, as_of_date=None, fast_window=12, slow_window=26, signal_window=9, warmup_multiplier=5, show_progress=False, progress_every=20):
    """计算 ``2 * (DIF - DEA) / abs(close)``。"""
    del progress_every
    _resolve_macd_hist_relative_data_window({"fast_window": fast_window, "slow_window": slow_window, "signal_window": signal_window, "warmup_multiplier": warmup_multiplier}); started_at = time.perf_counter()
    df, targets = _prepare_daily_panel(data, target_dates, as_of_date, ("close",))
    if df.empty or targets.empty: return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress: print("\r[macd_hist_relative] [1/3] 计算快慢 EMA...", end="", flush=True)
    try:
        grouped = df.groupby("instrument", sort=False)["close"]
        fast = grouped.transform(lambda item: item.ewm(span=int(fast_window), adjust=False, min_periods=int(fast_window)).mean())
        slow = grouped.transform(lambda item: item.ewm(span=int(slow_window), adjust=False, min_periods=int(slow_window)).mean())
        dif = fast - slow
        if show_progress: print("\r[macd_hist_relative] [2/3] 计算 DEA 与柱状线...", end="", flush=True)
        dea = dif.groupby(df["instrument"], sort=False).transform(lambda item: item.ewm(span=int(signal_window), adjust=False, min_periods=int(signal_window)).mean())
        valid = np.isfinite(dif) & np.isfinite(dea) & np.isfinite(df.close) & (df.close != 0)
        values = pd.Series(np.nan, index=df.index, dtype=float); values.loc[valid] = 2.0 * (dif.loc[valid] - dea.loc[valid]) / df.loc[valid, "close"].abs()
        result = df.loc[df.date.isin(targets), ["date", "instrument"]].copy(); result["macd_hist_relative"] = values.loc[result.index].to_numpy()
        if show_progress: print(f"\r[macd_hist_relative] [3/3] 完成 | 耗时 {time.perf_counter()-started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress: print()


FACTOR = {"name": "macd_hist_relative", "func": calc_macd_hist_relative,
          "input_schema": {"required": {"date": {}, "instrument": {}, "close": {}}, "conditional": {}},
          "parameters": {"target_dates": {"default": None}, "as_of_date": {"default": None}, "fast_window": {"default": 12}, "slow_window": {"default": 26}, "signal_window": {"default": 9}, "warmup_multiplier": {"default": 5}, "show_progress": {"default": False}, "progress_every": {"default": 20}},
          "data_window": {"resolver": _resolve_macd_hist_relative_data_window, "default": _resolve_macd_hist_relative_data_window({})},
          "output_schema": {"date": {}, "instrument": {}, "macd_hist_relative": {"dtype": "float64"}}}
FACTOR_INFO = """# macd_hist_relative\n\n先由快、慢 EMA 的差值得到 `DIF`，再计算其 `signal_window` 日 EMA 得到 `DEA`；最终因子为 `2 * (DIF - DEA) / abs(close)`。默认参数为快线 12、慢线 26、信号线 9。"""
