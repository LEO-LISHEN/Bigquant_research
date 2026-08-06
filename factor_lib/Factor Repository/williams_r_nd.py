# -*- coding: utf-8 -*-
"""N 日 Williams %R 因子。"""
import time
import numpy as np
import pandas as pd
def _prepare_daily_panel(data, target_dates, as_of_date, required_columns):
    if not isinstance(data, pd.DataFrame): raise TypeError("data 必须是 pandas.DataFrame。")
    columns = ["date", "instrument", *required_columns]; missing = set(columns) - set(data.columns)
    if missing: raise ValueError(f"williams_r_nd 缺少输入字段：{sorted(missing)}")
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

OUTPUT_COLUMNS = ["date", "instrument", "williams_r_nd"]


def _resolve_williams_r_nd_data_window(params):
    window = params.get("window", 14)
    if not isinstance(window, (int, np.integer)) or isinstance(window, (bool, np.bool_)) or window < 1: raise ValueError("window 必须是正整数。")
    return {"lookback_trading_days": int(window)-1, "requires_target_date_data": True, "minimum_history_observations": int(window)-1, "preheating_required": window > 1, "insufficient_window_behavior": "高低价窗口不足或高低价相等时输出 NaN。"}


def calc_williams_r_nd(data, target_dates=None, as_of_date=None, window=14, show_progress=False, progress_every=20):
    """计算 ``-100 * (HHV(high,N) - close) / (HHV - LLV)``。"""
    del progress_every
    _resolve_williams_r_nd_data_window({"window": window}); started_at = time.perf_counter()
    df, targets = _prepare_daily_panel(data, target_dates, as_of_date, ("high", "low", "close"))
    if df.empty or targets.empty: return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress: print(f"\r[williams_r_nd] [1/2] 计算 {window} 日高低价区间...", end="", flush=True)
    try:
        grouped = df.groupby("instrument", sort=False)
        highest = grouped["high"].transform(lambda item: item.rolling(int(window), min_periods=int(window)).max())
        lowest = grouped["low"].transform(lambda item: item.rolling(int(window), min_periods=int(window)).min())
        spread = highest - lowest; valid = np.isfinite(highest) & np.isfinite(lowest) & np.isfinite(df.close) & (spread != 0)
        values = pd.Series(np.nan, index=df.index, dtype=float); values.loc[valid] = -100.0 * (highest.loc[valid] - df.loc[valid, "close"]) / spread.loc[valid]
        result = df.loc[df.date.isin(targets), ["date", "instrument"]].copy(); result["williams_r_nd"] = values.loc[result.index].to_numpy()
        if show_progress: print(f"\r[williams_r_nd] [2/2] 完成 | 耗时 {time.perf_counter()-started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress: print()


FACTOR = {"name": "williams_r_nd", "func": calc_williams_r_nd,
          "input_schema": {"required": {"date": {}, "instrument": {}, "high": {}, "low": {}, "close": {}}, "conditional": {}},
          "parameters": {"target_dates": {"default": None}, "as_of_date": {"default": None}, "window": {"default": 14}, "show_progress": {"default": False}, "progress_every": {"default": 20}},
          "data_window": {"resolver": _resolve_williams_r_nd_data_window, "default": _resolve_williams_r_nd_data_window({})},
          "output_schema": {"date": {}, "instrument": {}, "williams_r_nd": {"dtype": "float64"}}}
FACTOR_INFO = """# williams_r_nd\n\nWilliams %R 指标：`-100 * (N 日最高价 - close) / (N 日最高价 - N 日最低价)`。参数 `window` 默认 14；值越接近 0 表示收盘价越靠近窗口最高价，越接近 -100 则越靠近窗口最低价。"""
