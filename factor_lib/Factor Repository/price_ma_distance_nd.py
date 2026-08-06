# -*- coding: utf-8 -*-
"""收盘价相对 N 日均线距离因子。"""
import time
import numpy as np
import pandas as pd
def _prepare_daily_panel(data, target_dates, as_of_date, required_columns):
    if not isinstance(data, pd.DataFrame): raise TypeError("data 必须是 pandas.DataFrame。")
    columns = ["date", "instrument", *required_columns]; missing = set(columns) - set(data.columns)
    if missing: raise ValueError(f"price_ma_distance_nd 缺少输入字段：{sorted(missing)}")
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

OUTPUT_COLUMNS = ["date", "instrument", "price_ma_distance_nd"]


def _resolve_price_ma_distance_nd_data_window(params):
    window = params.get("window", 5)
    if not isinstance(window, (int, np.integer)) or isinstance(window, (bool, np.bool_)) or window < 1: raise ValueError("window 必须是正整数。")
    return {"lookback_trading_days": int(window)-1, "requires_target_date_data": True, "minimum_history_observations": int(window)-1, "preheating_required": window > 1, "insufficient_window_behavior": "均线窗口不足或均线为零时输出 NaN。"}


def calc_price_ma_distance_nd(data, target_dates=None, as_of_date=None, window=5, show_progress=False, progress_every=20):
    """计算 ``(close_t - MA(close, window)_t) / MA(close, window)_t``。"""
    del progress_every
    _resolve_price_ma_distance_nd_data_window({"window": window}); started_at = time.perf_counter()
    df, targets = _prepare_daily_panel(data, target_dates, as_of_date, ("close",))
    if df.empty or targets.empty: return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress: print(f"\r[price_ma_distance_nd] [1/2] 计算 {window} 日价格均线...", end="", flush=True)
    try:
        average = df.groupby("instrument", sort=False)["close"].transform(lambda item: item.rolling(int(window), min_periods=int(window)).mean())
        valid = np.isfinite(df.close) & np.isfinite(average) & (average != 0)
        values = pd.Series(np.nan, index=df.index, dtype=float); values.loc[valid] = df.loc[valid, "close"] / average.loc[valid] - 1.0
        result = df.loc[df.date.isin(targets), ["date", "instrument"]].copy(); result["price_ma_distance_nd"] = values.loc[result.index].to_numpy()
        if show_progress: print(f"\r[price_ma_distance_nd] [2/2] 完成 | 耗时 {time.perf_counter()-started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress: print()


FACTOR = {"name": "price_ma_distance_nd", "func": calc_price_ma_distance_nd,
          "input_schema": {"required": {"date": {}, "instrument": {}, "close": {}}, "conditional": {}},
          "parameters": {"target_dates": {"default": None}, "as_of_date": {"default": None}, "window": {"default": 5}, "show_progress": {"default": False}, "progress_every": {"default": 20}},
          "data_window": {"resolver": _resolve_price_ma_distance_nd_data_window, "default": _resolve_price_ma_distance_nd_data_window({})},
          "output_schema": {"date": {}, "instrument": {}, "price_ma_distance_nd": {"dtype": "float64"}}}
FACTOR_INFO = """# price_ma_distance_nd\n\n价格均线偏离因子：`close_t / MA(close, N)_t - 1`。参数 `window` 为均线窗口，默认 5；正值表示收盘价位于均线之上。"""
