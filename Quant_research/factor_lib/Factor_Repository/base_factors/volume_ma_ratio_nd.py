# -*- coding: utf-8 -*-
"""短期均量相对长期均量因子。"""
import time
import numpy as np
import pandas as pd
def _prepare_daily_panel(data, target_dates, as_of_date, required_columns):
    if not isinstance(data, pd.DataFrame): raise TypeError("data 必须是 pandas.DataFrame。")
    columns = ["date", "instrument", *required_columns]; missing = set(columns) - set(data.columns)
    if missing: raise ValueError(f"volume_ma_ratio_nd 缺少输入字段：{sorted(missing)}")
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

OUTPUT_COLUMNS = ["date", "instrument", "volume_ma_ratio_nd"]


def _resolve_volume_ma_ratio_nd_data_window(params):
    short_window, long_window = params.get("short_window", 5), params.get("long_window", 20)
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and v >= 1 for v in (short_window, long_window)): raise ValueError("short_window 和 long_window 必须是正整数。")
    if short_window > long_window: raise ValueError("short_window 不能大于 long_window。")
    return {"lookback_trading_days": int(long_window)-1, "requires_target_date_data": True, "minimum_history_observations": int(long_window)-1, "preheating_required": long_window > 1, "insufficient_window_behavior": "短期或长期均量窗口不足时输出 NaN。"}


def calc_volume_ma_ratio_nd(data, target_dates=None, as_of_date=None, short_window=5, long_window=20, show_progress=False, progress_every=20):
    """计算 ``MA(volume, short_window) / MA(volume, long_window) - 1``。"""
    del progress_every
    _resolve_volume_ma_ratio_nd_data_window({"short_window": short_window, "long_window": long_window}); started_at = time.perf_counter()
    df, targets = _prepare_daily_panel(data, target_dates, as_of_date, ("volume",))
    if df.empty or targets.empty: return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress: print(f"\r[volume_ma_ratio_nd] [1/2] 计算 {short_window}/{long_window} 日均量比...", end="", flush=True)
    try:
        grouped = df.groupby("instrument", sort=False)["volume"]
        short_ma = grouped.transform(lambda item: item.rolling(int(short_window), min_periods=int(short_window)).mean())
        long_ma = grouped.transform(lambda item: item.rolling(int(long_window), min_periods=int(long_window)).mean())
        valid = np.isfinite(short_ma) & np.isfinite(long_ma) & (long_ma != 0)
        values = pd.Series(np.nan, index=df.index, dtype=float); values.loc[valid] = short_ma.loc[valid] / long_ma.loc[valid] - 1.0
        result = df.loc[df.date.isin(targets), ["date", "instrument"]].copy(); result["volume_ma_ratio_nd"] = values.loc[result.index].to_numpy()
        if show_progress: print(f"\r[volume_ma_ratio_nd] [2/2] 完成 | 耗时 {time.perf_counter()-started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress: print()


FACTOR = {"name": "volume_ma_ratio_nd", "func": calc_volume_ma_ratio_nd,
          "factor_type": "base", "candidate_instances": {"5d_20d": {"short_window": 5, "long_window": 20}},
          "input_schema": {"required": {"date": {}, "instrument": {}, "volume": {}}, "conditional": {}},
          "parameters": {"target_dates": {"default": None}, "as_of_date": {"default": None}, "short_window": {"default": 5}, "long_window": {"default": 20}, "show_progress": {"default": False}, "progress_every": {"default": 20}},
          "data_window": {"resolver": _resolve_volume_ma_ratio_nd_data_window, "default": _resolve_volume_ma_ratio_nd_data_window({})},
          "output_schema": {"date": {}, "instrument": {}, "volume_ma_ratio_nd": {"dtype": "float64"}}}
FACTOR_INFO = """# volume_ma_ratio_nd\n\n均量趋势因子：`MA(volume, short)_t / MA(volume, long)_t - 1`。`short_window` 和 `long_window` 分别控制短、长期均量窗口；正值表示近期平均成交量高于长期平均水平。"""
