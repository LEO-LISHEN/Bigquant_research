# -*- coding: utf-8 -*-
"""当日最高价相对前收收益因子。"""

import time
import numpy as np
import pandas as pd

def _prepare_daily_panel(data, target_dates, as_of_date, required_columns):
    if not isinstance(data, pd.DataFrame): raise TypeError("data 必须是 pandas.DataFrame。")
    columns = ["date", "instrument", *required_columns]; missing = set(columns) - set(data.columns)
    if missing: raise ValueError(f"high_relative_return 缺少输入字段：{sorted(missing)}")
    df = data.loc[:, columns].copy(); df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any() or df["instrument"].isna().any(): raise ValueError("date 或 instrument 存在缺失或无效值。")
    if df.duplicated(["date", "instrument"], keep=False).any(): raise ValueError("输入存在重复的 date + instrument 记录。")
    for column in required_columns: df[column] = pd.to_numeric(df[column], errors="coerce")
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        if pd.isna(cutoff): raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= cutoff.normalize()].copy()
    available = pd.DatetimeIndex(df["date"].unique()).sort_values()
    if target_dates is None: targets = available
    else:
        values = [target_dates] if isinstance(target_dates, (str, pd.Timestamp)) else list(target_dates)
        targets = pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize().unique().sort_values()
        if not targets.difference(available).empty: raise ValueError("缺少目标日期的原始数据。")
    return df.sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True), targets

OUTPUT_COLUMNS = ["date", "instrument", "high_relative_return"]


def _resolve_high_relative_return_data_window(_):
    return {"lookback_trading_days": 1, "requires_target_date_data": True,
            "minimum_history_observations": 1, "preheating_required": True,
            "insufficient_window_behavior": "缺少上一交易日收盘价时输出 NaN。"}


def calc_high_relative_return(data, target_dates=None, as_of_date=None, show_progress=False, progress_every=20):
    """计算 ``high_t / close_{t-1} - 1``。"""
    del progress_every
    started_at = time.perf_counter()
    df, targets = _prepare_daily_panel(data, target_dates, as_of_date, ("high", "close"))
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress:
        print("\r[high_relative_return] [1/2] 计算前收相对最高价...", end="", flush=True)
    try:
        previous_close = df.groupby("instrument", sort=False)["close"].shift(1)
        valid = np.isfinite(df["high"]) & np.isfinite(previous_close) & (previous_close != 0)
        values = pd.Series(np.nan, index=df.index, dtype=float)
        values.loc[valid] = df.loc[valid, "high"] / previous_close.loc[valid] - 1.0
        result = df.loc[df["date"].isin(targets), ["date", "instrument"]].copy()
        result["high_relative_return"] = values.loc[result.index].to_numpy()
        if show_progress:
            print(f"\r[high_relative_return] [2/2] 完成 | 耗时 {time.perf_counter() - started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "high_relative_return", "func": calc_high_relative_return,
    "input_schema": {"required": {"date": {}, "instrument": {}, "high": {}, "close": {}}, "conditional": {}},
    "parameters": {"target_dates": {"default": None}, "as_of_date": {"default": None}, "show_progress": {"default": False}, "progress_every": {"default": 20}},
    "data_window": {"resolver": _resolve_high_relative_return_data_window, "default": _resolve_high_relative_return_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "high_relative_return": {"dtype": "float64"}},
}

FACTOR_INFO = """# high_relative_return\n\n日内最大上行幅度，计算当日最高价相对上一交易日收盘价的变化：`high_t / close_{t-1} - 1`。数值越大，表示该日盘中上冲越明显。"""
