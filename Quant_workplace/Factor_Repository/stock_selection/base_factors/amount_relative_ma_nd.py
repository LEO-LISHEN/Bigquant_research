# -*- coding: utf-8 -*-
"""成交额相对 N 日均额因子。"""
import time
import numpy as np
import pandas as pd
def _prepare_daily_panel(data, target_dates, as_of_date, required_columns):
    if not isinstance(data, pd.DataFrame): raise TypeError("data 必须是 pandas.DataFrame。")
    columns = ["date", "instrument", *required_columns]; missing = set(columns) - set(data.columns)
    if missing: raise ValueError(f"amount_relative_ma_nd 缺少输入字段：{sorted(missing)}")
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

OUTPUT_COLUMNS = ["date", "instrument", "amount_relative_ma_nd"]


def _resolve_amount_relative_ma_nd_data_window(params):
    window = params.get("window", 20)
    if not isinstance(window, (int, np.integer)) or isinstance(window, (bool, np.bool_)) or window < 1: raise ValueError("window 必须是正整数。")
    return {"lookback_trading_days": int(window)-1, "requires_target_date_data": True, "minimum_history_observations": int(window)-1, "preheating_required": window > 1, "insufficient_window_behavior": "滚动窗口不足或均额为零时输出 NaN。"}


def calc_amount_relative_ma_nd(data, target_dates=None, as_of_date=None, window=20, show_progress=False, progress_every=20):
    """计算 ``amount_t / MA(amount, window)_t - 1``。"""
    del progress_every
    _resolve_amount_relative_ma_nd_data_window({"window": window}); started_at = time.perf_counter()
    df, targets = _prepare_daily_panel(data, target_dates, as_of_date, ("amount",))
    if df.empty or targets.empty: return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress: print(f"\r[amount_relative_ma_nd] [1/2] 计算 {window} 日均额...", end="", flush=True)
    try:
        average = df.groupby("instrument", sort=False)["amount"].transform(lambda item: item.rolling(int(window), min_periods=int(window)).mean())
        valid = np.isfinite(df.amount) & np.isfinite(average) & (average != 0)
        values = pd.Series(np.nan, index=df.index, dtype=float); values.loc[valid] = df.loc[valid, "amount"] / average.loc[valid] - 1.0
        result = df.loc[df.date.isin(targets), ["date", "instrument"]].copy(); result["amount_relative_ma_nd"] = values.loc[result.index].to_numpy()
        if show_progress: print(f"\r[amount_relative_ma_nd] [2/2] 完成 | 耗时 {time.perf_counter()-started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress: print()


FACTOR = {"name": "amount_relative_ma_nd", "func": calc_amount_relative_ma_nd,
          "factor_type": "base", "candidate_instances": {"20d": {"window": 20}},
          "input_schema": {"required": {"date": {}, "instrument": {}, "amount": {}}, "conditional": {}},
          "parameters": {"target_dates": {"default": None}, "as_of_date": {"default": None}, "window": {"default": 20}, "show_progress": {"default": False}, "progress_every": {"default": 20}},
          "data_window": {"resolver": _resolve_amount_relative_ma_nd_data_window, "default": _resolve_amount_relative_ma_nd_data_window({})},
          "output_schema": {"date": {}, "instrument": {}, "amount_relative_ma_nd": {"dtype": "float64"}}}
FACTOR_INFO = """# amount_relative_ma_nd\n\n成交额偏离因子：`amount_t / MA(amount, N)_t - 1`，其中 `MA` 为包含目标日的简单移动均值。参数 `window` 控制回看交易日数，默认 20。"""
