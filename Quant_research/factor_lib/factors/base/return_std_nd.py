# -*- coding: utf-8 -*-
"""N 日收益率样本标准差因子。"""
import time
import numpy as np
import pandas as pd
def _prepare_daily_panel(data, target_dates, as_of_date, required_columns):
    if not isinstance(data, pd.DataFrame): raise TypeError("data 必须是 pandas.DataFrame。")
    columns = ["date", "instrument", *required_columns]; missing = set(columns) - set(data.columns)
    if missing: raise ValueError(f"return_std_nd 缺少输入字段：{sorted(missing)}")
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

OUTPUT_COLUMNS = ["date", "instrument", "return_std_nd"]


def _resolve_return_std_nd_data_window(params):
    window = params.get("window", 5)
    if not isinstance(window, (int, np.integer)) or isinstance(window, (bool, np.bool_)) or window < 2: raise ValueError("window 必须是至少为 2 的整数。")
    return {"lookback_trading_days": int(window), "requires_target_date_data": True, "minimum_history_observations": int(window), "preheating_required": True, "insufficient_window_behavior": "收益率窗口不足时输出 NaN。"}


def calc_return_std_nd(data, target_dates=None, as_of_date=None, window=5, show_progress=False, progress_every=20):
    """计算最近 N 个日收益率的样本标准差（``ddof=1``）。"""
    del progress_every
    _resolve_return_std_nd_data_window({"window": window}); started_at = time.perf_counter()
    df, targets = _prepare_daily_panel(data, target_dates, as_of_date, ("close",))
    if df.empty or targets.empty: return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress: print(f"\r[return_std_nd] [1/2] 计算 {window} 日收益波动率...", end="", flush=True)
    try:
        returns = df.groupby("instrument", sort=False)["close"].pct_change(fill_method=None)
        values = returns.groupby(df["instrument"], sort=False).transform(lambda item: item.rolling(int(window), min_periods=int(window)).std(ddof=1))
        result = df.loc[df.date.isin(targets), ["date", "instrument"]].copy(); result["return_std_nd"] = values.loc[result.index].to_numpy()
        if show_progress: print(f"\r[return_std_nd] [2/2] 完成 | 耗时 {time.perf_counter()-started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress: print()


FACTOR = {"name": "return_std_nd", "func": calc_return_std_nd,
          "factor_type": "base", "candidate_instances": {"5d": {"window": 5}, "20d": {"window": 20}},
          "input_schema": {"required": {"date": {}, "instrument": {}, "close": {}}, "conditional": {}},
          "parameters": {"target_dates": {"default": None}, "as_of_date": {"default": None}, "window": {"default": 5}, "show_progress": {"default": False}, "progress_every": {"default": 20}},
          "data_window": {"resolver": _resolve_return_std_nd_data_window, "default": _resolve_return_std_nd_data_window({})},
          "output_schema": {"date": {}, "instrument": {}, "return_std_nd": {"dtype": "float64"}}}
FACTOR_INFO = """# return_std_nd\n\n收益波动率因子，计算最近 N 个日收益率的样本标准差（`ddof=1`）。参数 `window` 控制统计窗口，默认 5；数值越大表示近期价格波动越大。"""
