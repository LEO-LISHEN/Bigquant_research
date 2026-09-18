# -*- coding: utf-8 -*-
"""N 日心理线（PSY）基础因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "psy_nd"]


def _resolve_data_window(params):
    window = int(params.get("window", 20))
    return {"lookback_trading_days": window, "requires_target_date_data": True, "minimum_history_observations": window + 1, "preheating_required": True, "insufficient_window_behavior": "连续有效收盘价不足 window+1 个交易日时输出 NaN。"}


def _prepare(data, target_dates, as_of_date):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = ["date", "instrument", "close"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"psy_nd 缺少输入字段：{missing}")
    df = data.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any() or df["instrument"].isna().any():
        raise ValueError("date 或 instrument 存在缺失或无效值。")
    if df.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError("输入存在重复的 date + instrument 记录。")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        if pd.isna(cutoff):
            raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= cutoff.normalize()].copy()
    available = pd.DatetimeIndex(df["date"].unique()).sort_values()
    if target_dates is None:
        targets = available
    else:
        raw = [target_dates] if isinstance(target_dates, (str, pd.Timestamp)) else list(target_dates)
        targets = pd.DatetimeIndex(pd.to_datetime(raw, errors="raise")).normalize().unique().sort_values()
        missing_dates = targets.difference(available)
        if not missing_dates.empty:
            raise ValueError(f"缺少目标日期原始数据：{missing_dates[:5].strftime('%Y-%m-%d').tolist()}")
    return df.sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True), targets


def calc_psy_nd(data, target_dates=None, as_of_date=None, window=20, show_progress=False, progress_every=20):
    """计算最近 N 日上涨日占比，取值范围通常为 0 到 100。"""
    del progress_every
    if not isinstance(window, int) or window < 1:
        raise ValueError("window 必须为正整数。")
    started_at = time.perf_counter()
    df, targets = _prepare(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress:
        print(f"\r[psy_nd] [1/2] 计算 {window} 日 PSY...", end="", flush=True)
    try:
        valid_close = df["close"].where(np.isfinite(df["close"]) & df["close"].gt(0))
        previous = valid_close.groupby(df["instrument"], sort=False).shift(1)
        up_flag = pd.Series(np.nan, index=df.index, dtype=float)
        comparable = valid_close.notna() & previous.notna()
        up_flag.loc[comparable] = valid_close.loc[comparable].gt(previous.loc[comparable]).astype(float)
        df["psy_nd"] = up_flag.groupby(df["instrument"], sort=False).transform(
            lambda series: series.rolling(window, min_periods=window).mean().mul(100.0)
        )
        result = df.loc[df["date"].isin(targets), OUTPUT_COLUMNS].copy()
        if show_progress:
            print(f"\r[psy_nd] [2/2] 完成 | 耗时 {time.perf_counter() - started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "psy_nd", "func": calc_psy_nd, "factor_type": "base", "candidate_instances": {"20d": {"window": 20}}, "category": "technical", "direction": -1,
    "description": "最近 N 个交易日上涨天数占比；较高值代表近期上涨日更集中。", "formula": "PSY_t = 100 × mean(1[close_i > close_{i-1}])，i=t-N+1,...,t。",
    "input_schema": {"required": {"date": {"dtype": "datetime64[ns]", "frequency": "daily", "meaning": "点时截面日期"}, "instrument": {"dtype": "string", "meaning": "证券代码"}, "close": {"dtype": "float64", "frequency": "daily", "meaning": "统一复权口径的收盘价"}}, "conditional": {}},
    "parameters": {"window": {"default": 20, "range": "正整数", "meaning": "统计上涨日的交易日窗口；改变预热长度"}, "target_dates": {"default": None, "meaning": "实际输出因子截面；None 为输入全部日期"}, "as_of_date": {"default": None, "meaning": "全局信息截止日"}, "show_progress": {"default": False, "meaning": "是否显示单行进度"}, "progress_every": {"default": 20, "meaning": "兼容统一接口；核心计算采用向量化滚动窗口"}},
    "data_window": {"resolver": _resolve_data_window, "default": _resolve_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "psy_nd": {"dtype": "float64", "meaning": "N 日上涨日占比，通常为 0 至 100"}},
    "usage_notes": "PSY 只计上涨天数、不计涨幅大小。direction=-1 记录常见的均值回复研究设定；动量设定也可能使用相反方向，应在研究中检验。", "pit_notes": "只使用目标日收盘后可得到的历史价格；若以收盘数据构建信号，交易应在下一可交易时点进行。",
}

FACTOR_INFO = """# N 日心理线（PSY）\n\nPSY 统计最近 N 个交易日中收盘上涨的比例，反映上涨日的集中程度。它不衡量每次涨跌的幅度，常作为超买超卖或短期趋势强度的辅助特征。"""
