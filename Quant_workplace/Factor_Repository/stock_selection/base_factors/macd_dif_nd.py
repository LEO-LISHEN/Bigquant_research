# -*- coding: utf-8 -*-
"""MACD DIF 基础因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "macd_dif_nd"]


def _resolve_data_window(params):
    fast = int(params.get("fast_window", 12))
    slow = int(params.get("slow_window", 26))
    signal = int(params.get("signal_window", 9))
    multiplier = int(params.get("warmup_multiplier", 5))
    return {"lookback_trading_days": multiplier * slow + signal - 2, "requires_target_date_data": True, "minimum_history_observations": slow, "preheating_required": True, "insufficient_window_behavior": "样本不足慢线窗口时输出 NaN；建议按声明预热长度准备数据。"}


def _prepare(data, target_dates, as_of_date):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = ["date", "instrument", "close"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"macd_dif_nd 缺少输入字段：{missing}")
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


def calc_macd_dif_nd(data, target_dates=None, as_of_date=None, fast_window=12, slow_window=26, signal_window=9, warmup_multiplier=5, show_progress=False, progress_every=20):
    """计算 ``DIF = EMA_fast(close) - EMA_slow(close)``。"""
    del progress_every
    if not all(isinstance(value, int) and value >= 1 for value in (fast_window, slow_window, signal_window, warmup_multiplier)):
        raise ValueError("MACD 参数均必须为正整数。")
    if fast_window >= slow_window:
        raise ValueError("fast_window 必须小于 slow_window。")
    started_at = time.perf_counter()
    df, targets = _prepare(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress:
        print(f"\r[macd_dif_nd] [1/2] 计算 MACD DIF ({fast_window},{slow_window},{signal_window})...", end="", flush=True)
    try:
        del warmup_multiplier
        close = df["close"].where(np.isfinite(df["close"]) & df["close"].gt(0))
        fast_ema = close.groupby(df["instrument"], sort=False).transform(lambda series: series.ewm(span=fast_window, adjust=False, min_periods=fast_window).mean())
        slow_ema = close.groupby(df["instrument"], sort=False).transform(lambda series: series.ewm(span=slow_window, adjust=False, min_periods=slow_window).mean())
        df["macd_dif_nd"] = fast_ema - slow_ema
        result = df.loc[df["date"].isin(targets), OUTPUT_COLUMNS].copy()
        if show_progress:
            print(f"\r[macd_dif_nd] [2/2] 完成 | 耗时 {time.perf_counter() - started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "macd_dif_nd", "func": calc_macd_dif_nd, "factor_type": "base", "candidate_instances": {"12_26_9": {"fast_window": 12, "slow_window": 26, "signal_window": 9, "warmup_multiplier": 5}}, "category": "technical", "direction": -1,
    "description": "MACD 快慢指数均线差（DIF）；正值表示短期 EMA 高于长期 EMA。", "formula": "DIF_t = EMA_fast(close)_t - EMA_slow(close)_t。",
    "input_schema": {"required": {"date": {"dtype": "datetime64[ns]", "frequency": "daily", "meaning": "点时截面日期"}, "instrument": {"dtype": "string", "meaning": "证券代码"}, "close": {"dtype": "float64", "frequency": "daily", "meaning": "统一复权口径的收盘价"}}, "conditional": {}},
    "parameters": {"fast_window": {"default": 12, "range": "小于 slow_window 的正整数", "meaning": "快速 EMA 跨度"}, "slow_window": {"default": 26, "range": "大于 fast_window 的正整数", "meaning": "慢速 EMA 跨度并决定最少历史"}, "signal_window": {"default": 9, "range": "正整数", "meaning": "与完整 MACD 参数保持一致；DIF 本身不使用该值"}, "warmup_multiplier": {"default": 5, "range": "正整数", "meaning": "建议预热倍数；改变 data_window 声明，不改变 DIF 公式"}, "target_dates": {"default": None, "meaning": "实际输出因子截面；None 为输入全部日期"}, "as_of_date": {"default": None, "meaning": "全局信息截止日"}, "show_progress": {"default": False, "meaning": "是否显示单行进度"}, "progress_every": {"default": 20, "meaning": "兼容统一接口；核心计算采用向量化 EMA"}},
    "data_window": {"resolver": _resolve_data_window, "default": _resolve_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "macd_dif_nd": {"dtype": "float64", "meaning": "MACD DIF 原值"}},
    "usage_notes": "DIF 为原始价格尺度，横截面使用前通常需按研究目的标准化或与价格做相对化。direction=-1 仅沿用候选研究记录，不应自动决定建仓方向。", "pit_notes": "EMA 仅沿每只股票的历史正收盘价递推；收盘信号应在下一可交易时点执行。",
}

FACTOR_INFO = """# MACD DIF\n\nDIF 是快速 EMA 与慢速 EMA 的差，描述短期价格相对长期趋势的位置。该脚本输出原始 DIF，不将其除以价格，也不做标准化。"""
