# -*- coding: utf-8 -*-
"""股息率（Dividend Yield）基础因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "dividend_yield"]


def _resolve_data_window(_params):
    return {
        "lookback_trading_days": 0,
        "requires_target_date_data": True,
        "minimum_history_observations": 1,
        "preheating_required": False,
        "insufficient_window_behavior": "目标日缺少有效股息率时输出 NaN。",
    }


def _prepare(data, target_dates, as_of_date):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = ["date", "instrument", "dividend_yield_ratio"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"dividend_yield 缺少输入字段：{missing}")
    df = data.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any() or df["instrument"].isna().any():
        raise ValueError("date 或 instrument 存在缺失或无效值。")
    if df.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError("输入存在重复的 date + instrument 记录。")
    df["dividend_yield_ratio"] = pd.to_numeric(
        df["dividend_yield_ratio"], errors="coerce"
    )
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
    return df, targets


def calc_dividend_yield(
    data,
    target_dates=None,
    as_of_date=None,
    show_progress=False,
    progress_every=20,
):
    """返回 BigQuant 点时股息率原值，不转换其原始数值单位。"""
    del progress_every
    started_at = time.perf_counter()
    df, targets = _prepare(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress:
        print("\r[dividend_yield] [1/2] 整理股息率截面...", end="", flush=True)
    try:
        result = df.loc[df["date"].isin(targets), ["date", "instrument"]].copy()
        result["dividend_yield"] = df.loc[result.index, "dividend_yield_ratio"].to_numpy(dtype=float)
        result["dividend_yield"] = result["dividend_yield"].where(
            np.isfinite(result["dividend_yield"])
        )
        if show_progress:
            elapsed = time.perf_counter() - started_at
            print(f"\r[dividend_yield] [2/2] 完成 | 耗时 {elapsed:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "dividend_yield",
    "func": calc_dividend_yield,
    "factor_type": "base",
    "candidate_instances": {"default": {}},
    "category": "valuation",
    "direction": 1,
    "description": "点时股息率原值；较高值通常代表较高的现金分红收益率。",
    "formula": "dividend_yield_t = dividend_yield_ratio_t。",
    "input_schema": {
        "required": {
            "date": {"dtype": "datetime64[ns]", "frequency": "daily", "meaning": "点时截面日期"},
            "instrument": {"dtype": "string", "meaning": "证券代码"},
            "dividend_yield_ratio": {"dtype": "float64", "frequency": "daily_pit", "meaning": "数据源提供的点时股息率"},
        },
        "conditional": {},
    },
    "parameters": {
        "target_dates": {"default": None, "meaning": "实际输出因子截面；None 为输入全部日期"},
        "as_of_date": {"default": None, "meaning": "全局信息截止日"},
        "show_progress": {"default": False, "meaning": "是否显示单行进度"},
        "progress_every": {"default": 20, "meaning": "兼容统一接口；本因子为向量化计算"},
    },
    "data_window": {"resolver": _resolve_data_window, "default": _resolve_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "dividend_yield": {"dtype": "float64", "meaning": "股息率原值"}},
    "usage_notes": "保留适配器返回的原始数值单位，不在因子层乘以 100 或做横截面标准化。",
    "pit_notes": "dividend_yield_ratio 必须是目标日可获得的点时字段；数据适配器负责处理除权、公告时点与数据源口径。",
}

FACTOR_INFO = """# 股息率\n\n直接使用点时股息率字段衡量现金分红相对价格的水平。该脚本只输出原始暴露，去极值、中性化或标准化应由后续研究流程决定。"""
