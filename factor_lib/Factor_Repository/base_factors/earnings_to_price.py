# -*- coding: utf-8 -*-
"""EP（Earnings-to-Price）估值因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "earnings_to_price"]


def _resolve_data_window(_params):
    return {
        "lookback_trading_days": 0,
        "requires_target_date_data": True,
        "minimum_history_observations": 1,
        "preheating_required": False,
        "insufficient_window_behavior": "目标日缺少有效 PE_TTM 时输出 NaN。",
    }


def _prepare(data, target_dates, as_of_date):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = ["date", "instrument", "pe_ttm"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"earnings_to_price 缺少输入字段：{missing}")

    df = data.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any() or df["instrument"].isna().any():
        raise ValueError("date 或 instrument 存在缺失或无效值。")
    if df.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError("输入存在重复的 date + instrument 记录。")
    df["pe_ttm"] = pd.to_numeric(df["pe_ttm"], errors="coerce")

    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        if pd.isna(cutoff):
            raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= cutoff.normalize()].copy()

    available = pd.DatetimeIndex(df["date"].unique()).sort_values()
    if target_dates is None:
        targets = available
    else:
        values = [target_dates] if isinstance(target_dates, (str, pd.Timestamp)) else list(target_dates)
        targets = pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize().unique().sort_values()
        missing_dates = targets.difference(available)
        if not missing_dates.empty:
            raise ValueError(f"缺少目标日期原始数据：{missing_dates[:5].strftime('%Y-%m-%d').tolist()}")
    return df, targets


def calc_earnings_to_price(data, target_dates=None, as_of_date=None, show_progress=False, progress_every=20):
    """计算 ``EP = 1 / PE_TTM``；非正或缺失 PE_TTM 输出 NaN。"""
    del progress_every
    started_at = time.perf_counter()
    df, targets = _prepare(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress:
        print("\r[earnings_to_price] [1/2] 计算 EP=1/PE_TTM...", end="", flush=True)
    try:
        values = pd.Series(np.nan, index=df.index, dtype=float)
        valid = np.isfinite(df["pe_ttm"]) & df["pe_ttm"].gt(0)
        values.loc[valid] = 1.0 / df.loc[valid, "pe_ttm"]
        result = df.loc[df["date"].isin(targets), ["date", "instrument"]].copy()
        result["earnings_to_price"] = values.loc[result.index].to_numpy()
        if show_progress:
            print(f"\r[earnings_to_price] [2/2] 完成 | 耗时 {time.perf_counter() - started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "earnings_to_price", "func": calc_earnings_to_price,
    "factor_type": "base", "candidate_instances": {"default": {}},
    "category": "valuation", "direction": 1,
    "description": "盈利收益率，PE_TTM 的倒数；数值越高通常代表相对估值更低。",
    "formula": "earnings_to_price_t = 1 / pe_ttm_t；仅 pe_ttm_t > 0 时有效。",
    "input_schema": {"required": {
        "date": {"dtype": "datetime64[ns]", "frequency": "daily", "meaning": "点时截面日期"},
        "instrument": {"dtype": "string", "meaning": "证券代码"},
        "pe_ttm": {"dtype": "float64", "frequency": "daily_pit", "meaning": "滚动十二期市盈率"},
    }, "conditional": {}},
    "parameters": {
        "target_dates": {"default": None, "meaning": "实际输出因子截面；None 为输入全部日期"},
        "as_of_date": {"default": None, "meaning": "全局信息截止日"},
        "show_progress": {"default": False, "meaning": "是否显示单行进度"},
        "progress_every": {"default": 20, "meaning": "兼容统一接口；本因子为向量化计算"},
    },
    "data_window": {"resolver": _resolve_data_window, "default": _resolve_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "earnings_to_price": {"dtype": "float64", "meaning": "PE_TTM 倒数"}},
    "usage_notes": "PE 为负或零的公司不具有通常的盈利收益率解释，保留 NaN。",
    "pit_notes": "pe_ttm 必须是目标日可得的日频点时估值字段；因子只使用目标日数据。",
}

FACTOR_INFO = """# 盈利收益率（EP）

以滚动十二期市盈率的倒数衡量盈利相对价格的水平。`PE_TTM` 非正时不计算；较高值通常表示估值相对较低。"""
