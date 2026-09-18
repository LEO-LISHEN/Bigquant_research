# -*- coding: utf-8 -*-
"""对数总市值（Log Market Cap）基础因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "log_market_cap"]


def _resolve_data_window(_params):
    return {"lookback_trading_days": 0, "requires_target_date_data": True, "minimum_history_observations": 1, "preheating_required": False, "insufficient_window_behavior": "非正或缺失市值输出 NaN。"}


def _prepare(data, target_dates, as_of_date):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = ["date", "instrument", "total_market_cap"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"log_market_cap 缺少输入字段：{missing}")
    df = data.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any() or df["instrument"].isna().any():
        raise ValueError("date 或 instrument 存在缺失或无效值。")
    if df.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError("输入存在重复的 date + instrument 记录。")
    df["total_market_cap"] = pd.to_numeric(df["total_market_cap"], errors="coerce")
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


def calc_log_market_cap(data, target_dates=None, as_of_date=None, show_progress=False, progress_every=20):
    """计算 ``log(total_market_cap)``。"""
    del progress_every
    started_at = time.perf_counter()
    df, targets = _prepare(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress:
        print("\r[log_market_cap] [1/2] 计算对数市值...", end="", flush=True)
    try:
        result = df.loc[df["date"].isin(targets), ["date", "instrument"]].copy()
        cap = df.loc[result.index, "total_market_cap"]
        result["log_market_cap"] = np.log(cap.where(cap.gt(0))).to_numpy(dtype=float)
        if show_progress:
            print(f"\r[log_market_cap] [2/2] 完成 | 耗时 {time.perf_counter() - started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "log_market_cap", "func": calc_log_market_cap, "factor_type": "base", "candidate_instances": {"default": {}}, "category": "size", "direction": -1,
    "description": "总市值的自然对数；较小市值对应较低的因子值。", "formula": "log_market_cap_t = ln(total_market_cap_t)，仅 total_market_cap_t > 0 时有效。",
    "input_schema": {"required": {"date": {"dtype": "datetime64[ns]", "frequency": "daily", "meaning": "点时截面日期"}, "instrument": {"dtype": "string", "meaning": "证券代码"}, "total_market_cap": {"dtype": "float64", "frequency": "daily_pit", "meaning": "总市值"}}, "conditional": {}},
    "parameters": {"target_dates": {"default": None, "meaning": "实际输出因子截面；None 为输入全部日期"}, "as_of_date": {"default": None, "meaning": "全局信息截止日"}, "show_progress": {"default": False, "meaning": "是否显示单行进度"}, "progress_every": {"default": 20, "meaning": "兼容统一接口；本因子为向量化计算"}},
    "data_window": {"resolver": _resolve_data_window, "default": _resolve_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "log_market_cap": {"dtype": "float64", "meaning": "总市值自然对数"}},
    "usage_notes": "这是规模暴露而不是中性化控制变量；direction=-1 仅记录常见小市值研究方向，不应由策略强制使用。", "pit_notes": "total_market_cap 必须是目标日可得的点时市值字段；不使用未来市值。",
}

FACTOR_INFO = """# 对数总市值\n\n对总市值取自然对数，压缩大市值股票的数量级差异。它常用于描述规模暴露，也可作为其他因子的市值中性化控制变量。"""
