# -*- coding: utf-8 -*-
"""对数收盘价（Log Price）基础因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "log_price"]


def _resolve_data_window(_params):
    return {"lookback_trading_days": 0, "requires_target_date_data": True, "minimum_history_observations": 1, "preheating_required": False, "insufficient_window_behavior": "非正或缺失收盘价输出 NaN。"}


def _prepare(data, target_dates, as_of_date):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = ["date", "instrument", "close"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"log_price 缺少输入字段：{missing}")
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
    return df, targets


def calc_log_price(data, target_dates=None, as_of_date=None, show_progress=False, progress_every=20):
    """计算 ``log(close)``。"""
    del progress_every
    started_at = time.perf_counter()
    df, targets = _prepare(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress:
        print("\r[log_price] [1/2] 计算对数价格...", end="", flush=True)
    try:
        result = df.loc[df["date"].isin(targets), ["date", "instrument"]].copy()
        close = df.loc[result.index, "close"]
        result["log_price"] = np.log(close.where(close.gt(0))).to_numpy(dtype=float)
        if show_progress:
            print(f"\r[log_price] [2/2] 完成 | 耗时 {time.perf_counter() - started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "log_price", "func": calc_log_price, "factor_type": "base", "candidate_instances": {"default": {}}, "category": "price_level", "direction": -1,
    "description": "收盘价的自然对数；较低值对应较低的名义价格水平。", "formula": "log_price_t = ln(close_t)，仅 close_t > 0 时有效。",
    "input_schema": {"required": {"date": {"dtype": "datetime64[ns]", "frequency": "daily", "meaning": "点时截面日期"}, "instrument": {"dtype": "string", "meaning": "证券代码"}, "close": {"dtype": "float64", "frequency": "daily", "meaning": "复权口径须由数据适配器统一的收盘价"}}, "conditional": {}},
    "parameters": {"target_dates": {"default": None, "meaning": "实际输出因子截面；None 为输入全部日期"}, "as_of_date": {"default": None, "meaning": "全局信息截止日"}, "show_progress": {"default": False, "meaning": "是否显示单行进度"}, "progress_every": {"default": 20, "meaning": "兼容统一接口；本因子为向量化计算"}},
    "data_window": {"resolver": _resolve_data_window, "default": _resolve_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "log_price": {"dtype": "float64", "meaning": "收盘价自然对数"}},
    "usage_notes": "名义价格并非稳定的经济尺度，建议作为候选特征而不是单独依赖的选股信号。direction=-1 是研究记录，不约束策略。", "pit_notes": "必须在适配器层明确 close 的复权口径，并仅使用目标日可取得的价格。",
}

FACTOR_INFO = """# 对数价格\n\n对目标日收盘价取自然对数，描述股票的名义价格水平。该指标受复权口径、拆并股和价格档位影响，应结合其他特征使用。"""
