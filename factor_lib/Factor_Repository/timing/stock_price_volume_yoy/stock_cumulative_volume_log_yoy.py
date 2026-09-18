# -*- coding: utf-8 -*-
"""个股累计成交量对数同比因子。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


FACTOR_NAME = "stock_cumulative_volume_log_yoy"
OUTPUT_COLUMNS = ["date", "instrument", FACTOR_NAME]


def _normalize_target_dates(target_dates):
    if isinstance(target_dates, (str, pd.Timestamp, np.datetime64)):
        values = [target_dates]
    else:
        try:
            values = list(target_dates)
        except TypeError as exc:
            raise TypeError("target_dates 必须是日期或日期序列。") from exc
    return pd.DatetimeIndex(
        pd.to_datetime(values, errors="raise")
    ).normalize().unique().sort_values()


def _validate_progress_arguments(show_progress, progress_every):
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if (
        not isinstance(progress_every, (int, np.integer))
        or isinstance(progress_every, (bool, np.bool_))
        or progress_every <= 0
    ):
        raise ValueError("progress_every 必须是正整数。")


def _render_progress(completed, total, instrument, started_at, stage):
    elapsed = time.perf_counter() - started_at
    percentage = completed / total if total else 1.0
    message = (
        f"\r[{FACTOR_NAME}] {stage} | {completed}/{total} 只股票"
        f"（{percentage:.1%}）"
    )
    if instrument is not None:
        message += f" | 当前：{instrument}"
    message += f" | 已耗时：{elapsed:.1f}s"
    if 0 < completed < total:
        message += f" | 预计剩余：{elapsed / completed * (total - completed):.1f}s"
    print(message.ljust(180), end="", flush=True)


def _prepare_input_data(data, target_dates, as_of_date):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required_columns = {"date", "instrument", "volume"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"{FACTOR_NAME} 缺少输入字段：{missing_columns}。")

    df = data.loc[:, ["date", "instrument", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any() or df["instrument"].isna().any():
        raise ValueError(f"{FACTOR_NAME} 的 date 或 instrument 存在缺失或无效值。")
    if df.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError(f"{FACTOR_NAME} 的输入存在重复 date + instrument。")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    if as_of_date is not None:
        try:
            cutoff = pd.Timestamp(as_of_date).normalize()
        except (TypeError, ValueError) as exc:
            raise ValueError("as_of_date 必须是可解析日期。") from exc
        if pd.isna(cutoff):
            raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= cutoff].copy()

    if df.empty:
        if target_dates is None:
            return df, pd.DatetimeIndex([])
        requested = _normalize_target_dates(target_dates)
        if requested.empty:
            return df, requested
        raise ValueError(f"{FACTOR_NAME} 在截断后没有可用原始数据。")

    df = df.sort_values(["instrument", "date"], kind="mergesort")
    df["month"] = df["date"].dt.to_period("M")
    available_targets = pd.DatetimeIndex(
        df.groupby("month", sort=False)["date"].max().sort_values().to_numpy()
    )
    if target_dates is None:
        targets = available_targets
    else:
        targets = _normalize_target_dates(target_dates)
        missing = targets.difference(available_targets)
        if not missing.empty:
            examples = [date.strftime("%Y-%m-%d") for date in missing[:5]]
            raise ValueError(
                f"{FACTOR_NAME} 的 target_dates 必须是输入数据中每月最后一个"
                f"交易日；无效日期：{examples}。"
            )
    return df, targets


def _calculate_one_stock(stock_data, target_set):
    monthly = (
        stock_data.groupby("month", sort=True)
        .agg(date=("date", "max"), instrument=("instrument", "first"), volume=("volume", lambda values: values.sum() if np.isfinite(values.to_numpy(dtype=float)).all() and (values.to_numpy(dtype=float) >= 0).all() else np.nan))
    )
    if monthly.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    month_index = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    indexed = monthly.reindex(month_index)
    cumulative = indexed["volume"].astype(float).rolling(12, min_periods=12).sum()
    previous = cumulative.shift(12)
    current_values = cumulative.to_numpy(dtype=float)
    previous_values = previous.to_numpy(dtype=float)
    valid = (
        np.isfinite(current_values)
        & np.isfinite(previous_values)
        & (current_values > 0)
        & (previous_values > 0)
    )
    values = np.full(len(indexed), np.nan, dtype=float)
    values[valid] = np.log(current_values[valid]) - np.log(previous_values[valid])
    indexed[FACTOR_NAME] = values
    result = indexed.loc[indexed["date"].notna(), ["date", "instrument", FACTOR_NAME]].copy()
    return result.loc[result["date"].isin(target_set)]


def calc_stock_cumulative_volume_log_yoy(
    data,
    target_dates=None,
    as_of_date=None,
    show_progress=False,
    progress_every=20,
):
    """计算每只股票近 12 月累计成交量的严格 12 个月对数同比。

    先将日成交量汇总为自然月成交量 ``V_t``，再计算
    ``ln(sum(V_{t-11},...,V_t)) - ln(sum(V_{t-23},...,V_{t-12}))``。
    函数不在因子层筛选股票或形成交易信号。
    """
    _validate_progress_arguments(show_progress, progress_every)
    df, targets = _prepare_input_data(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    target_set = set(targets)
    grouped = list(df.groupby("instrument", sort=False))
    total = len(grouped)
    started_at = time.perf_counter()
    parts = []
    if show_progress:
        _render_progress(0, total, None, started_at, "开始计算累计成交量对数同比")
    try:
        for position, (instrument, stock_data) in enumerate(grouped, start=1):
            parts.append(_calculate_one_stock(stock_data, target_set))
            if show_progress and (
                position == 1 or position % progress_every == 0 or position == total
            ):
                _render_progress(position, total, instrument, started_at, "已完成股票计算")
    finally:
        if show_progress:
            print()

    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["date", "instrument"], keep=False).any():
        raise RuntimeError(f"{FACTOR_NAME} 输出出现重复主键。")
    return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)


FACTOR = {
    "name": FACTOR_NAME,
    "func": calc_stock_cumulative_volume_log_yoy,
    "factor_type": "base",
    "factor_role": "timing",
    "primary_data_domain": "security_daily",
    "candidate_instances": {"default": {}},
    "category": "stock_price_volume_timing",
    "direction": 1,
    "description": "个股近 12 个自然月累计成交量相对前一年度累计成交量的对数变化。",
    "formula": "VC_t=sum(V_{t-11},...,V_t); stock_cumulative_volume_log_yoy_t=ln(VC_t)-ln(VC_{t-12})",
    "input_schema": {
        "required": {
            "date": {"dtype": "datetime64[ns] 或可解析日期", "frequency": "daily", "data_domain": "security_daily", "meaning": "个股交易日。"},
            "instrument": {"dtype": "string", "frequency": "daily", "data_domain": "security_daily", "meaning": "股票代码。"},
            "volume": {"dtype": "float64", "frequency": "daily", "data_domain": "security_daily", "meaning": "个股日成交量。"},
        },
        "conditional": {},
    },
    "parameters": {
        "target_dates": {"default": None, "accepted_values": "每月最后一个交易日的日期序列或 None", "effect": "限定输出的月末股票截面。", "changes_data_requirements": False},
        "as_of_date": {"default": None, "accepted_values": "可解析日期或 None", "effect": "全局信息截止日。", "changes_data_requirements": False},
        "show_progress": {"default": False, "accepted_values": "bool", "effect": "显示按股票计数的单行进度。", "changes_data_requirements": False},
        "progress_every": {"default": 20, "accepted_values": "正整数", "effect": "每处理多少只股票刷新一次进度。", "changes_data_requirements": False},
    },
    "data_window": {
        "default": {"lookback_trading_days": 540, "requires_target_date_data": True, "minimum_history_observations": 24, "preheating_required": True, "insufficient_window_behavior": "缺少连续 24 个自然月的有效月成交量时输出 NaN，不填充。"},
        "resolver_notes": "540 个交易日仅用于覆盖 24 个自然月预热，非公式参数。",
    },
    "output_schema": {
        "date": {"dtype": "datetime64[ns]", "meaning": "目标月最后一个交易日。"},
        "instrument": {"dtype": "string", "meaning": "股票代码。"},
        FACTOR_NAME: {"dtype": "float64", "meaning": "近 12 月累计成交量的 12 个月对数同比。"},
    },
    "usage_notes": ["这是个股级时序筛选组件，不是横截面排名因子。", "target_dates 必须为输入数据中每月最后一个交易日。"],
    "pit_notes": ["仅使用目标月最后一个交易日及以前的日成交量。", "T 日收盘后得到的指标最早在 T+1 交易日可用于下单。"],
    "references": ["华泰证券《周期择时1：华泰价量择时模型》，2016-09-12"],
    "status": "research",
    "version": "1.0.0",
}


FACTOR_INFO = """# 个股累计成交量对数同比\n\n使用个股日成交量汇总得到月成交量，再计算近 12 个月累计成交量的对数同比。"""
