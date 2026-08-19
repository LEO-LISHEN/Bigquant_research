# -*- coding: utf-8 -*-
"""严格交易日口径的 N 月换手率偏离因子。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "turnover_bias_nm"]


def _build_normal_trading_mask(data):
    """识别未停牌、有成交且未触及涨跌停的严格正常交易日。"""
    turn = pd.to_numeric(data["turn"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    upper = pd.to_numeric(data["upper_limit"], errors="coerce")
    lower = pd.to_numeric(data["lower_limit"], errors="coerce")
    suspended_raw = data["suspended"]
    if pd.api.types.is_bool_dtype(suspended_raw):
        suspended = suspended_raw.fillna(True)
    elif pd.api.types.is_numeric_dtype(suspended_raw):
        suspended = pd.to_numeric(suspended_raw, errors="coerce").fillna(1).ne(0)
    else:
        text = suspended_raw.astype(str).str.strip().str.lower()
        suspended = text.isin({"1", "true", "t", "yes", "y", "suspended"}) | suspended_raw.isna()
    tolerance = np.maximum(np.abs(close), 1.0) * 1e-8
    not_at_limit = (
        np.isfinite(close) & np.isfinite(upper) & np.isfinite(lower)
        & (close > 0) & (upper > lower)
        & (close < upper - tolerance) & (close > lower + tolerance)
    )
    return (
        np.isfinite(turn) & (turn > 0)
        & np.isfinite(volume) & (volume > 0)
        & (~suspended) & not_at_limit
    ).astype(bool)


def _resolve_data_window(params):
    n_months = params.get("n_months", 1)
    trading_days_per_month = params.get("trading_days_per_month", 21)
    if not isinstance(n_months, int) or isinstance(n_months, bool) or n_months < 1:
        raise ValueError("n_months 必须是正整数。")
    if not isinstance(trading_days_per_month, int) or isinstance(trading_days_per_month, bool) or trading_days_per_month < 1:
        raise ValueError("trading_days_per_month 必须是正整数。")
    return {
        "lookback_trading_days": n_months * trading_days_per_month,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": True,
        "insufficient_window_behavior": "不设置人为最低有效日门槛；目标日前没有任何正常交易日基准时输出NaN。",
    }


def _prepare(data, target_dates, as_of_date):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = ["date", "instrument", "turn", "volume", "close", "upper_limit", "lower_limit", "suspended"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"turnover_bias_nm 缺少字段：{missing}。")
    df = data.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any() or df["instrument"].isna().any():
        raise ValueError("date 或 instrument 包含无效值。")
    df["instrument"] = df["instrument"].astype(str)
    if df.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError("data 存在重复 date + instrument。")
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date).normalize()
        df = df.loc[df["date"] <= cutoff].copy()
    available = pd.DatetimeIndex(df["date"].unique()).sort_values()
    if target_dates is None:
        targets = available
    else:
        values = [target_dates] if isinstance(target_dates, (str, pd.Timestamp)) else list(target_dates)
        targets = pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize().unique().sort_values()
        missing_dates = targets.difference(available)
        if not missing_dates.empty:
            raise ValueError(f"缺少目标日原始数据：{missing_dates[:5].strftime('%Y-%m-%d').tolist()}。")
    return df.sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True), targets


def calc_turnover_bias_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=1,
    trading_days_per_month=21,
    show_progress=False,
    progress_every=20,
):
    """计算当前换手率相对过去 N 月正常交易日均值的偏离。

    ``turnover_bias = turn_t / mean(turn_{t-W:t-1}) - 1``。基准窗口不含
    目标日自身，避免当前异常换手被均值同步稀释。
    """
    del progress_every
    window_info = _resolve_data_window({"n_months": n_months, "trading_days_per_month": trading_days_per_month})
    window = window_info["lookback_trading_days"]
    started_at = time.perf_counter()
    df, targets = _prepare(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if show_progress:
        print(f"\r[turnover_bias_nm] [1/2] 筛选正常交易日并计算 {window} 日历史基准...", end="", flush=True)
    try:
        normal = _build_normal_trading_mask(df)
        turn = pd.to_numeric(df["turn"], errors="coerce").where(normal)
        baseline = turn.groupby(df["instrument"], sort=False).transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).mean()
        )
        current_turn = pd.to_numeric(df["turn"], errors="coerce")
        df["turnover_bias_nm"] = current_turn.div(baseline.where(baseline > 0)).sub(1.0).where(normal)
        result = df.loc[df["date"].isin(targets), OUTPUT_COLUMNS].copy()
        if show_progress:
            print(f"\r[turnover_bias_nm] [2/2] 完成 | {len(result):,} 条输出 | 耗时 {time.perf_counter() - started_at:.1f}s", end="", flush=True)
        return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "turnover_bias_nm",
    "func": calc_turnover_bias_nm,
    "factor_type": "base",
    "candidate_instances": {
        "1m": {"n_months": 1, "trading_days_per_month": 21},
        "3m": {"n_months": 3, "trading_days_per_month": 21},
        "6m": {"n_months": 6, "trading_days_per_month": 21},
        "12m": {"n_months": 12, "trading_days_per_month": 21},
    },
    "category": "liquidity",
    "direction": 0,
    "description": "当前正常交易日换手率相对过去 N 月正常交易日平均换手率的偏离。",
    "formula": "turnover_bias_t = turn_t / mean(turn_{t-W:t-1}) - 1。",
    "input_schema": {
        "required": {"date": {}, "instrument": {}, "turn": {}, "volume": {}, "close": {}, "upper_limit": {}, "lower_limit": {}, "suspended": {}},
        "conditional": {},
    },
    "parameters": {
        "n_months": {"default": 1, "range": "正整数", "meaning": "历史基准窗口月数。"},
        "trading_days_per_month": {"default": 21, "range": "正整数", "meaning": "月数到交易日窗口的换算口径。"},
        "target_dates": {"default": None, "meaning": "实际输出截面；None 为全部输入日期。"},
        "as_of_date": {"default": None, "meaning": "全局信息截止日。"},
        "show_progress": {"default": False, "meaning": "是否显示单行计算进度。"},
        "progress_every": {"default": 20, "meaning": "兼容统一接口；该计算为向量化滚动操作。"},
    },
    "data_window": {"resolver": _resolve_data_window, "default": _resolve_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "turnover_bias_nm": {"dtype": "float64", "meaning": "当前换手率相对历史正常交易基准的偏离率。"}},
    "usage_notes": "窗口不足时采用实际可得的正常交易日作为基准；基准窗口不含目标日。目标日停牌、无成交或触及涨跌停时输出NaN。",
    "pit_notes": "只使用目标日及以前的换手率与交易状态；历史基准显式排除目标日自身。",
}


FACTOR_INFO = """# N 月换手率偏离（严格口径）

该因子衡量当前换手率相对自身近期正常交易基准的异常程度。正值代表成交活跃度高于历史常态，负值代表低于常态。

历史基准不包含当天；停牌、无成交、涨停和跌停日不进入基准，也不在当日输出信号。窗口不足时使用实际可得的正常交易日，因此应配合覆盖率使用。
"""
