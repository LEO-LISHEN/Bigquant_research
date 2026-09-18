# -*- coding: utf-8 -*-
"""N 月滚动历史市场 Alpha 因子。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "h_alpha_nm"]


def _build_monthly_return_panels(security_daily, domain_data, market_index, as_of_date):
    """在本因子内部将股票及市场日频收盘价整理为月末收益率。"""
    if domain_data is None or not hasattr(domain_data, "get_domain"):
        raise TypeError("h_alpha_nm 需要包含 market_daily 的 FactorDataBundle。")
    stock = security_daily.loc[:, ["date", "instrument", "close"]].copy()
    stock["date"] = pd.to_datetime(stock["date"], errors="coerce").dt.normalize()
    stock["instrument"] = stock["instrument"].astype(str)
    stock["close"] = pd.to_numeric(stock["close"], errors="coerce").where(lambda series: series > 0)
    market = domain_data.get_domain("market_daily").copy()
    required = {"date", "market_index", "market_close"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"market_daily 缺少字段：{missing}。")
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market = market.loc[market["market_index"].astype(str) == market_index, ["date", "market_close"]].copy()
    market["market_close"] = pd.to_numeric(market["market_close"], errors="coerce").where(lambda series: series > 0)
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date).normalize()
        stock = stock.loc[stock["date"] <= cutoff]
        market = market.loc[market["date"] <= cutoff]
    if market.empty or market["date"].isna().any() or market.duplicated("date", keep=False).any():
        raise ValueError(f"指数 {market_index!r} 的月末收益原始数据无效或不足。")
    market = market.sort_values("date", kind="mergesort")
    market["month"] = market["date"].dt.to_period("M")
    month_end = market.groupby("month", sort=True, as_index=False).tail(1).sort_values("date", kind="mergesort")
    month_end_dates = pd.DatetimeIndex(month_end["date"])
    if len(month_end_dates) < 2:
        raise ValueError("市场指数月末数据不足，无法构造月收益。")
    instruments = pd.Index(stock["instrument"].unique()).sort_values()
    stock_close = stock.pivot(index="date", columns="instrument", values="close").reindex(index=month_end_dates, columns=instruments)
    market_close = pd.Series(month_end["market_close"].to_numpy(dtype=float), index=month_end_dates)
    return {
        "stock_returns": stock_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan),
        "market_returns": market_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan),
        "month_periods": pd.PeriodIndex(month_end["month"], freq="M"),
    }


def _resolve_data_window(params):
    n_months = params.get("n_months", 36)
    trading_days_per_month = params.get("trading_days_per_month", 21)
    if not isinstance(n_months, int) or isinstance(n_months, bool) or n_months < 1:
        raise ValueError("n_months 必须是正整数。")
    if not isinstance(trading_days_per_month, int) or isinstance(trading_days_per_month, bool) or trading_days_per_month < 1:
        raise ValueError("trading_days_per_month 必须是正整数。")
    return {
        "lookback_trading_days": n_months * trading_days_per_month + 1,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": True,
        "insufficient_window_behavior": (
            "不设置人为最低观测门槛；仅在少于2个有效月收益配对，"
            "或市场收益方差为0时输出NaN。"
        ),
    }


def _normalize_targets(data, target_dates, as_of_date):
    dates = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("date 包含无效值。")
    available = pd.DatetimeIndex(dates.unique()).sort_values()
    if as_of_date is not None:
        available = available[available <= pd.Timestamp(as_of_date).normalize()]
    if target_dates is None:
        return available
    values = [target_dates] if isinstance(target_dates, (str, pd.Timestamp)) else list(target_dates)
    result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize().unique().sort_values()
    missing = result.difference(available)
    if not missing.empty:
        raise ValueError(f"缺少目标日原始数据：{missing[:5].strftime('%Y-%m-%d').tolist()}。")
    return result


def _market_alpha(stock_returns, market_returns):
    """按列计算简单市场模型的 OLS 截距。"""
    y = stock_returns.to_numpy(dtype=float)
    x = market_returns.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(x)[:, None]
    count = valid.sum(axis=0).astype(float)
    x_matrix = np.where(valid, x[:, None], 0.0)
    y_matrix = np.where(valid, y, 0.0)
    sum_x = x_matrix.sum(axis=0)
    sum_y = y_matrix.sum(axis=0)
    denom_count = np.where(count > 0, count, 1.0)
    cov = (x_matrix * y_matrix).sum(axis=0) - sum_x * sum_y / denom_count
    var_x = (x_matrix * x_matrix).sum(axis=0) - sum_x * sum_x / denom_count
    result = np.full(y.shape[1], np.nan, dtype=float)
    usable = (count >= 2) & np.isfinite(var_x) & (var_x > np.finfo(float).eps)
    beta = np.full(y.shape[1], np.nan, dtype=float)
    beta[usable] = cov[usable] / var_x[usable]
    result[usable] = sum_y[usable] / count[usable] - beta[usable] * sum_x[usable] / count[usable]
    return result


def calc_h_alpha_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=36,
    trading_days_per_month=21,
    market_index="csi_all_share",
    show_progress=False,
    progress_every=20,
    domain_data=None,
):
    """计算简单市场模型的 N 月滚动截距（历史 Alpha）。"""
    _resolve_data_window({"n_months": n_months, "trading_days_per_month": trading_days_per_month})
    if not isinstance(progress_every, int) or isinstance(progress_every, bool) or progress_every < 1:
        raise ValueError("progress_every 必须是正整数。")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = {"date", "instrument", "close"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"h_alpha_nm 缺少字段：{missing}。")
    if data.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError("data 存在重复 date + instrument。")

    started_at = time.perf_counter()
    targets = _normalize_targets(data, target_dates, as_of_date)
    if targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    panels = _build_monthly_return_panels(data, domain_data, market_index, as_of_date)
    stock_returns = panels["stock_returns"]
    market_returns = panels["market_returns"]
    month_periods = panels["month_periods"]
    instruments = stock_returns.columns
    target_state = data.copy()
    target_state["date"] = pd.to_datetime(target_state["date"], errors="coerce").dt.normalize()
    target_state["instrument"] = target_state["instrument"].astype(str)
    if as_of_date is not None:
        target_state = target_state.loc[target_state["date"] <= pd.Timestamp(as_of_date).normalize()]

    cached = {}
    result_parts = []
    total = len(targets)
    try:
        for position, target in enumerate(targets, start=1):
            completed_period = target.to_period("M") - 1
            end_position = month_periods.get_indexer([completed_period])[0]
            values = np.full(len(instruments), np.nan, dtype=float)
            if end_position >= 0:
                if end_position not in cached:
                    start_position = max(0, end_position - int(n_months) + 1)
                    cached[end_position] = _market_alpha(
                        stock_returns.iloc[start_position:end_position + 1],
                        market_returns.iloc[start_position:end_position + 1],
                    )
                values = cached[end_position]
            target_instruments = pd.Index(target_state.loc[target_state["date"] == target, "instrument"].drop_duplicates())
            factor = pd.Series(values, index=instruments).reindex(target_instruments)
            result_parts.append(pd.DataFrame({"date": target, "instrument": target_instruments, "h_alpha_nm": factor.to_numpy()}))
            if show_progress and (position == 1 or position % progress_every == 0 or position == total):
                elapsed = time.perf_counter() - started_at
                eta = elapsed / position * (total - position)
                print(f"\r[h_alpha_nm] 回归计算 {position}/{total} ({position / total:.1%}) | 当前 {target:%Y-%m-%d} | 已耗时 {elapsed:.1f}s | 预计剩余 {eta:.1f}s", end="", flush=True)
        return pd.concat(result_parts, ignore_index=True).sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "h_alpha_nm",
    "func": calc_h_alpha_nm,
    "factor_type": "base",
    "candidate_instances": {
        "36m": {"n_months": 36, "trading_days_per_month": 21, "market_index": "csi_all_share"},
        "60m": {"n_months": 60, "trading_days_per_month": 21, "market_index": "csi_all_share"},
    },
    "category": "risk",
    "direction": 0,
    "description": "个股月收益对指定市场指数月收益的 N 月滚动简单市场模型截距。",
    "formula": "r_i = alpha_i + beta_i * r_m + epsilon_i；输出 alpha_i。",
    "input_schema": {"required": {"date": {}, "instrument": {}, "close": {}, "market_close": {}}, "conditional": {}},
    "parameters": {
        "n_months": {"default": 36, "range": "正整数", "meaning": "最多使用的完整月收益数量，改变预热期。"},
        "trading_days_per_month": {"default": 21, "range": "正整数", "meaning": "仅用于估计日频原始数据预热长度。"},
        "market_index": {"default": "csi_all_share", "meaning": "市场代理指数统一名称。"},
        "target_dates": {"default": None, "meaning": "实际输出截面；None 为全部输入日期。"},
        "as_of_date": {"default": None, "meaning": "全局信息截止日。"},
        "show_progress": {"default": False, "meaning": "是否显示单行计算进度。"},
        "progress_every": {"default": 20, "meaning": "目标日循环的刷新间隔。"},
    },
    "data_window": {"resolver": _resolve_data_window, "default": _resolve_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "h_alpha_nm": {"dtype": "float64", "meaning": "历史市场 Alpha；正值代表控制市场收益后的历史平均残差收益为正。"}},
    "usage_notes": "这里是未扣无风险利率的简单市场模型截距，不等同于严格 CAPM Alpha。短历史股票会以可得月收益估计，稳定性需在研究层单独评估。",
    "pit_notes": "月内目标日仅使用上一个完整自然月及以前数据；市场指数由数据适配器按 market_index 点时加载。",
}


FACTOR_INFO = """# N 月历史市场 Alpha

对个股与市场指数的月收益做滚动简单市场模型回归，输出截距。该值描述无法由市场共同涨跌解释的历史平均收益部分。

这不是严格的 CAPM 超额收益 Alpha，因为当前口径没有减去无风险利率。常用实例为 36 月和 60 月；不设置人为最低样本门槛，但回归必须在数学上可定义。
"""
