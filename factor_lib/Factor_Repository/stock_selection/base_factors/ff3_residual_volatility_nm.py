# -*- coding: utf-8 -*-
"""N 月 Fama-French 三因子风格残差波动率。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "ff3_residual_volatility_nm"]


def _build_monthly_return_panels(security_daily, domain_data, market_index, as_of_date):
    """在本因子内部将股票及市场日频收盘价整理为月末收益率。"""
    if domain_data is None or not hasattr(domain_data, "get_domain"):
        raise TypeError("ff3_residual_volatility_nm 需要包含 market_daily 的 FactorDataBundle。")
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
        "month_end_dates": month_end_dates,
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
        "lookback_trading_days": (n_months + 1) * trading_days_per_month + 1,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": True,
        # 本因子只以每个自然月最后一个交易日构造月收益、SMB 和 HML；
        # 目标日自身仍须保留，用于输出该日股票的因子值。该声明由通用
        # SVM 自动特征读取器解释，不应由下游按因子名称编写特例。
        "input_date_sampling": "month_end_plus_target_dates",
        "insufficient_window_behavior": (
            "不设置人为最低观测门槛；仅在三因子回归数学上不可定义时输出NaN。"
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


def _cap_weighted_return(returns, caps, mask):
    values = returns[mask]
    weights = caps[mask]
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def _build_ff3_style_returns(stock_returns, month_end_cap, month_end_pb, show_progress, progress_every, started_at):
    """按滞后月末市值与 BP 构造 value-weighted SMB/HML。"""
    dates = stock_returns.index
    smb = np.full(len(dates), np.nan, dtype=float)
    hml = np.full(len(dates), np.nan, dtype=float)
    for position in range(1, len(dates)):
        returns = stock_returns.iloc[position]
        caps = month_end_cap.iloc[position - 1]
        pb = month_end_pb.iloc[position - 1]
        bp = 1.0 / pb.where(pb > 0)
        valid = returns.notna() & caps.notna() & (caps > 0) & bp.notna() & (bp > 0)
        if valid.any():
            cap_values = caps[valid]
            bp_values = bp[valid]
            size_cut = cap_values.median()
            value_low = bp_values.quantile(0.30)
            value_high = bp_values.quantile(0.70)
            small = valid & (caps <= size_cut)
            big = valid & (caps > size_cut)
            low = valid & (bp <= value_low)
            middle = valid & (bp > value_low) & (bp <= value_high)
            high = valid & (bp > value_high)
            portfolios = {
                "sl": _cap_weighted_return(returns, caps, small & low),
                "sm": _cap_weighted_return(returns, caps, small & middle),
                "sh": _cap_weighted_return(returns, caps, small & high),
                "bl": _cap_weighted_return(returns, caps, big & low),
                "bm": _cap_weighted_return(returns, caps, big & middle),
                "bh": _cap_weighted_return(returns, caps, big & high),
            }
            values = np.asarray(list(portfolios.values()), dtype=float)
            if np.isfinite(values).all():
                smb[position] = np.mean([portfolios["sl"], portfolios["sm"], portfolios["sh"]]) - np.mean([portfolios["bl"], portfolios["bm"], portfolios["bh"]])
                hml[position] = np.mean([portfolios["sh"], portfolios["bh"]]) - np.mean([portfolios["sl"], portfolios["bl"]])
        completed = position + 1
        if show_progress and (completed == 2 or completed % progress_every == 0 or completed == len(dates)):
            elapsed = time.perf_counter() - started_at
            eta = elapsed / completed * (len(dates) - completed)
            print(f"\r[ff3_residual_volatility_nm] [1/2] 构造 SMB/HML {completed}/{len(dates)} ({completed / len(dates):.1%}) | 当前 {dates[position]:%Y-%m-%d} | 已耗时 {elapsed:.1f}s | 预计剩余 {eta:.1f}s", end="", flush=True)
    return pd.DataFrame({"smb": smb, "hml": hml}, index=dates)


def _residual_volatility(stock_returns, factor_returns):
    """逐股票执行含截距三因子 OLS，返回残差样本标准差。"""
    y_matrix = stock_returns.to_numpy(dtype=float)
    x_matrix = factor_returns.to_numpy(dtype=float)
    result = np.full(y_matrix.shape[1], np.nan, dtype=float)
    for column in range(y_matrix.shape[1]):
        y = y_matrix[:, column]
        valid = np.isfinite(y) & np.isfinite(x_matrix).all(axis=1)
        # 截距 + 3 个因子：至少要有一个残差自由度。
        if int(valid.sum()) <= 4:
            continue
        design = np.column_stack([np.ones(int(valid.sum())), x_matrix[valid]])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        try:
            coefficients = np.linalg.lstsq(design, y[valid], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        residual = y[valid] - design @ coefficients
        if len(residual) > 1:
            value = np.std(residual, ddof=1)
            if np.isfinite(value):
                result[column] = value
    return result


def calc_ff3_residual_volatility_nm(
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
    """计算 N 月三因子风格模型残差的样本标准差。

    MKT 为指定市场指数月收益；SMB/HML 在因子内部由滞后月末市值和
    BP 的 2×3 组合构造。目标月不参与计算，月内目标日复用最近完整月
    的估计，保证不读取尚未结束月份的收益。
    """
    _resolve_data_window({"n_months": n_months, "trading_days_per_month": trading_days_per_month})
    if not isinstance(progress_every, int) or isinstance(progress_every, bool) or progress_every < 1:
        raise ValueError("progress_every 必须是正整数。")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = {"date", "instrument", "close", "total_market_cap", "pb"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"ff3_residual_volatility_nm 缺少字段：{missing}。")
    if data.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError("data 存在重复 date + instrument。")

    started_at = time.perf_counter()
    targets = _normalize_targets(data, target_dates, as_of_date)
    if targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    source = data.loc[:, ["date", "instrument", "close", "total_market_cap", "pb"]].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    source["instrument"] = source["instrument"].astype(str)
    if as_of_date is not None:
        source = source.loc[source["date"] <= pd.Timestamp(as_of_date).normalize()].copy()
    for column in ["total_market_cap", "pb"]:
        source[column] = pd.to_numeric(source[column], errors="coerce")

    panels = _build_monthly_return_panels(source, domain_data, market_index, as_of_date)
    stock_returns = panels["stock_returns"]
    market_returns = panels["market_returns"]
    month_end_dates = panels["month_end_dates"]
    month_periods = panels["month_periods"]
    instruments = stock_returns.columns
    month_end_cap = source.pivot(index="date", columns="instrument", values="total_market_cap").reindex(index=month_end_dates, columns=instruments)
    month_end_pb = source.pivot(index="date", columns="instrument", values="pb").reindex(index=month_end_dates, columns=instruments)
    style_returns = _build_ff3_style_returns(stock_returns, month_end_cap, month_end_pb, show_progress, int(progress_every), started_at)
    factor_returns = pd.concat([market_returns.rename("mkt"), style_returns], axis=1)
    target_state = source.loc[:, ["date", "instrument"]]

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
                    cached[end_position] = _residual_volatility(
                        stock_returns.iloc[start_position:end_position + 1],
                        factor_returns.iloc[start_position:end_position + 1],
                    )
                values = cached[end_position]
            target_instruments = pd.Index(target_state.loc[target_state["date"] == target, "instrument"].drop_duplicates())
            factor = pd.Series(values, index=instruments).reindex(target_instruments)
            result_parts.append(pd.DataFrame({"date": target, "instrument": target_instruments, "ff3_residual_volatility_nm": factor.to_numpy()}))
            if show_progress and (position == 1 or position % progress_every == 0 or position == total):
                elapsed = time.perf_counter() - started_at
                eta = elapsed / position * (total - position)
                print(f"\r[ff3_residual_volatility_nm] [2/2] 回归残差波动率 {position}/{total} ({position / total:.1%}) | 当前 {target:%Y-%m-%d} | 已耗时 {elapsed:.1f}s | 预计剩余 {eta:.1f}s", end="", flush=True)
        return pd.concat(result_parts, ignore_index=True).sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "ff3_residual_volatility_nm",
    "func": calc_ff3_residual_volatility_nm,
    "factor_type": "base",
    "candidate_instances": {
        "36m": {"n_months": 36, "trading_days_per_month": 21, "market_index": "csi_all_share"},
        "60m": {"n_months": 60, "trading_days_per_month": 21, "market_index": "csi_all_share"},
    },
    "category": "risk",
    "direction": 0,
    "description": "个股月收益对市场、规模、价值三因子回归后的 N 月残差波动率。",
    "formula": "std(epsilon_i)，其中 r_i = alpha + b_m*MKT + b_s*SMB + b_h*HML + epsilon_i。",
    "input_schema": {"required": {"date": {}, "instrument": {}, "close": {}, "total_market_cap": {}, "pb": {}, "market_close": {}}, "conditional": {}},
    "parameters": {
        "n_months": {"default": 36, "range": "正整数", "meaning": "最多使用的完整月收益数量，改变预热期。"},
        "trading_days_per_month": {"default": 21, "range": "正整数", "meaning": "仅用于估计日频原始数据预热长度。"},
        "market_index": {"default": "csi_all_share", "meaning": "市场代理指数统一名称。"},
        "target_dates": {"default": None, "meaning": "实际输出截面；None 为全部输入日期。"},
        "as_of_date": {"default": None, "meaning": "全局信息截止日。"},
        "show_progress": {"default": False, "meaning": "是否显示单行计算进度。"},
        "progress_every": {"default": 20, "meaning": "月度风格构造和目标日循环的刷新间隔。"},
    },
    "data_window": {"resolver": _resolve_data_window, "default": _resolve_data_window({})},
    "output_schema": {"date": {}, "instrument": {}, "ff3_residual_volatility_nm": {"dtype": "float64", "meaning": "三因子未解释收益的月度样本波动率。"}},
    "usage_notes": "SMB/HML 使用全股票样本的滞后月末市值、BP 进行 2×3 分组，并以市值加权组合收益构造；该实现是可复现的 FF3 风格口径，未引入无风险利率。",
    "pit_notes": "用于某月收益的市值与PB均取前一月末；月内目标日仅使用上一个完整自然月及以前数据。",
}


FACTOR_INFO = """# N 月三因子残差波动率

先以市场、规模和价值风格收益解释个股月收益，再计算剩余残差的波动率。较高数值表示股票具有更强、且难以由常见共同风险解释的特质波动。

市场因子使用指定指数收益；SMB 与 HML 在内部以滞后月末市值和 BP 的 2×3 组合构造。常用实例为 36 月与 60 月；短历史股票只要回归数学上可定义便输出结果，但其估计稳定性较低。
"""
