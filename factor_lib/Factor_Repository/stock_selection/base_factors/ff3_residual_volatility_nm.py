# -*- coding: utf-8 -*-
"""N 月 Fama-French 三因子风格残差波动率：月末稀疏数据优化版。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


FACTOR_NAME = "ff3_residual_volatility_nm"
OUTPUT_COLUMNS = ["date", "instrument", FACTOR_NAME]


def _resolve_data_window(params):
    n_months = params.get("n_months", 36)
    days = params.get("trading_days_per_month", 21)
    if not isinstance(n_months, int) or isinstance(n_months, bool) or n_months < 1:
        raise ValueError("n_months 必须是正整数。")
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        raise ValueError("trading_days_per_month 必须是正整数。")
    return {
        "lookback_trading_days": (n_months + 1) * days + 1,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": True,
        "input_date_sampling": "month_end_plus_target_dates",
        "insufficient_window_behavior": "回归数学上不可定义时输出 NaN。",
    }


def _progress(show_progress, stage, completed, total, started_at, detail=""):
    if not show_progress:
        return
    elapsed = time.perf_counter() - started_at
    text = f"\r[{FACTOR_NAME}] [{stage}] {completed}/{total} ({completed / total:.1%}) | 已耗时 {elapsed:.1f}s"
    if detail:
        text += f" | {detail}"
    if 0 < completed < total:
        text += f" | 预计剩余 {elapsed / completed * (total - completed):.1f}s"
    print(text.ljust(180), end="", flush=True)


def _targets(data, target_dates, as_of_date):
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
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan


def _build_style_returns(stock_returns, caps, pb, show_progress, progress_every, started_at):
    dates = stock_returns.index
    smb = np.full(len(dates), np.nan)
    hml = np.full(len(dates), np.nan)
    total = max(len(dates) - 1, 1)
    for index in range(1, len(dates)):
        returns = stock_returns.iloc[index]
        prior_cap = caps.iloc[index - 1]
        bp = 1.0 / pb.iloc[index - 1].where(pb.iloc[index - 1] > 0)
        valid = returns.notna() & prior_cap.notna() & (prior_cap > 0) & bp.notna() & (bp > 0)
        if valid.any():
            size_cut = prior_cap[valid].median()
            low_cut = bp[valid].quantile(0.30)
            high_cut = bp[valid].quantile(0.70)
            small, big = valid & (prior_cap <= size_cut), valid & (prior_cap > size_cut)
            low = valid & (bp <= low_cut)
            middle = valid & (bp > low_cut) & (bp <= high_cut)
            high = valid & (bp > high_cut)
            values = {
                "sl": _cap_weighted_return(returns, prior_cap, small & low),
                "sm": _cap_weighted_return(returns, prior_cap, small & middle),
                "sh": _cap_weighted_return(returns, prior_cap, small & high),
                "bl": _cap_weighted_return(returns, prior_cap, big & low),
                "bm": _cap_weighted_return(returns, prior_cap, big & middle),
                "bh": _cap_weighted_return(returns, prior_cap, big & high),
            }
            if np.isfinite(list(values.values())).all():
                smb[index] = np.mean([values["sl"], values["sm"], values["sh"]]) - np.mean([values["bl"], values["bm"], values["bh"]])
                hml[index] = np.mean([values["sh"], values["bh"]]) - np.mean([values["sl"], values["bl"]])
        done = index
        if done == 1 or done % progress_every == 0 or done == total:
            _progress(show_progress, "6/7 构造 SMB/HML", done, total, started_at, f"当前 {dates[index]:%Y-%m-%d}")
    return pd.DataFrame({"smb": smb, "hml": hml}, index=dates)


def _residual_volatility(stock_returns, factor_returns, show_progress, progress_every, started_at, target, target_position, target_total):
    y_matrix = stock_returns.to_numpy(dtype=float)
    x_matrix = factor_returns.to_numpy(dtype=float)
    result = np.full(y_matrix.shape[1], np.nan)
    stock_total = y_matrix.shape[1]
    stock_step = max(250, progress_every * 50)
    for column in range(stock_total):
        valid = np.isfinite(y_matrix[:, column]) & np.isfinite(x_matrix).all(axis=1)
        if valid.sum() > 4:
            design = np.column_stack([np.ones(valid.sum()), x_matrix[valid]])
            if np.linalg.matrix_rank(design) == design.shape[1]:
                try:
                    residual = y_matrix[valid, column] - design @ np.linalg.lstsq(design, y_matrix[valid, column], rcond=None)[0]
                    result[column] = np.std(residual, ddof=1) if len(residual) > 1 else np.nan
                except np.linalg.LinAlgError:
                    pass
        done = column + 1
        if done == 1 or done % stock_step == 0 or done == stock_total:
            _progress(show_progress, "7/7 回归残差波动率", done, stock_total, started_at, f"目标 {target_position}/{target_total}：{target:%Y-%m-%d}")
    return result


def calc_ff3_residual_volatility_nm(data, target_dates=None, as_of_date=None, n_months=36, trading_days_per_month=21, market_index="csi_all_share", show_progress=False, progress_every=20, domain_data=None):
    """计算 FF3 残差波动率。

    允许输入仅包含“月末交易日 + 目标日”的稀疏面板；公式与原版一致。
    """
    _resolve_data_window({"n_months": n_months, "trading_days_per_month": trading_days_per_month})
    if not isinstance(progress_every, int) or isinstance(progress_every, bool) or progress_every < 1:
        raise ValueError("progress_every 必须是正整数。")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    required = {"date", "instrument", "close", "total_market_cap", "pb"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{FACTOR_NAME} 缺少字段：{missing}。")
    if data.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError("data 存在重复 date + instrument。")
    if domain_data is None or not hasattr(domain_data, "get_domain"):
        raise TypeError(f"{FACTOR_NAME} 需要包含 market_daily 的 FactorDataBundle。")

    started_at = time.perf_counter()
    targets = _targets(data, target_dates, as_of_date)
    if targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    _progress(show_progress, "1/7 校验输入", 1, 1, started_at, f"原始 {len(data):,} 行，目标 {len(targets)} 个")

    source = data.loc[:, ["date", "instrument", "close", "total_market_cap", "pb"]].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    source["instrument"] = source["instrument"].astype(str)
    for column in ["close", "total_market_cap", "pb"]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    # 与原实现一致：无效或非正收盘价不得参与月收益计算。
    source["close"] = source["close"].where(source["close"] > 0)
    if as_of_date is not None:
        source = source.loc[source["date"] <= pd.Timestamp(as_of_date).normalize()].copy()
    _progress(show_progress, "2/7 清洗并截断", 1, 1, started_at, f"保留 {len(source):,} 行")

    market = domain_data.get_domain("market_daily").loc[:, ["date", "market_index", "market_close"]].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market = market.loc[market["market_index"].astype(str) == market_index, ["date", "market_close"]]
    market["market_close"] = pd.to_numeric(market["market_close"], errors="coerce").where(lambda x: x > 0)
    if as_of_date is not None:
        market = market.loc[market["date"] <= pd.Timestamp(as_of_date).normalize()]
    if market.empty or market["date"].isna().any() or market.duplicated("date", keep=False).any():
        raise ValueError(f"指数 {market_index!r} 的市场数据无效。")
    market["month"] = market["date"].dt.to_period("M")
    month_end = market.sort_values("date", kind="mergesort").groupby("month", sort=True, as_index=False).tail(1).sort_values("date", kind="mergesort")
    month_dates = pd.DatetimeIndex(month_end["date"])
    if len(month_dates) < 2:
        raise ValueError("市场月末数据不足。")
    target_state = source.loc[source["date"].isin(targets), ["date", "instrument"]].copy()
    monthly_source = source.loc[source["date"].isin(month_dates)].copy()
    if monthly_source.empty:
        raise ValueError("输入缺少市场月末对应的股票数据。")
    _progress(show_progress, "3/7 提取月末样本", 1, 1, started_at, f"月末 {len(month_dates)} 个，股票数据 {len(monthly_source):,} 行")

    instruments = pd.Index(monthly_source["instrument"].unique()).sort_values()
    stock_close = monthly_source.pivot(index="date", columns="instrument", values="close").reindex(index=month_dates, columns=instruments)
    stock_returns = stock_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    del stock_close
    _progress(show_progress, "4/7 构造月末收盘价宽表", 1, 1, started_at, f"{len(month_dates)} 月 × {len(instruments):,} 股")

    month_cap = monthly_source.pivot(index="date", columns="instrument", values="total_market_cap").reindex(index=month_dates, columns=instruments)
    month_pb = monthly_source.pivot(index="date", columns="instrument", values="pb").reindex(index=month_dates, columns=instruments)
    market_returns = pd.Series(month_end["market_close"].to_numpy(dtype=float), index=month_dates).pct_change(fill_method=None).rename("mkt")
    del monthly_source, source
    _progress(show_progress, "5/7 构造月末市值与 PB 宽表", 1, 1, started_at, f"{len(month_dates)} 月 × {len(instruments):,} 股")

    style_returns = _build_style_returns(stock_returns, month_cap, month_pb, show_progress, progress_every, started_at)
    factor_returns = pd.concat([market_returns, style_returns], axis=1)
    month_periods = pd.PeriodIndex(month_end["month"], freq="M")
    del month_cap, month_pb, style_returns, market

    cached, parts = {}, []
    target_total = len(targets)
    try:
        for target_position, target in enumerate(targets, start=1):
            completed_period = target.to_period("M") - 1
            end_position = month_periods.get_indexer([completed_period])[0]
            values = np.full(len(instruments), np.nan)
            if end_position >= 0:
                if end_position not in cached:
                    start_position = max(0, end_position - n_months + 1)
                    cached[end_position] = _residual_volatility(stock_returns.iloc[start_position:end_position + 1], factor_returns.iloc[start_position:end_position + 1], show_progress, progress_every, started_at, target, target_position, target_total)
                values = cached[end_position]
            target_instruments = pd.Index(target_state.loc[target_state["date"] == target, "instrument"].drop_duplicates())
            parts.append(pd.DataFrame({"date": target, "instrument": target_instruments, FACTOR_NAME: pd.Series(values, index=instruments).reindex(target_instruments).to_numpy()}))
        return pd.concat(parts, ignore_index=True).sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": FACTOR_NAME,
    "func": calc_ff3_residual_volatility_nm,
    "factor_type": "base",
    "candidate_instances": {"36m": {"n_months": 36, "trading_days_per_month": 21, "market_index": "csi_all_share"}, "60m": {"n_months": 60, "trading_days_per_month": 21, "market_index": "csi_all_share"}},
    "category": "risk",
    "direction": 0,
    "description": "个股月收益对市场、规模、价值三因子回归后的 N 月残差波动率。",
    "formula": "std(epsilon_i)，其中 r_i=alpha+b_m*MKT+b_s*SMB+b_h*HML+epsilon_i。",
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
    "output_schema": {"date": {}, "instrument": {}, FACTOR_NAME: {"dtype": "float64", "meaning": "三因子未解释收益的月度样本波动率。"}},
    "usage_notes": "输入可稀疏到月末交易日加目标日；公式不变。",
    "pit_notes": "SMB/HML 使用滞后月末市值和 PB；月内目标日仅使用上一个完整月及以前数据。",
    "version": "1.1.0",
}


FACTOR_INFO = """# N 月三因子残差波动率

先以市场、规模和价值风格收益解释个股月收益，再计算剩余残差的波动率。较高数值表示股票具有更强、且难以由常见共同风险解释的特质波动。

市场因子使用指定指数收益；SMB 与 HML 在内部以滞后月末市值和 BP 的 2×3 组合构造。常用实例为 36 月与 60 月；短历史股票只要回归数学上可定义便输出结果，但其估计稳定性较低。
"""
