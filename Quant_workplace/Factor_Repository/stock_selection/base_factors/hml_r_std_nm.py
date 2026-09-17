# -*- coding: utf-8 -*-
"""最高/最低价相对前收盘波动差因子。"""

import time

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_size_industry import (
    neutralize_size_industry,
)


OUTPUT_COLUMNS = ["date", "instrument", "hml_r_std_nm"]


def _resolve_hml_r_std_nm_data_window(resolved_params):
    """根据月份参数解析滚动窗口。"""
    n_months = resolved_params.get("n_months", 5)
    trading_days_per_month = resolved_params.get(
        "trading_days_per_month",
        21,
    )
    for name, value in {
        "n_months": n_months,
        "trading_days_per_month": trading_days_per_month,
    }.items():
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} 必须是正整数。")

    window = int(n_months) * int(trading_days_per_month)
    return {
        "lookback_trading_days": window - 1,
        "requires_target_date_data": True,
        "minimum_history_observations": window - 1,
        "preheating_required": True,
        "insufficient_window_behavior": (
            "目标日及以前的有效日内涨跌幅不足 min_ts_observations 时，"
            "该股票目标日因子值输出 NaN。"
        ),
    }


def _normalize_target_dates(data_dates, target_dates):
    available = pd.DatetimeIndex(
        data_dates.dropna().unique()
    ).normalize().sort_values()
    if target_dates is None:
        return available
    if isinstance(target_dates, (str, pd.Timestamp)):
        target_dates = [target_dates]
    else:
        try:
            target_dates = list(target_dates)
        except TypeError:
            target_dates = [target_dates]

    normalized = pd.DatetimeIndex(
        pd.to_datetime(target_dates, errors="raise")
    ).normalize().unique().sort_values()
    missing = normalized.difference(available)
    if not missing.empty:
        preview = [d.strftime("%Y-%m-%d") for d in missing[:5]]
        raise ValueError(
            "hml_r_std_nm 缺少目标日期原始数据："
            f"{preview}。"
        )
    return normalized


def _winsorize_quantile(series, lower, upper):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    finite = np.isfinite(values)
    result = pd.Series(
        np.nan,
        index=series.index,
        name=series.name,
        dtype=float,
    )
    valid = values.loc[finite]
    if valid.empty:
        return result
    low_value = valid.quantile(lower)
    high_value = valid.quantile(upper)
    result.loc[finite] = valid.clip(low_value, high_value)
    return result


def calc_hml_r_std_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=5,
    trading_days_per_month=21,
    min_ts_observations=80,
    winsor_lower=0.01,
    winsor_upper=0.99,
    neutralize_industry=True,
    min_cs_count=100,
    show_progress=False,
    progress_every=20,
):
    """计算市值行业中性化后的 hml_r_std_nm。

    原始定义：

    ``high_r_t = high_t / pre_close_t - 1``

    ``low_r_t = low_t / pre_close_t - 1``

    ``raw_t = StdSamp(high_r, L) - StdSamp(low_r, L)``

    其中 ``L = n_months * trading_days_per_month``。原始因子先在
    目标日截面按分位数缩尾，再对 log(总市值) 和行业哑变量回归，
    最终输出回归残差的样本标准差 Z-score。

    本函数只计算因子，不查询数据、不构造标签、不选股和不回测。
    """
    started_at = time.perf_counter()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if (
        not isinstance(progress_every, (int, np.integer))
        or isinstance(progress_every, (bool, np.bool_))
        or int(progress_every) <= 0
    ):
        raise ValueError("progress_every 必须是正整数。")

    window_info = _resolve_hml_r_std_nm_data_window(
        {
            "n_months": n_months,
            "trading_days_per_month": trading_days_per_month,
        }
    )
    window = window_info["lookback_trading_days"] + 1
    if (
        not isinstance(min_ts_observations, (int, np.integer))
        or isinstance(min_ts_observations, (bool, np.bool_))
        or not 2 <= int(min_ts_observations) <= window
    ):
        raise ValueError(
            "min_ts_observations 必须是2至滚动窗口长度之间的整数。"
        )
    if (
        not isinstance(min_cs_count, (int, np.integer))
        or isinstance(min_cs_count, (bool, np.bool_))
        or int(min_cs_count) <= 0
    ):
        raise ValueError("min_cs_count 必须是正整数。")
    if not isinstance(neutralize_industry, (bool, np.bool_)):
        raise TypeError("neutralize_industry 必须是 bool。")
    if not (
        np.isfinite(winsor_lower)
        and np.isfinite(winsor_upper)
        and 0 <= float(winsor_lower) < float(winsor_upper) <= 1
    ):
        raise ValueError(
            "winsor_lower 和 winsor_upper 必须满足 "
            "0 <= lower < upper <= 1。"
        )

    required = {
        "date",
        "instrument",
        "high",
        "low",
        "close",
        "pre_close",
        "total_market_cap",
    }
    if neutralize_industry:
        required.add("industry")
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"hml_r_std_nm 缺少字段：{sorted(missing)}"
        )

    keep_columns = list(required)
    df = data.loc[:, keep_columns].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("date 存在无法解析的日期或缺失值。")
    if df["instrument"].isna().any():
        raise ValueError("instrument 不允许缺失。")
    df["instrument"] = df["instrument"].astype(str)
    if df.duplicated(["date", "instrument"]).any():
        raise ValueError("data 存在重复的 date + instrument。")

    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date).normalize()
        df = df[df["date"] <= cutoff].copy()
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in [
        "high",
        "low",
        "close",
        "pre_close",
        "total_market_cap",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    target_index = _normalize_target_dates(df["date"], target_dates)
    if target_index.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.sort_values(
        ["instrument", "date"],
        kind="mergesort",
    ).reset_index(drop=True)
    shifted_close = df.groupby("instrument", sort=False)["close"].shift(1)
    effective_pre_close = df["pre_close"].where(
        np.isfinite(df["pre_close"]) & (df["pre_close"] > 0),
        shifted_close,
    )
    df["high_r"] = df["high"] / effective_pre_close - 1.0
    df["low_r"] = df["low"] / effective_pre_close - 1.0
    df.loc[~np.isfinite(df["high_r"]), "high_r"] = np.nan
    df.loc[~np.isfinite(df["low_r"]), "low_r"] = np.nan

    min_ts_observations = int(min_ts_observations)
    df["high_r_std"] = (
        df.groupby("instrument", sort=False)["high_r"]
        .rolling(
            window=window,
            min_periods=min_ts_observations,
        )
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
    df["low_r_std"] = (
        df.groupby("instrument", sort=False)["low_r"]
        .rolling(
            window=window,
            min_periods=min_ts_observations,
        )
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
    df["factor_raw"] = df["high_r_std"] - df["low_r_std"]

    target_data = df[df["date"].isin(target_index)].copy()
    grouped = target_data.groupby("date", sort=True)
    total_dates = len(grouped)
    result_parts = []

    try:
        for position, (date, cross_section) in enumerate(grouped, start=1):
            cross_section = cross_section.copy()
            factor_w = _winsorize_quantile(
                cross_section["factor_raw"],
                float(winsor_lower),
                float(winsor_upper),
            )
            industry = (
                cross_section["industry"]
                if neutralize_industry
                else None
            )
            factor = neutralize_size_industry(
                target=factor_w,
                market_cap=cross_section["total_market_cap"],
                industry=industry,
                min_obs=int(min_cs_count),
                standardize_residual=True,
                zscore_ddof=1,
                show_progress=False,
            )
            factor = factor.replace([np.inf, -np.inf], np.nan)
            result_parts.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "instrument": cross_section[
                            "instrument"
                        ].to_numpy(),
                        "hml_r_std_nm": factor.to_numpy(),
                    }
                )
            )

            refresh = (
                position == 1
                or position % int(progress_every) == 0
                or position == total_dates
            )
            if show_progress and refresh:
                elapsed = time.perf_counter() - started_at
                remaining = elapsed / position * (total_dates - position)
                print(
                    "\r[hml_r_std_nm] "
                    f"{position}/{total_dates} 个截面 "
                    f"| {position / total_dates:.1%} "
                    f"| 当前：{date:%Y-%m-%d} "
                    f"| 已耗时：{elapsed:.1f}s "
                    f"| 预计剩余：{remaining:.1f}s",
                    end="",
                    flush=True,
                )

        if not result_parts:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        return (
            pd.concat(result_parts, ignore_index=True)
            .sort_values(["date", "instrument"], kind="mergesort")
            .reset_index(drop=True)
        )
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": 'hml_r_std_nm',
    "func": calc_hml_r_std_nm,
    "factor_type": "base",
    "candidate_instances": {"5m": {"n_months": 5, "trading_days_per_month": 21}},
    "input_schema": {
        "required": {
            'date': {},
            'instrument': {},
            'high': {},
            'low': {},
            'close': {},
            'pre_close': {},
            'total_market_cap': {},
        },
        "conditional": {
            'industry': {"required_when": {'neutralize_industry': True}},
        },
    },
    "parameters": {
        'target_dates': {"default": None},
        'as_of_date': {"default": None},
        'n_months': {"default": 5},
        'trading_days_per_month': {"default": 21},
        'min_ts_observations': {"default": 80},
        'winsor_lower': {"default": 0.01},
        'winsor_upper': {"default": 0.99},
        'neutralize_industry': {"default": True},
        'min_cs_count': {"default": 100},
        'show_progress': {"default": False},
        'progress_every': {"default": 20},
    },
    "data_window": {
        "resolver": _resolve_hml_r_std_nm_data_window,
        "default": {
            "lookback_trading_days": 104,
            "requires_target_date_data": True,
            "minimum_history_observations": 104,
            "preheating_required": True,
        },
    },
    "output_schema": {
        'date': {},
        'instrument': {},
        'hml_r_std_nm': {},
    },
}


FACTOR_INFO = """
# 日内高低价区间波动（N 月）

以日内高低价区间收益的滚动标准差衡量波动程度，再做市值与可选行业中性化。数值较低通常代表更稳定的价格行为。

- **计算**：窗口由 `n_months × trading_days_per_month` 决定。
- **时点**：高、低、收盘和前收盘价均须采用目标日点时行情。
- **推荐实例**：`n_months=5`，对应原 5 个月版本。
"""
