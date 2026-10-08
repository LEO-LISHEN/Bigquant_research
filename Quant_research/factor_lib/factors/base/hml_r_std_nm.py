# -*- coding: utf-8 -*-
"""最高/最低价相对前收盘波动差因子。"""

import time

import numpy as np
import pandas as pd



# 本因子专用的预处理实现；保留旧数值规则和处理顺序。
def _zscore(
    series,
    ddof=0,
    show_progress=False,
):
    """对单个截面执行 Z-score 标准化。

    参数
    ----
    series : pandas.Series
        待标准化的截面数据。
    ddof : int，默认 0
        标准差的自由度。0 为总体标准差；1 为样本标准差。
        保留默认值 0，以兼容因子库中已有调用。
    show_progress : bool，默认 False
        是否显示一行处理状态；嵌套调用时应保持 False。

    返回
    ----
    pandas.Series
        与输入索引一致的标准化结果。有效样本不足或无波动时返回 NaN。
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series 必须是 pandas.Series。")
    if (
        not isinstance(ddof, (int, np.integer))
        or isinstance(ddof, (bool, np.bool_))
        or int(ddof) < 0
    ):
        raise ValueError("ddof 必须是非负整数。")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")

    ddof = int(ddof)
    values = pd.to_numeric(series, errors="coerce").astype(float)
    finite = np.isfinite(values)
    result = pd.Series(
        np.nan,
        index=series.index,
        name=series.name,
        dtype=float,
    )

    if show_progress:
        print(
            "\r[Z-score] 正在执行截面标准化...",
            end="",
            flush=True,
        )

    try:
        valid = values.loc[finite]
        if len(valid) <= ddof:
            return result

        std = valid.std(ddof=ddof)
        if not np.isfinite(std) or std <= 0:
            return result

        result.loc[finite] = (
            valid - valid.mean()
        ) / std
        return result
    finally:
        if show_progress:
            print(
                "\r[Z-score] 截面标准化完成。      "
            )


def _neutralize_ols(target, controls, min_obs=30):
    """
    对单个截面执行 OLS 中性化，返回回归残差。

    参数
    ----
    target : pandas.Series
        待中性化的因子暴露。
    controls : pandas.Series 或 pandas.DataFrame
        控制变量，例如 log(市值) 和行业哑变量。
    min_obs : int
        最小有效样本数。

    注意
    ----
    本函数只适用于单个交易日截面。
    多日面板必须先按 date 分组，再逐日调用。
    """
    if not isinstance(target, pd.Series):
        raise TypeError("target 必须是 pandas.Series")

    if isinstance(controls, pd.Series):
        controls = controls.to_frame()
    elif not isinstance(controls, pd.DataFrame):
        raise TypeError("controls 必须是 pandas.Series 或 pandas.DataFrame")

    controls = controls.reindex(target.index)
    controls = controls.astype(float)

    valid = target.notna() & controls.notna().all(axis=1)
    result = pd.Series(np.nan, index=target.index, name=target.name)

    # 与原 BP notebook 一致：有效样本不足时，退化为去均值。
    if valid.sum() < min_obs:
        return target - target.mean()

    x = np.column_stack(
        [
            np.ones(valid.sum()),
            controls.loc[valid].to_numpy(dtype=float),
        ]
    )
    y = target.loc[valid].to_numpy(dtype=float)

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    result.loc[valid] = y - x @ beta

    return result


def _neutralize_size_industry(
    target,
    market_cap,
    industry=None,
    min_obs=30,
    standardize_residual=True,
    zscore_ddof=0,
    show_progress=False,
):
    """对单个截面执行市值与可选行业中性化。

    回归形式为：

    ``target ~ intercept + log(market_cap) + industry_dummies``

    返回OLS残差；当 ``standardize_residual=True`` 时，再对残差执行
    Z-score。该函数只处理一个日期截面，多日面板应先按 date 分组。

    参数
    ----
    target : pandas.Series
        待中性化的因子暴露。
    market_cap : pandas.Series
        总市值或其他正值市值字段；函数内部取自然对数。
    industry : pandas.Series 或 None，默认 None
        行业分类。传入时加入行业哑变量；None 表示仅做市值中性化。
        行业缺失的记录不参与行业中性化。
    min_obs : int，默认 30
        最小有效截面样本数。
    standardize_residual : bool，默认 True
        是否对中性化残差继续执行Z-score。
    zscore_ddof : int，默认 0
        残差标准化采用的标准差自由度。
    show_progress : bool，默认 False
        是否显示一行处理状态；嵌套调用时应保持 False。

    返回
    ----
    pandas.Series
        与 target 索引一致的残差或标准化残差。
        样本不足以支持回归时返回全 NaN，不退化为简单去均值。
    """
    if not isinstance(target, pd.Series):
        raise TypeError("target 必须是 pandas.Series。")
    if not isinstance(market_cap, pd.Series):
        raise TypeError("market_cap 必须是 pandas.Series。")
    if industry is not None and not isinstance(industry, pd.Series):
        raise TypeError("industry 必须是 pandas.Series 或 None。")
    if (
        not isinstance(min_obs, (int, np.integer))
        or isinstance(min_obs, (bool, np.bool_))
        or int(min_obs) <= 0
    ):
        raise ValueError("min_obs 必须是正整数。")
    if not isinstance(
        standardize_residual,
        (bool, np.bool_),
    ):
        raise TypeError("standardize_residual 必须是 bool。")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")

    min_obs = int(min_obs)
    market_cap = market_cap.reindex(target.index)
    if industry is not None:
        industry = industry.reindex(target.index)

    y = pd.to_numeric(target, errors="coerce").astype(float)
    cap = pd.to_numeric(
        market_cap,
        errors="coerce",
    ).astype(float)
    log_market_cap = np.log(
        cap.where(cap > 0)
    ).rename("log_market_cap")

    valid = (
        np.isfinite(y)
        & np.isfinite(log_market_cap)
    )
    if industry is not None:
        valid &= industry.notna()

    result = pd.Series(
        np.nan,
        index=target.index,
        name=target.name,
        dtype=float,
    )

    if show_progress:
        mode = "市值+行业" if industry is not None else "市值"
        print(
            f"\r[{mode}中性化] 正在执行截面OLS...",
            end="",
            flush=True,
        )

    try:
        controls = log_market_cap.to_frame()

        if industry is not None and valid.any():
            valid_industry = (
                industry.loc[valid]
                .astype(str)
            )
            valid_dummies = pd.get_dummies(
                valid_industry,
                prefix="industry",
                drop_first=True,
                dtype=float,
            )
            industry_dummies = pd.DataFrame(
                0.0,
                index=target.index,
                columns=valid_dummies.columns,
            )
            industry_dummies.loc[
                valid_dummies.index,
                valid_dummies.columns,
            ] = valid_dummies
            controls = pd.concat(
                [controls, industry_dummies],
                axis=1,
            )

        # neutralize_ols 会再加入截距。除了最小样本数，还要求自由度
        # 至少大于2，避免行业较多时出现欠定回归。
        minimum_required = max(
            min_obs,
            controls.shape[1] + 4,
        )
        if int(valid.sum()) < minimum_required:
            return result

        residual = _neutralize_ols(
            target=y.where(valid),
            controls=controls,
            min_obs=min_obs,
        )
        residual = residual.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if standardize_residual:
            residual = _zscore(
                residual,
                ddof=zscore_ddof,
                show_progress=False,
            )

        residual.name = target.name
        return residual
    finally:
        if show_progress:
            print(
                "\r[市值行业中性化] 截面处理完成。      "
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
            factor = _neutralize_size_industry(
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
