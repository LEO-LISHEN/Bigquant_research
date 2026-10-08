# -*- coding: utf-8 -*-
"""单市场因子模型特质波动率。"""

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


OUTPUT_COLUMNS = ["date", "instrument", "id1_std_nm"]


def _resolve_id1_std_nm_data_window(resolved_params):
    """根据月份参数解析价格和回归窗口。"""
    n_months = resolved_params.get("n_months", 3)
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
        "lookback_trading_days": window,
        "requires_target_date_data": True,
        "minimum_history_observations": window,
        "preheating_required": True,
        "insufficient_window_behavior": (
            "回归窗口内有效股票收益与市场收益配对数不足"
            "min_ts_observations时，该股票目标日因子值输出NaN。"
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
            f"id1_std_nm 缺少目标日期原始数据：{preview}。"
        )
    return normalized


def _get_market_close(domain_data, market_index, as_of_date=None):
    """从分域容器读取指定市场指数的原始收盘点位。"""
    if domain_data is None or not hasattr(domain_data, "get_domain"):
        raise TypeError(
            "id1_std_nm 需要包含 market_daily 的 FactorDataBundle；"
            "请通过 loader 和 get_factor 调用。"
        )
    if not isinstance(market_index, str) or not market_index.strip():
        raise ValueError("market_index 必须是非空统一指数名称。")

    market_data = domain_data.get_domain("market_daily").copy()
    required = {"date", "market_index", "market_close"}
    missing = required - set(market_data.columns)
    if missing:
        raise ValueError(
            f"market_daily 缺少字段：{sorted(missing)}"
        )

    market_data["date"] = pd.to_datetime(
        market_data["date"], errors="coerce"
    ).dt.normalize()
    if market_data["date"].isna().any():
        raise ValueError("market_daily.date 包含无效日期。")
    market_data["market_index"] = market_data[
        "market_index"
    ].astype(str)
    market_data = market_data.loc[
        market_data["market_index"] == market_index.strip()
    ].copy()
    if as_of_date is not None:
        market_data = market_data.loc[
            market_data["date"] <= pd.Timestamp(as_of_date).normalize()
        ].copy()
    if market_data.empty:
        raise ValueError(
            f"market_daily 中没有指数 {market_index!r} 的数据。"
        )
    if market_data.duplicated(["date", "market_index"]).any():
        raise ValueError(
            "market_daily 存在重复的 date + market_index。"
        )

    market_close = pd.to_numeric(
        market_data.set_index("date")["market_close"],
        errors="coerce",
    ).sort_index()
    return market_close.where(
        np.isfinite(market_close) & (market_close > 0)
    )


def _capm_residual_standard_error(y_matrix, market_return, min_obs):
    """按Notebook的SSE/(n-2)公式批量计算CAPM残差标准误。"""
    y = np.asarray(y_matrix, dtype=float)
    x = np.asarray(market_return, dtype=float).reshape(-1, 1)
    valid = np.isfinite(y) & np.isfinite(x)

    n = valid.sum(axis=0).astype(float)
    x_full = np.broadcast_to(x, y.shape)
    safe_x = np.where(valid, x_full, 0.0)
    safe_y = np.where(valid, y, 0.0)

    sum_x = safe_x.sum(axis=0)
    sum_y = safe_y.sum(axis=0)
    sum_xx = (safe_x * safe_x).sum(axis=0)
    sum_xy = (safe_x * safe_y).sum(axis=0)
    sum_yy = (safe_y * safe_y).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = sum_xx - sum_x * sum_x / n
        beta = (sum_xy - sum_x * sum_y / n) / denom
        alpha = sum_y / n - beta * sum_x / n
        sse = (
            sum_yy
            + n * alpha * alpha
            + beta * beta * sum_xx
            - 2.0 * alpha * sum_y
            - 2.0 * beta * sum_xy
            + 2.0 * alpha * beta * sum_x
        )
        residual_variance = sse / (n - 2.0)

    tolerance = 1e-14
    residual_variance = np.where(
        (residual_variance < 0)
        & (residual_variance > -tolerance),
        0.0,
        residual_variance,
    )
    usable = (
        (n >= int(min_obs))
        & (n > 2)
        & np.isfinite(denom)
        & (denom > 0)
        & np.isfinite(residual_variance)
        & (residual_variance >= 0)
    )
    result = np.full(y.shape[1], np.nan, dtype=float)
    result[usable] = np.sqrt(residual_variance[usable])
    return result


def calc_id1_std_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=3,
    trading_days_per_month=21,
    market_index="csi_all_share",
    min_ts_observations=50,
    neutralize_industry=True,
    min_cs_count=80,
    standardize_residual=True,
    show_progress=False,
    progress_every=20,
    domain_data=None,
):
    """计算市值行业中性化后的 id1_std_nm。

    对每只股票最近 ``L`` 个交易日的日收益进行滚动CAPM回归：

    ``stock_return = alpha + beta * market_return + epsilon``

    原始因子为 ``sqrt(SSE / (n - 2))``。随后在每个目标日截面
    对 log(总市值) 和行业哑变量回归，输出中性化残差；默认继续
    对截面残差进行Z-score标准化。

    股票面板由 ``data`` 提供；市场指数面板由 ``domain_data`` 提供。
    函数内部根据 ``market_index`` 选取市场原始收盘点位并计算日收益率。
    精确复现原 Notebook 时使用中证全指。
    """
    started_at = time.perf_counter()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if not isinstance(neutralize_industry, (bool, np.bool_)):
        raise TypeError("neutralize_industry 必须是 bool。")
    if not isinstance(standardize_residual, (bool, np.bool_)):
        raise TypeError("standardize_residual 必须是 bool。")
    for name, value in {
        "progress_every": progress_every,
        "min_cs_count": min_cs_count,
    }.items():
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} 必须是正整数。")

    window_info = _resolve_id1_std_nm_data_window(
        {
            "n_months": n_months,
            "trading_days_per_month": trading_days_per_month,
        }
    )
    window = window_info["lookback_trading_days"]
    if (
        not isinstance(min_ts_observations, (int, np.integer))
        or isinstance(min_ts_observations, (bool, np.bool_))
        or not 3 <= int(min_ts_observations) <= window
    ):
        raise ValueError(
            "min_ts_observations 必须是3至回归窗口长度之间的整数。"
        )

    required = {
        "date",
        "instrument",
        "close",
        "volume",
        "amount",
        "suspended",
        "total_market_cap",
    }
    if neutralize_industry:
        required.add("industry")
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"id1_std_nm 缺少字段：{sorted(missing)}")

    df = data.loc[:, list(required)].copy()
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
        "close",
        "volume",
        "amount",
        "suspended",
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
    df["stock_return"] = (
        df.groupby("instrument", sort=False)["close"]
        .pct_change(fill_method=None)
    )
    valid_trade = (
        (df["suspended"] == 0)
        & (df["volume"] > 0)
        & (df["amount"] > 0)
        & np.isfinite(df["stock_return"])
    )
    df.loc[~valid_trade, "stock_return"] = np.nan

    all_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    all_instruments = pd.Index(
        df["instrument"].astype(str).unique()
    ).sort_values()
    return_wide = (
        df.pivot(index="date", columns="instrument", values="stock_return")
        .reindex(index=all_dates, columns=all_instruments)
    )
    market_close = _get_market_close(
        domain_data,
        market_index,
        as_of_date=as_of_date,
    ).reindex(all_dates)
    missing_market_dates = market_close.index[market_close.isna()]
    if len(missing_market_dates) > 0:
        preview = [
            date.strftime("%Y-%m-%d")
            for date in missing_market_dates[:5]
        ]
        raise ValueError(
            f"指数 {market_index!r} 缺少或存在无效收盘价的日期："
            f"{preview}。"
        )
    market_return = (
        market_close.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .rename("market_return")
    )

    target_state = df[df["date"].isin(target_index)].copy()
    target_state["instrument"] = target_state["instrument"].astype(str)
    result_parts = []
    total_dates = len(target_index)

    try:
        for position, date in enumerate(target_index, start=1):
            date_position = all_dates.get_indexer([date])[0]
            first_position = date_position - window + 1
            raw_values = np.full(len(all_instruments), np.nan, dtype=float)
            if first_position > 0:
                window_dates = all_dates[first_position:date_position + 1]
                y_matrix = return_wide.loc[window_dates].to_numpy(dtype=float)
                x_vector = market_return.loc[window_dates].to_numpy(dtype=float)
                raw_values = _capm_residual_standard_error(
                    y_matrix,
                    x_vector,
                    int(min_ts_observations),
                )

            raw_factor = pd.Series(
                raw_values,
                index=all_instruments,
                name="factor_raw",
            )
            cross_section = (
                target_state[target_state["date"] == date]
                .drop_duplicates("instrument", keep="last")
                .set_index("instrument")
                .reindex(all_instruments)
            )
            industry = (
                cross_section["industry"]
                if neutralize_industry
                else None
            )
            factor = _neutralize_size_industry(
                target=raw_factor,
                market_cap=cross_section["total_market_cap"],
                industry=industry,
                min_obs=int(min_cs_count),
                standardize_residual=standardize_residual,
                zscore_ddof=1,
                show_progress=False,
            ).replace([np.inf, -np.inf], np.nan)
            result_parts.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "instrument": all_instruments.to_numpy(),
                        "id1_std_nm": factor.to_numpy(),
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
                    "\r[id1_std_nm] "
                    f"{position}/{total_dates} 个截面 "
                    f"| {position / total_dates:.1%} "
                    f"| 当前：{date:%Y-%m-%d} "
                    f"| 已耗时：{elapsed:.1f}s "
                    f"| 预计剩余：{remaining:.1f}s",
                    end="",
                    flush=True,
                )

        return (
            pd.concat(result_parts, ignore_index=True)
            .sort_values(["date", "instrument"], kind="mergesort")
            .reset_index(drop=True)
        )
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": 'id1_std_nm',
    "func": calc_id1_std_nm,
    "factor_type": "base",
    "candidate_instances": {"3m": {"n_months": 3, "trading_days_per_month": 21}},
    "input_schema": {
        "required": {
            'date': {},
            'instrument': {},
            'close': {},
            'volume': {},
            'amount': {},
            'suspended': {},
            'total_market_cap': {},
            'market_close': {},
        },
        "conditional": {
            'industry': {"required_when": {'neutralize_industry': True}},
        },
    },
    "parameters": {
        'target_dates': {"default": None},
        'as_of_date': {"default": None},
        'n_months': {"default": 3},
        'trading_days_per_month': {"default": 21},
        'market_index': {"default": 'csi_all_share'},
        'min_ts_observations': {"default": 50},
        'neutralize_industry': {"default": True},
        'min_cs_count': {"default": 80},
        'standardize_residual': {"default": True},
        'show_progress': {"default": False},
        'progress_every': {"default": 20},
    },
    "data_window": {
        "resolver": _resolve_id1_std_nm_data_window,
        "default": {
            "lookback_trading_days": 63,
            "requires_target_date_data": True,
            "minimum_history_observations": 63,
            "preheating_required": True,
        },
    },
    "output_schema": {
        'date': {},
        'instrument': {},
        'id1_std_nm': {},
    },
}


FACTOR_INFO = """
# 单因子特质波动率（N 月）

以个股收益对市场收益进行滚动 CAPM 回归，将残差波动率作为原始因子，再进行市值与可选行业中性化。数值较低通常代表较低的特质风险。

- **计算**：市场指数由 `market_index` 参数指定，窗口随 `n_months` 变化。
- **时点**：股票行情、市场指数、停牌状态、市值和行业均须按历史日期对齐。
- **推荐实例**：`n_months=3`、`market_index='csi_all_share'`。
"""
