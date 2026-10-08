# -*- coding: utf-8 -*-
"""QFA_ROE：市值、行业中性化后的单季度平均ROE质量因子。"""

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


def _winsorize_mad(series, k=5.0):
    """
    基于中位数 ± k × MAD 的截面去极值。
    不填补缺失值，由调用因子决定如何处理。
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series 必须是 pandas.Series")

    median = series.median()
    mad = (series - median).abs().median()

    if pd.isna(mad) or mad == 0:
        return series.copy()

    lower = median - k * mad
    upper = median + k * mad
    return series.clip(lower=lower, upper=upper)


OUTPUT_COLUMNS = ["date", "instrument", "qfa_roe"]


def _normalize_target_dates(available_dates, target_dates):
    available = pd.DatetimeIndex(
        pd.to_datetime(available_dates, errors="raise")
    ).normalize().unique().sort_values()

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
        preview = [date.strftime("%Y-%m-%d") for date in missing[:5]]
        raise ValueError(
            "qfa_roe 缺少目标日财务截面："
            f"{preview}。请检查预存日期和 as_of_date。"
        )
    return normalized


def _robust_zscore(series, winsor_k):
    """复现notebook：MAD去极值；MAD退化时改用1%/99%分位；再做总体Z-score。"""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    valid_values = values.dropna()

    result = pd.Series(np.nan, index=series.index, dtype=float)
    if len(valid_values) < 3:
        return result

    median = valid_values.median()
    mad = (valid_values - median).abs().median()
    if pd.notna(mad) and mad > 1e-12:
        processed = _winsorize_mad(values, k=float(winsor_k))
    else:
        lower, upper = valid_values.quantile([0.01, 0.99])
        processed = values.clip(lower=lower, upper=upper)

    return _zscore(processed, ddof=0, show_progress=False).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _prepare_industry_labels(industry, min_industry_count):
    labels = (
        industry.fillna("UNKNOWN")
        .astype(str)
        .replace({"": "UNKNOWN", "None": "UNKNOWN", "nan": "UNKNOWN"})
    )
    counts = labels.value_counts(dropna=False)
    small_industries = counts[counts < int(min_industry_count)].index
    return labels.where(~labels.isin(small_industries), "OTHER")


def calc_qfa_roe(
    data,
    target_dates=None,
    as_of_date=None,
    neutralize_industry=True,
    winsor_k=5.0,
    min_cs_count=30,
    min_industry_count=10,
    show_progress=False,
    progress_every=20,
):
    """计算市值、行业中性化后的QFA_ROE因子。

    原始定义为截至目标日最新可得的单季度平均净资产收益率：

    ``quarterly_roe_avg``
    ``→ MAD去极值 + 总体Z-score``
    ``→ 对log(流通市值)和行业哑变量做截面OLS``
    ``→ OLS残差再次MAD去极值 + 总体Z-score``

    参数
    ----
    data : pandas.DataFrame
        必须包含date、instrument、quarterly_roe_avg、
        float_market_cap。当neutralize_industry=True时还必须包含industry。
        财务字段必须是目标日已经可得的点时数据。
    target_dates : 日期或日期序列，可选
        实际输出截面；None表示输出data内全部日期。
    as_of_date : 日期，可选
        全局信息截止日，晚于该日的数据不参与计算。
    neutralize_industry : bool，默认True
        True为流通市值+行业中性化；False为仅流通市值中性化。
    winsor_k : float，默认5.0
        原始因子及中性化残差的MAD去极值倍数。
    min_cs_count : int，默认30
        单日中性化所需最少有效样本数。
    min_industry_count : int，默认10
        单日样本数少于该值的行业合并为OTHER，以降低稀疏哑变量风险。
    show_progress : bool，默认False
        是否使用终端单行刷新显示计算进度。
    progress_every : int，默认20
        每处理多少个目标截面刷新一次进度。

    返回
    ----
    pandas.DataFrame
        date、instrument、qfa_roe三列。缺失原始财务值、市值无效或
        中性化样本不足时保留NaN，不填充为0；因子值越高代表质量越高。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    if not isinstance(neutralize_industry, (bool, np.bool_)):
        raise TypeError("neutralize_industry 必须是 bool。")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if (
        not isinstance(progress_every, (int, np.integer))
        or isinstance(progress_every, (bool, np.bool_))
        or int(progress_every) <= 0
    ):
        raise ValueError("progress_every 必须是正整数。")
    if (
        not isinstance(min_cs_count, (int, np.integer))
        or isinstance(min_cs_count, (bool, np.bool_))
        or int(min_cs_count) < 3
    ):
        raise ValueError("min_cs_count 必须是大于等于3的整数。")
    if (
        not isinstance(min_industry_count, (int, np.integer))
        or isinstance(min_industry_count, (bool, np.bool_))
        or int(min_industry_count) <= 0
    ):
        raise ValueError("min_industry_count 必须是正整数。")

    winsor_k = float(winsor_k)
    if not np.isfinite(winsor_k) or winsor_k <= 0:
        raise ValueError("winsor_k 必须是有限正数。")

    required_columns = {
        "date",
        "instrument",
        "quarterly_roe_avg",
        "float_market_cap",
    }
    if neutralize_industry:
        required_columns.add("industry")
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "qfa_roe 因子缺少字段："
            f"{sorted(missing_columns)}"
        )

    selected_columns = [
        "date",
        "instrument",
        "quarterly_roe_avg",
        "float_market_cap",
    ]
    if neutralize_industry:
        selected_columns.append("industry")
    df = data.loc[:, selected_columns].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("qfa_roe 输入存在无效date。")
    if df["instrument"].isna().any():
        raise ValueError("qfa_roe 的instrument不允许缺失。")

    duplicated = df.duplicated(["date", "instrument"], keep=False)
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["date", "instrument"]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            "qfa_roe 输入存在重复date + instrument："
            f"{examples}"
        )

    for column in ["quarterly_roe_avg", "float_market_cap"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )

    if as_of_date is not None:
        as_of_timestamp = pd.Timestamp(as_of_date)
        if pd.isna(as_of_timestamp):
            raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= as_of_timestamp.normalize()].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    target_date_index = _normalize_target_dates(df["date"], target_dates)
    if target_date_index.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    target_panel = df.loc[df["date"].isin(target_date_index)].copy()
    grouped_dates = list(target_panel.groupby("date", sort=True))
    total_dates = len(grouped_dates)
    result_parts = []
    started_at = time.perf_counter()

    if show_progress:
        print(
            f"\r[qfa_roe] 0/{total_dates}个截面 | 0.0%",
            end="",
            flush=True,
        )

    try:
        for position, (date, section) in enumerate(grouped_dates, start=1):
            section = section.copy()
            raw_z = _robust_zscore(
                section["quarterly_roe_avg"],
                winsor_k=winsor_k,
            )

            industry = None
            if neutralize_industry:
                industry = _prepare_industry_labels(
                    section["industry"],
                    min_industry_count=min_industry_count,
                )

                # 原notebook要求有效样本数必须大于设计矩阵列数+5；
                # 否则提前回退为仅流通市值中性化。
                valid_for_neutralization = (
                    raw_z.notna()
                    & section["float_market_cap"].notna()
                    & (section["float_market_cap"] > 0)
                    & industry.notna()
                )
                valid_industry = industry.loc[valid_for_neutralization]
                industry_dummy_count = max(
                    int(valid_industry.nunique(dropna=True)) - 1,
                    0,
                )
                design_column_count = 2 + industry_dummy_count
                if (
                    int(valid_for_neutralization.sum())
                    <= design_column_count + 5
                ):
                    industry = None

            residual = _neutralize_size_industry(
                target=raw_z,
                market_cap=section["float_market_cap"],
                industry=industry,
                min_obs=int(min_cs_count),
                standardize_residual=False,
                zscore_ddof=0,
                show_progress=False,
            )

            # 复现notebook的自由度保护：行业哑变量过多导致回归不可用时，
            # 回退为仅做流通市值中性化，而不是整日丢弃。
            fallback_to_size_only = (
                industry is not None
                and int(raw_z.notna().sum()) >= int(min_cs_count)
                and int(residual.notna().sum()) == 0
            )
            if fallback_to_size_only:
                residual = _neutralize_size_industry(
                    target=raw_z,
                    market_cap=section["float_market_cap"],
                    industry=None,
                    min_obs=int(min_cs_count),
                    standardize_residual=False,
                    zscore_ddof=0,
                    show_progress=False,
                )

            factor = _robust_zscore(residual, winsor_k=winsor_k)
            result_parts.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "instrument": section["instrument"].to_numpy(),
                        "qfa_roe": factor.to_numpy(),
                    }
                )
            )

            should_refresh = (
                position == 1
                or position % int(progress_every) == 0
                or position == total_dates
            )
            if show_progress and should_refresh:
                elapsed = time.perf_counter() - started_at
                remaining = elapsed / position * (total_dates - position)
                print(
                    "\r"
                    f"[qfa_roe] {position}/{total_dates}个截面 "
                    f"| {position / total_dates:.1%} "
                    f"| 当前：{date:%Y-%m-%d} "
                    f"| 原始有效：{raw_z.notna().sum():,} "
                    f"| 输出有效：{factor.notna().sum():,} "
                    f"| 已耗时：{elapsed:.1f}s "
                    f"| 预计剩余：{remaining:.1f}s",
                    end="",
                    flush=True,
                )
    finally:
        if show_progress:
            print()

    if not result_parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return (
        pd.concat(result_parts, ignore_index=True)
        .sort_values(["date", "instrument"], kind="mergesort")
        .reset_index(drop=True)
    )


FACTOR = {
    "name": 'qfa_roe',
    "func": calc_qfa_roe,
    "factor_type": "base",
    "candidate_instances": {"default": {}},
    "input_schema": {
        "required": {
            'date': {},
            'instrument': {},
            'quarterly_roe_avg': {},
            'float_market_cap': {},
        },
        "conditional": {
            'industry': {"required_when": {'neutralize_industry': True}},
        },
    },
    "parameters": {
        'target_dates': {"default": None},
        'as_of_date': {"default": None},
        'neutralize_industry': {"default": True},
        'winsor_k': {"default": 5.0},
        'min_cs_count': {"default": 30},
        'min_industry_count': {"default": 10},
        'show_progress': {"default": False},
        'progress_every': {"default": 20},
    },
    "data_window": {
        "lookback_trading_days": 0,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": False,
    },
    "output_schema": {
        'date': {},
        'instrument': {},
        'qfa_roe': {},
    },
}


FACTOR_INFO = """
# QFA_ROE（单季平均净资产收益率）

使用目标日可得的单季平均 ROE，经市值中性化、可选行业中性化、去极值和标准化后形成质量暴露。数值越高通常代表盈利质量更强。

- **计算**：以点时财务字段 `quarterly_roe_avg` 为原始输入。
- **时点**：财务字段必须按真实披露可得时间对齐。
- **研究提示**：ROE 可能受杠杆和一次性损益影响，宜配合利润与现金流指标判断。
"""

