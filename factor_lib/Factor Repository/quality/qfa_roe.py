# -*- coding: utf-8 -*-
"""QFA_ROE：市值、行业中性化后的单季度平均ROE质量因子。"""

import time

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_size_industry import (
    neutralize_size_industry,
)
from factor_lib.common.preprocess.winsorize_mad import winsorize_mad
from factor_lib.common.preprocess.zscore import zscore


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
        processed = winsorize_mad(values, k=float(winsor_k))
    else:
        lower, upper = valid_values.quantile([0.01, 0.99])
        processed = values.clip(lower=lower, upper=upper)

    return zscore(processed, ddof=0, show_progress=False).replace(
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

    ``quarterly_average_roe``
    ``→ MAD去极值 + 总体Z-score``
    ``→ 对log(流通市值)和行业哑变量做截面OLS``
    ``→ OLS残差再次MAD去极值 + 总体Z-score``

    参数
    ----
    data : pandas.DataFrame
        必须包含date、instrument、quarterly_average_roe、
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
        "quarterly_average_roe",
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
        "quarterly_average_roe",
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

    for column in ["quarterly_average_roe", "float_market_cap"]:
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
                section["quarterly_average_roe"],
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

            residual = neutralize_size_industry(
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
                residual = neutralize_size_industry(
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
    "name": "qfa_roe",
    "func": calc_qfa_roe,
    "category": "quality",
    "direction": 1,
    "description": (
        "单季度平均净资产收益率质量因子，经截面稳健标准化、"
        "流通市值和可选行业中性化，并对残差再次稳健标准化；"
        "数值越高代表盈利质量越高。"
    ),
    "formula": (
        "raw_t=quarterly_average_roe_t="
        "quarterly_parent_net_profit_t/average_parent_equity_t；"
        "raw先执行median±winsor_k×MAD去极值和总体Z-score；"
        "随后对log(float_market_cap)和可选行业哑变量做截面OLS；"
        "最终因子为OLS残差再次MAD去极值后的总体Z-score。"
    ),
    "input_schema": {
        "required": {
            "date": {
                "dtype": "datetime64[ns]或可解析日期",
                "meaning": "点时财务因子截面日期。",
            },
            "instrument": {
                "dtype": "string",
                "meaning": "证券唯一标识；同一date+instrument不允许重复。",
            },
            "quarterly_average_roe": {
                "dtype": "float",
                "meaning": (
                    "截至目标日最新可得的单季度平均净资产收益率；"
                    "对应BigQuant的roe_avg_mrq口径，但字段名保持数据源无关。"
                ),
            },
            "float_market_cap": {
                "dtype": "float",
                "meaning": "目标日流通市值；取自然对数后用于中性化。",
            },
        },
        "conditional": {
            "industry": {
                "dtype": "string",
                "meaning": "目标日点时一级行业分类。",
                "required_when": {"neutralize_industry": True},
            },
        },
    },
    "parameters": {
        "target_dates": {
            "default": None,
            "accepted_values": "单个日期、日期序列或None。",
            "effect": "指定实际输出的因子截面。",
            "changes_data_requirements": True,
        },
        "as_of_date": {
            "default": None,
            "accepted_values": "可解析日期或None。",
            "effect": "全局信息截止日，晚于该日的数据不参与计算。",
            "changes_data_requirements": False,
        },
        "neutralize_industry": {
            "default": True,
            "accepted_values": [True, False],
            "effect": "控制是否在流通市值之外加入行业哑变量。",
            "changes_data_requirements": True,
        },
        "winsor_k": {
            "default": 5.0,
            "accepted_values": "有限正数。",
            "effect": "控制原始因子和中性化残差的MAD去极值边界。",
            "changes_data_requirements": False,
        },
        "min_cs_count": {
            "default": 30,
            "accepted_values": "大于等于3的整数。",
            "effect": "控制单日截面中性化的最低有效样本数。",
            "changes_data_requirements": False,
        },
        "min_industry_count": {
            "default": 10,
            "accepted_values": "正整数。",
            "effect": "样本数不足的行业合并为OTHER，降低哑变量稀疏度。",
            "changes_data_requirements": False,
        },
        "show_progress": {
            "default": False,
            "accepted_values": [True, False],
            "effect": "仅控制进度显示。",
            "changes_data_requirements": False,
        },
        "progress_every": {
            "default": 20,
            "accepted_values": "正整数。",
            "effect": "进度刷新间隔，单位为目标截面数。",
            "changes_data_requirements": False,
        },
    },
    "data_window": {
        "lookback_trading_days": 0,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": False,
        "insufficient_window_behavior": (
            "直接使用目标日已经点时对齐的单季度平均ROE；"
            "目标日整体缺失时报错，个股缺失或截面不足时保留NaN。"
        ),
    },
    "output_schema": {
        "date": {
            "dtype": "datetime64[ns]",
            "meaning": "目标因子截面日期。",
        },
        "instrument": {
            "dtype": "string",
            "meaning": "证券唯一标识。",
        },
        "qfa_roe": {
            "dtype": "float64",
            "meaning": "市值行业中性化后的质量因子；数值越高越优。",
        },
    },
    "usage_notes": [
        "默认使用流通市值而不是总市值，以保持原notebook中性化口径。",
        "小行业会按目标日截面合并为OTHER。",
        "行业哑变量导致自由度不足时，按原notebook逻辑回退为仅流通市值中性化。",
        "研究和策略层应剔除NaN因子值，不应填充为0。",
    ],
    "pit_notes": [
        "quarterly_average_roe必须是目标日已经可得的点时财务指标。",
        "不能按报告期结束日提前回填尚未公告的财报。",
        "行业分类和流通市值也必须使用目标日点时值。",
        (
            "若数据源以最新更正后的财报覆盖历史版本，"
            "仍可能存在历史修订数据偏差。"
        ),
    ],
    "references": [
        (
            "研究稿/华泰因子复现/华泰财务质量因子/"
            "qfa_roe因子/qfa_roe.ipynb"
        ),
    ],
    "tags": [
        "quality",
        "roe",
        "quarterly",
        "neutralized",
        "cross_sectional",
    ],
    "status": "research",
    "version": "1.0.0",
}
