# -*- coding: utf-8 -*-
"""华泰 N 日累计主力流出额因子。"""

import time

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_size_industry import (
    neutralize_size_industry,
)


OUTPUT_COLUMNS = ["date", "instrument", "mfd_sellamt_nd"]


def _resolve_mfd_sellamt_nd_data_window(resolved_params):
    """根据 n_days 解析 N 日累计值需要的历史窗口。"""
    n_days = resolved_params.get("n_days", 1)
    if (
        not isinstance(n_days, (int, np.integer))
        or isinstance(n_days, (bool, np.bool_))
        or int(n_days) <= 0
    ):
        raise ValueError("n_days 必须是正整数。")

    n_days = int(n_days)
    lookback_days = n_days - 1
    return {
        "lookback_trading_days": lookback_days,
        "requires_target_date_data": True,
        "minimum_history_observations": lookback_days,
        "preheating_required": lookback_days > 0,
        "insufficient_window_behavior": (
            "目标日及以前的 N 日窗口内，有效主力流出额观测数低于 "
            "min_observations 时，该股票目标日因子值输出 NaN。"
        ),
    }


def _normalize_target_dates(data_dates, target_dates):
    """规范目标日期，并确认目标日存在于股票面板。"""
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
        preview = [date.strftime("%Y-%m-%d") for date in missing[:5]]
        raise ValueError(
            "mfd_sellamt_nd 缺少目标日期原始数据："
            f"{preview}。请检查预存日期和 as_of_date。"
        )
    return normalized


def _robust_zscore(series, winsor_k):
    """复现旧 notebook 的 scaled-MAD 去极值和总体 Z-score。"""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()

    result = pd.Series(np.nan, index=series.index, dtype=float)
    if len(valid) < 3:
        return result

    median = valid.median()
    mad = (valid - median).abs().median()
    if pd.notna(mad) and mad > 1e-12:
        robust_scale = 1.4826 * mad
        lower = median - float(winsor_k) * robust_scale
        upper = median + float(winsor_k) * robust_scale
    else:
        lower, upper = valid.quantile([0.01, 0.99])

    clipped = values.clip(lower=lower, upper=upper)
    clipped_valid = clipped.dropna()
    standard_deviation = clipped_valid.std(ddof=0)
    if (
        not np.isfinite(standard_deviation)
        or standard_deviation <= 1e-12
    ):
        return result

    result.loc[clipped_valid.index] = (
        clipped_valid - clipped_valid.mean()
    ) / standard_deviation
    return result


def calc_mfd_sellamt_nd(
    data,
    target_dates=None,
    as_of_date=None,
    n_days=1,
    min_observations=None,
    neutralize_industry=True,
    winsor_k=3.0,
    min_cs_count=30,
    show_progress=False,
    progress_every=20,
):
    """计算市值、行业中性化后的 N 日累计主力流出额因子。

    时间序列原始值为：

    ``rolling_outflow_t = Sum(main_outflow_amount, N days)``
    ``raw_t = -rolling_outflow_t``

    随后逐目标日执行：

    ``scaled-MAD去极值 + 总体Z-score``
    ``→ 对log(流通市值)和可选行业哑变量做截面OLS``
    ``→ OLS残差再次scaled-MAD去极值 + 总体Z-score``

    负号复现旧 notebook 的反向化处理，因此最终数值越高，代表
    N 日累计主力流出额越低，并按旧研究方向视为越优。

    参数
    ----
    data : pandas.DataFrame
        必须包含 date、instrument、main_outflow_amount、
        float_market_cap；neutralize_industry=True 时还必须包含
        industry。data 应覆盖目标日及其 N-1 个历史预热交易日。
    target_dates : 日期或日期序列，可选
        实际输出截面；None 表示输出 data 中的全部日期。
    as_of_date : 日期，可选
        全局信息截止日，晚于该日的数据不会参与计算。
    n_days : int，默认 1
        主力流出额累计窗口，包含目标日。
    min_observations : int 或 None，默认 None
        N 日窗口内最少有效资金流观测数；None 表示严格要求 N 个。
    neutralize_industry : bool，默认 True
        True 为流通市值+行业中性化；False 为仅流通市值中性化。
    winsor_k : float，默认 3.0
        scaled-MAD 去极值倍数；稳健尺度为 1.4826×MAD。
    min_cs_count : int，默认 30
        单日截面中性化所需的最少有效股票数。
    show_progress : bool，默认 False
        是否使用终端单行刷新显示计算进度。
    progress_every : int，默认 20
        每处理多少个目标截面刷新一次进度。

    返回
    ----
    pandas.DataFrame
        date、instrument、mfd_sellamt_nd 三列。预热不足、资金流缺失、
        市值无效或中性化样本不足时保留 NaN，不填充为 0。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")

    window_info = _resolve_mfd_sellamt_nd_data_window(
        {"n_days": n_days}
    )
    n_days = window_info["lookback_trading_days"] + 1

    if min_observations is None:
        min_observations = n_days
    if (
        not isinstance(min_observations, (int, np.integer))
        or isinstance(min_observations, (bool, np.bool_))
        or not 1 <= int(min_observations) <= n_days
    ):
        raise ValueError(
            f"min_observations 必须是 1 至 {n_days} 之间的整数或 None。"
        )
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
        raise ValueError("min_cs_count 必须是大于等于 3 的整数。")

    winsor_k = float(winsor_k)
    if not np.isfinite(winsor_k) or winsor_k <= 0:
        raise ValueError("winsor_k 必须是有限正数。")

    required_columns = {
        "date",
        "instrument",
        "main_outflow_amount",
        "float_market_cap",
    }
    if neutralize_industry:
        required_columns.add("industry")
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "mfd_sellamt_nd 因子缺少字段："
            f"{sorted(missing_columns)}"
        )

    selected_columns = [
        "date",
        "instrument",
        "main_outflow_amount",
        "float_market_cap",
    ]
    if neutralize_industry:
        selected_columns.append("industry")

    df = data.loc[:, selected_columns].copy()
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.normalize()
    if df["date"].isna().any():
        raise ValueError("mfd_sellamt_nd 输入存在无效 date。")
    if df["instrument"].isna().any():
        raise ValueError("mfd_sellamt_nd 的 instrument 不允许缺失。")

    duplicated = df.duplicated(["date", "instrument"], keep=False)
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["date", "instrument"]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            "mfd_sellamt_nd 输入存在重复 date + instrument："
            f"{examples}"
        )

    for column in ["main_outflow_amount", "float_market_cap"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        if pd.isna(cutoff):
            raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= cutoff.normalize()].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    target_date_index = _normalize_target_dates(
        df["date"], target_dates
    )
    if target_date_index.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.sort_values(
        ["instrument", "date"], kind="mergesort"
    ).reset_index(drop=True)
    started_at = time.perf_counter()

    if show_progress:
        print(
            "\r[mfd_sellamt_nd] [1/2] "
            f"计算 {n_days} 日累计主力流出额...",
            end="",
            flush=True,
        )

    try:
        rolling_amount = (
            df.groupby("instrument", sort=False)["main_outflow_amount"]
            .rolling(
                window=n_days,
                min_periods=int(min_observations),
            )
            .sum()
            .reset_index(level=0, drop=True)
        )
        df["_factor_raw"] = -rolling_amount.reindex(df.index)

        target_panel = df.loc[
            df["date"].isin(target_date_index)
        ].copy()
        grouped_dates = list(target_panel.groupby("date", sort=True))
        total_dates = len(grouped_dates)
        result_parts = []

        for position, (date, section) in enumerate(
            grouped_dates, start=1
        ):
            section = section.copy()
            raw_z = _robust_zscore(
                section["_factor_raw"], winsor_k=winsor_k
            )
            industry = (
                section["industry"].fillna("UNKNOWN").astype(str)
                if neutralize_industry
                else None
            )
            residual = neutralize_size_industry(
                target=raw_z,
                market_cap=section["float_market_cap"],
                industry=industry,
                min_obs=int(min_cs_count),
                standardize_residual=False,
                zscore_ddof=0,
                show_progress=False,
            )
            factor = _robust_zscore(
                residual, winsor_k=winsor_k
            )

            result_parts.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "instrument": section["instrument"].to_numpy(),
                        "mfd_sellamt_nd": factor.to_numpy(),
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
                    "\r[mfd_sellamt_nd] [2/2] "
                    f"{position}/{total_dates} 个截面 "
                    f"| {position / total_dates:.1%} "
                    f"| 当前：{date:%Y-%m-%d} "
                    f"| 累计值有效：{section['_factor_raw'].notna().sum():,} "
                    f"| 输出有效：{factor.notna().sum():,} "
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
    "name": "mfd_sellamt_nd",
    "func": calc_mfd_sellamt_nd,
    "category": "moneyflow",
    "direction": 1,
    "description": (
        "过去 N 个交易日主力流出额累计值的反向因子，经稳健标准化、"
        "流通市值和可选行业中性化，并对残差再次稳健标准化；"
        "最终数值越高，按旧华泰研究方向越优。"
    ),
    "formula": (
        "sum_outflow_t=Sum(main_outflow_amount_{t-N+1:t})；"
        "raw_t=-sum_outflow_t；raw逐截面执行scaled-MAD去极值和总体"
        "Z-score，再对log(float_market_cap)和可选行业哑变量做OLS；"
        "最终因子为残差再次scaled-MAD去极值后的总体Z-score。"
    ),
    "input_schema": {
        "required": {
            "date": {
                "dtype": "datetime64[ns] 或可解析日期",
                "meaning": "日频资金流观测日期及目标因子截面日期。",
            },
            "instrument": {
                "dtype": "string",
                "meaning": "证券唯一标识；同一date+instrument不允许重复。",
            },
            "main_outflow_amount": {
                "dtype": "float",
                "meaning": (
                    "单日主力流出额；语义为主力被动买入额与主力主动"
                    "卖出额之和，金额单位必须在整个窗口内一致。"
                ),
            },
            "float_market_cap": {
                "dtype": "float",
                "meaning": "目标日流通市值；取自然对数后用于截面中性化。",
            },
        },
        "conditional": {
            "industry": {
                "dtype": "string",
                "meaning": "目标日点时可得的一级行业分类。",
                "required_when": {"neutralize_industry": True},
            },
        },
    },
    "parameters": {
        "target_dates": {
            "default": None,
            "accepted_values": "单个日期、日期序列或None。",
            "effect": "指定实际输出截面；None输出data中的全部日期。",
            "changes_data_requirements": True,
        },
        "as_of_date": {
            "default": None,
            "accepted_values": "可解析日期或None。",
            "effect": "全局信息截止日，晚于该日的数据不参与计算。",
            "changes_data_requirements": False,
        },
        "n_days": {
            "default": 1,
            "accepted_values": "正整数。",
            "effect": "决定主力流出额累计窗口和历史预热长度。",
            "changes_data_requirements": True,
        },
        "min_observations": {
            "default": None,
            "accepted_values": "1至n_days之间的整数或None。",
            "effect": "控制N日窗口内最少有效资金流观测数；None等于n_days。",
            "changes_data_requirements": False,
        },
        "neutralize_industry": {
            "default": True,
            "accepted_values": [True, False],
            "effect": "True为流通市值+行业中性化；False仅做市值中性化。",
            "changes_data_requirements": True,
        },
        "winsor_k": {
            "default": 3.0,
            "accepted_values": "有限正数。",
            "effect": "控制两次scaled-MAD去极值边界。",
            "changes_data_requirements": False,
        },
        "min_cs_count": {
            "default": 30,
            "accepted_values": "大于等于3的整数。",
            "effect": "控制单日中性化的最低有效样本数。",
            "changes_data_requirements": False,
        },
        "show_progress": {
            "default": False,
            "accepted_values": [True, False],
            "effect": "仅控制进度显示，不改变计算结果。",
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
        "resolver": _resolve_mfd_sellamt_nd_data_window,
        "default": {
            "lookback_trading_days": 0,
            "requires_target_date_data": True,
            "minimum_history_observations": 0,
            "preheating_required": False,
            "insufficient_window_behavior": (
                "默认1日实例不需要历史预热；单日资金流缺失时输出NaN。"
            ),
        },
        "resolver_notes": (
            "N日累计窗口包含目标日，因此实际预热为n_days-1个交易日；"
            "策略和评价函数必须使用本次resolved_factor_params解析窗口。"
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
        "mfd_sellamt_nd": {
            "dtype": "float64",
            "meaning": (
                "市值行业中性化后的N日累计主力流出额反向因子；"
                "数值越高，按旧华泰研究方向越优。"
            ),
        },
    },
    "usage_notes": [
        "n_days=1复现旧notebook的mfd_sellamt_d。",
        "min_observations=None时严格要求窗口内N个有效日频观测。",
        "因子层不负责ST、停牌、上市天数、板块或策略股票池过滤。",
        "研究和策略层应剔除NaN因子值，不应填充为0。",
        "中性化结果依赖目标日股票截面，跨研究比较时应固定股票池。",
    ],
    "best_practice": {
        "instance_name": "mfd_sellamt_1d",
        "parameters": {
            "n_days": 1,
            "min_observations": None,
            "neutralize_industry": True,
            "winsor_k": 3.0,
            "min_cs_count": 30,
        },
        "description": "当前最佳实践保留原研究的单日主力流出额实例。",
    },
    "pit_notes": [
        "滚动累计只使用目标日及以前的数据，不读取未来资金流。",
        "行业和流通市值必须使用对应历史目标日的点时值。",
        "目标日完整资金流依赖当日成交，因此信号在目标日收盘后形成。",
        "不同数据源对主力单的划分阈值可能不同，跨源对照必须核对定义。",
    ],
    "references": [
        (
            "Bigquant_research：研究稿/华泰因子复现/华泰资金流向类因子/"
            "mfd_sellamt_d因子/mfd_sellamt_d.ipynb。"
        ),
        "BigQuant cn_stock_moneyflow 官方字段说明。",
    ],
    "tags": [
        "moneyflow",
        "main_outflow_amount",
        "rolling_sum",
        "size_neutralized",
        "industry_neutralized",
        "parameterized_window",
    ],
    "status": "research",
    "version": "1.0.0",
}
