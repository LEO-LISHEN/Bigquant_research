# -*- coding: utf-8 -*-
"""Profit_G_q：当季净利润同比增长率成长因子。"""

import time

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.winsorize_mad import winsorize_mad
from factor_lib.common.preprocess.zscore import zscore


OUTPUT_COLUMNS = ["date", "instrument", "profit_g_q"]


def calc_profit_g_q(
    data,
    target_dates=None,
    as_of_date=None,
    winsor_k=5.0,
    min_cs_count=30,
    show_progress=False,
    progress_every=20,
):
    """计算 Profit_G_q 裸因子。

    原始定义为最新可得财报的单季度净利润同比增长率。
    每个目标日独立执行：

    单季度净利润同比增长率
    → MAD 去极值
    → Z-score 标准化

    “裸因子”表示不做行业和市值中性化。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if (
        not isinstance(progress_every, (int, np.integer))
        or isinstance(progress_every, (bool, np.bool_))
        or progress_every <= 0
    ):
        raise ValueError("progress_every 必须是正整数。")
    if (
        not isinstance(min_cs_count, (int, np.integer))
        or isinstance(min_cs_count, (bool, np.bool_))
        or min_cs_count < 3
    ):
        raise ValueError("min_cs_count 必须是大于等于 3 的整数。")

    winsor_k = float(winsor_k)
    if not np.isfinite(winsor_k) or winsor_k <= 0:
        raise ValueError("winsor_k 必须是有限正数。")

    required_columns = {
        "date",
        "instrument",
        "quarterly_net_profit_yoy",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "profit_g_q 因子缺少字段："
            f"{sorted(missing_columns)}"
        )

    df = data.loc[
        :,
        [
            "date",
            "instrument",
            "quarterly_net_profit_yoy",
        ],
    ].copy()
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()
    if df["date"].isna().any():
        raise ValueError("profit_g_q 输入存在无效 date。")
    if df["instrument"].isna().any():
        raise ValueError(
            "profit_g_q 输入的 instrument 不允许缺失。"
        )

    duplicated = df.duplicated(
        ["date", "instrument"],
        keep=False,
    )
    if duplicated.any():
        examples = (
            df.loc[
                duplicated,
                ["date", "instrument"],
            ]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            "profit_g_q 输入存在重复 date + instrument："
            f"{examples}"
        )

    df["quarterly_net_profit_yoy"] = pd.to_numeric(
        df["quarterly_net_profit_yoy"],
        errors="coerce",
    )
    df["quarterly_net_profit_yoy"] = df[
        "quarterly_net_profit_yoy"
    ].replace([np.inf, -np.inf], np.nan)

    if as_of_date is not None:
        as_of_timestamp = pd.Timestamp(as_of_date)
        if pd.isna(as_of_timestamp):
            raise ValueError("as_of_date 必须是可解析日期。")
        as_of_timestamp = as_of_timestamp.normalize()
        df = df.loc[df["date"] <= as_of_timestamp].copy()

    if target_dates is None:
        target_date_index = pd.DatetimeIndex(
            df["date"].dropna().unique()
        ).sort_values()
    else:
        if isinstance(target_dates, (str, pd.Timestamp)):
            target_dates = [target_dates]
        else:
            try:
                target_dates = list(target_dates)
            except TypeError:
                target_dates = [target_dates]

        target_date_index = pd.DatetimeIndex(
            pd.to_datetime(target_dates, errors="raise")
        ).normalize().unique().sort_values()

    if target_date_index.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    available_dates = pd.DatetimeIndex(df["date"].unique())
    missing_target_dates = target_date_index.difference(
        available_dates
    )
    if not missing_target_dates.empty:
        preview = [
            date.strftime("%Y-%m-%d")
            for date in missing_target_dates[:5]
        ]
        raise ValueError(
            "profit_g_q 缺少目标日财务截面："
            f"{preview}。请检查财务适配器日期和 as_of_date。"
        )

    df = df.loc[df["date"].isin(target_date_index)].copy()
    total_dates = len(target_date_index)
    result_parts = []
    started_at = time.perf_counter()

    if show_progress:
        print(
            f"\r[profit_g_q] 0/{total_dates} 个截面 | 0.0%",
            end="",
            flush=True,
        )

    try:
        for position, date in enumerate(
            target_date_index,
            start=1,
        ):
            cross_section = df.loc[df["date"] == date].copy()
            raw_factor = cross_section[
                "quarterly_net_profit_yoy"
            ].astype(float)
            valid_count = int(raw_factor.notna().sum())

            factor = pd.Series(
                np.nan,
                index=cross_section.index,
                dtype=float,
            )
            if valid_count >= int(min_cs_count):
                valid_values = raw_factor.dropna()
                median = valid_values.median()
                mad = (valid_values - median).abs().median()

                if pd.notna(mad) and mad > 1e-12:
                    processed = winsorize_mad(
                        raw_factor,
                        k=winsor_k,
                    )
                else:
                    lower, upper = valid_values.quantile(
                        [0.01, 0.99]
                    )
                    processed = raw_factor.clip(
                        lower=lower,
                        upper=upper,
                    )

                factor = zscore(processed)
                factor = factor.replace(
                    [np.inf, -np.inf],
                    np.nan,
                )

            result_parts.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "instrument": cross_section[
                            "instrument"
                        ].to_numpy(),
                        "profit_g_q": factor.to_numpy(),
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
                remaining = (
                    elapsed / position
                    * (total_dates - position)
                )
                print(
                    "\r"
                    f"[profit_g_q] {position}/{total_dates} 个截面 "
                    f"| {position / total_dates:.1%} "
                    f"| 当前：{date:%Y-%m-%d} "
                    f"| 有效样本：{valid_count:,} "
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
        .sort_values(
            ["date", "instrument"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


FACTOR = {
    "name": "profit_g_q",
    "func": calc_profit_g_q,
    "category": "growth",
    "direction": 1,
    "description": (
        "最新可得财报的单季度净利润同比增长率，"
        "经截面 MAD 去极值和 Z-score 标准化；不做中性化。"
    ),
    "formula": (
        "raw_t = quarterly_net_profit_t "
        "/ quarterly_net_profit_{t-4q} - 1；"
        "对目标日截面执行 median±winsor_k×MAD 去极值后，"
        "使用总体标准差进行 Z-score 标准化。"
    ),
    "input_schema": {
        "required": {
            "date": {
                "dtype": "datetime64[ns]",
                "meaning": "目标财务因子截面交易日。",
                "frequency": "financial",
            },
            "instrument": {
                "dtype": "string",
                "meaning": "证券唯一标识。",
                "frequency": "financial",
            },
            "quarterly_net_profit_yoy": {
                "dtype": "float64",
                "meaning": (
                    "截至目标日最新可得的单季度净利润同比增长率。"
                ),
                "frequency": "financial",
            },
        },
        "conditional": {},
    },
    "parameters": {
        "target_dates": {
            "default": None,
            "accepted_values": "单个日期、日期序列或 None。",
            "effect": "指定实际输出的因子截面。",
            "changes_data_requirements": True,
        },
        "as_of_date": {
            "default": None,
            "accepted_values": "可解析日期或 None。",
            "effect": "全局信息截止日，晚于该日的数据不参与计算。",
            "changes_data_requirements": False,
        },
        "winsor_k": {
            "default": 5.0,
            "accepted_values": "有限正数。",
            "effect": "控制 MAD 去极值边界。",
            "changes_data_requirements": False,
        },
        "min_cs_count": {
            "default": 30,
            "accepted_values": "大于等于 3 的整数。",
            "effect": "单日有效样本不足时，该截面输出 NaN。",
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
            "该因子直接使用目标日点时财务指标；"
            "目标日缺失时报错，单只股票缺失时保留 NaN。"
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
        "profit_g_q": {
            "dtype": "float64",
            "meaning": "标准化净利润季度同比成长因子；数值越高越优。",
        },
    },
    "usage_notes": [
        "裸因子版本不进行行业或市值中性化。",
        "策略和研究层应剔除 NaN 因子值，不应填充为 0。",
        (
            "BigQuant 财务适配器标准字段 quarterly_net_profit_yoy "
            "映射到 net_profit_yoy_mrq。"
        ),
    ],
    "pit_notes": [
        (
            "必须使用目标日已经可得的财务数据，不能按报告期结束日"
            "提前回填尚未公告的财报。"
        ),
        (
            "当前 BigQuant 适配使用日频财务因子表；迁移到其他平台时，"
            "数据适配层必须按公告/可得时间完成点时对齐。"
        ),
        (
            "若数据源存在财报更正或历史重述，应核验其是否保留历史版本，"
            "否则仍可能存在修订数据偏差。"
        ),
    ],
    "references": [
        (
            "研究稿/华泰因子复现/华泰成长类因子/"
            "Profit_G_q因子/Profit_G_q.ipynb"
        ),
    ],
    "status": "migrated_pending_bigquant_validation",
    "version": "1.0.0",
}
