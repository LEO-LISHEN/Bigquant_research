# -*- coding: utf-8 -*-
"""计算多因子之间的截面相关性矩阵。"""

import time

import numpy as np
import pandas as pd


def calculate_factor_correlation(
    factor_data,
    factor_columns,
    date_column="date",
    instrument_column="instrument",
    method="spearman",
    min_obs=30,
    show_progress=False,
    progress_every=20,
):
    """
    按日计算因子截面相关系数，再对各交易日相关系数取均值。

    factor_data 是宽表，例如：
    date、instrument、book_to_price、roe、price_momentum。

    参数
    ----
    show_progress : bool，默认 False
        是否在终端用单行刷新方式显示计算进度。
    progress_every : int，默认 20
        每处理多少个交易日刷新一次进度。

    返回
    ----
    correlation_matrix : pandas.DataFrame
        各因子的平均截面相关系数矩阵。
    overlap_days : pandas.DataFrame
        每一对因子实际参与平均计算的交易日数量。
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method 仅支持 pearson 或 spearman")

    if len(factor_columns) < 2:
        raise ValueError("至少需要两个因子列")

    if progress_every <= 0:
        raise ValueError("progress_every 必须为正整数")

    required_columns = {
        date_column,
        instrument_column,
        *factor_columns,
    }
    missing_columns = required_columns - set(factor_data.columns)

    if missing_columns:
        raise ValueError(
            f"factor_data 缺少字段：{sorted(missing_columns)}"
        )

    panel = factor_data[
        [date_column, instrument_column, *factor_columns]
    ].copy()

    if panel.duplicated([date_column, instrument_column]).any():
        raise ValueError("factor_data 中存在重复的 date + instrument")

    panel[date_column] = pd.to_datetime(panel[date_column])
    panel = panel.replace([np.inf, -np.inf], np.nan)

    correlation_sum = pd.DataFrame(
        0.0,
        index=factor_columns,
        columns=factor_columns,
    )

    overlap_days = pd.DataFrame(
        0,
        index=factor_columns,
        columns=factor_columns,
        dtype=int,
    )

    total_dates = panel[date_column].nunique()
    start_time = time.perf_counter()

    if show_progress and total_dates > 0:
        print(
            f"\r[因子相关性] 0/{total_dates} 个截面 | 0.0%",
            end="",
            flush=True,
        )

    try:
        for position, (date, cross_section) in enumerate(
            panel.groupby(date_column, sort=True),
            start=1,
        ):
            correlation = cross_section[factor_columns].corr(
                method=method,
                min_periods=min_obs,
            )

            valid = correlation.notna()

            correlation_sum = correlation_sum.add(
                correlation.where(valid, 0.0),
                fill_value=0.0,
            )

            overlap_days = overlap_days.add(
                valid.astype(int),
                fill_value=0,
            )

            should_refresh = (
                position == 1
                or position % progress_every == 0
                or position == total_dates
            )

            if show_progress and should_refresh:
                elapsed = time.perf_counter() - start_time
                estimated_remaining = (
                    elapsed / position * (total_dates - position)
                )

                print(
                    "\r"
                    f"[因子相关性] {position}/{total_dates} 个截面 "
                    f"| {position / total_dates:.1%} "
                    f"| 当前：{pd.Timestamp(date):%Y-%m-%d} "
                    f"| 已耗时：{elapsed:.1f}s "
                    f"| 预计剩余：{estimated_remaining:.1f}s",
                    end="",
                    flush=True,
                )
    finally:
        if show_progress and total_dates > 0:
            print()

    correlation_matrix = correlation_sum.divide(
        overlap_days.replace(0, np.nan)
    )

    for factor_name in factor_columns:
        if overlap_days.loc[factor_name, factor_name] > 0:
            correlation_matrix.loc[factor_name, factor_name] = 1.0

    return correlation_matrix, overlap_days
