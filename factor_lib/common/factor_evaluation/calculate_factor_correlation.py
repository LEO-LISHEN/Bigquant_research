# -*- coding: utf-8 -*-
"""计算多因子之间的截面相关性矩阵。"""

import numpy as np
import pandas as pd


def calculate_factor_correlation(
    factor_data,
    factor_columns,
    date_column="date",
    instrument_column="instrument",
    method="spearman",
    min_obs=30,
):
    """
    按日计算因子截面相关系数，再对各交易日相关系数取均值。

    factor_data 是宽表，例如：
    date、instrument、book_to_price、roe、price_momentum。

    返回：
    correlation_matrix, overlap_days_matrix
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method 仅支持 pearson 或 spearman")

    if len(factor_columns) < 2:
        raise ValueError("至少需要两个因子列")

    required_columns = {date_column, instrument_column, *factor_columns}
    missing_columns = required_columns - set(factor_data.columns)

    if missing_columns:
        raise ValueError(f"factor_data 缺少字段：{sorted(missing_columns)}")

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

    for _, cross_section in panel.groupby(date_column, sort=True):
        correlation = cross_section[factor_columns].corr(
            method=method,
            min_periods=min_obs,
        )

        valid = correlation.notna()
        correlation_sum = correlation_sum.add(
            correlation.where(valid, 0.0),
            fill_value=0.0,
        )
        overlap_days = overlap_days.add(valid.astype(int), fill_value=0)

    correlation_matrix = correlation_sum.divide(
        overlap_days.replace(0, np.nan)
    )

    for factor_name in factor_columns:
        if overlap_days.loc[factor_name, factor_name] > 0:
            correlation_matrix.loc[factor_name, factor_name] = 1.0

    return correlation_matrix, overlap_days
