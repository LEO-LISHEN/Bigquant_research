# -*- coding: utf-8 -*-
"""计算单因子的标准基础指标。"""

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.zscore import zscore


def calculate_factor_basic_metrics(
    factor_data,
    label_data,
    factor_column,
    return_column="forward_return",
    date_column="date",
    instrument_column="instrument",
    return_end_date_column="return_end_date",
    min_obs=30,
):
    """
    计算标准因子基础指标。

    factor_data 必须包含：
    date、instrument、指定的 factor_column。

    label_data 必须包含：
    date、instrument、forward_return、return_end_date。

    注意：forward_return 必须是信号日之后、已经完整结束的持有期收益。
    """
    factor_required = {date_column, instrument_column, factor_column}
    label_required = {
        date_column,
        instrument_column,
        return_column,
        return_end_date_column,
    }

    missing_factor = factor_required - set(factor_data.columns)
    missing_label = label_required - set(label_data.columns)

    if missing_factor:
        raise ValueError(f"factor_data 缺少字段：{sorted(missing_factor)}")

    if missing_label:
        raise ValueError(f"label_data 缺少字段：{sorted(missing_label)}")

    factor_panel = factor_data[
        [date_column, instrument_column, factor_column]
    ].copy()

    label_panel = label_data[
        [
            date_column,
            instrument_column,
            return_column,
            return_end_date_column,
        ]
    ].copy()

    if factor_panel.duplicated([date_column, instrument_column]).any():
        raise ValueError("factor_data 中存在重复的 date + instrument")

    if label_panel.duplicated([date_column, instrument_column]).any():
        raise ValueError("label_data 中存在重复的 date + instrument")

    factor_panel[date_column] = pd.to_datetime(factor_panel[date_column])
    label_panel[date_column] = pd.to_datetime(label_panel[date_column])
    label_panel[return_end_date_column] = pd.to_datetime(
        label_panel[return_end_date_column]
    )

    panel = factor_panel.merge(
        label_panel,
        on=[date_column, instrument_column],
        how="inner",
    )

    panel = panel.replace([np.inf, -np.inf], np.nan)

    records = []

    for signal_date, cross_section in panel.groupby(date_column, sort=True):
        valid = cross_section[
            [factor_column, return_column, return_end_date_column]
        ].dropna(subset=[factor_column, return_column])

        sample_count = len(valid)
        return_end_dates = valid[return_end_date_column].dropna()
        return_end_date = (
            return_end_dates.max() if not return_end_dates.empty else pd.NaT
        )

        ic = np.nan
        rank_ic = np.nan
        factor_return = np.nan
        factor_t_value = np.nan

        if sample_count >= min_obs:
            ic = valid[factor_column].corr(valid[return_column], method="pearson")
            rank_ic = valid[factor_column].corr(
                valid[return_column],
                method="spearman",
            )

            standardized_factor = zscore(valid[factor_column])
            regression_data = pd.DataFrame(
                {
                    "factor": standardized_factor,
                    "return": valid[return_column],
                }
            ).dropna()

            if len(regression_data) >= max(min_obs, 3):
                x = np.column_stack(
                    [
                        np.ones(len(regression_data)),
                        regression_data["factor"].to_numpy(dtype=float),
                    ]
                )
                y = regression_data["return"].to_numpy(dtype=float)

                if np.linalg.matrix_rank(x) == 2:
                    beta = np.linalg.lstsq(x, y, rcond=None)[0]
                    residual = y - x @ beta
                    degrees_of_freedom = len(y) - 2

                    if degrees_of_freedom > 0:
                        residual_variance = (
                            np.sum(residual ** 2) / degrees_of_freedom
                        )
                        covariance = residual_variance * np.linalg.inv(x.T @ x)
                        standard_error = np.sqrt(covariance[1, 1])

                        factor_return = beta[1]
                        if standard_error > 0 and np.isfinite(standard_error):
                            factor_t_value = beta[1] / standard_error

        records.append(
            {
                "signal_date": signal_date,
                "return_end_date": return_end_date,
                "sample_count": sample_count,
                "ic": ic,
                "rank_ic": rank_ic,
                "factor_return": factor_return,
                "factor_t_value": factor_t_value,
            }
        )

    timeseries = pd.DataFrame(
        records,
        columns=[
            "signal_date",
            "return_end_date",
            "sample_count",
            "ic",
            "rank_ic",
            "factor_return",
            "factor_t_value",
        ],
    )

    ic_std = timeseries["ic"].std(ddof=1)
    rank_ic_std = timeseries["rank_ic"].std(ddof=1)

    summary = pd.DataFrame(
        [
            {
                "factor_column": factor_column,
                "ic_mean": timeseries["ic"].mean(),
                "icir": (
                    timeseries["ic"].mean() / ic_std
                    if pd.notna(ic_std) and ic_std != 0
                    else np.nan
                ),
                "rank_ic_mean": timeseries["rank_ic"].mean(),
                "rank_icir": (
                    timeseries["rank_ic"].mean() / rank_ic_std
                    if pd.notna(rank_ic_std) and rank_ic_std != 0
                    else np.nan
                ),
                "factor_return_mean": timeseries["factor_return"].mean(),
                "mean_t_value": timeseries["factor_t_value"].mean(),
                "cross_section_count": len(timeseries),
                "valid_ic_count": timeseries["ic"].notna().sum(),
            }
        ]
    )

    return {
        "summary": summary,
        "timeseries": timeseries,
    }
