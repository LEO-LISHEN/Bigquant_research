# -*- coding: utf-8 -*-

import pandas as pd


def winsorize_mad(series, k=5.0):
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
