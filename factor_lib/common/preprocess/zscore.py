# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def zscore(series):
    """
    使用总体标准差（ddof=0）进行截面标准化。
    截面无有效波动时返回 NaN，由调用因子决定是否填 0。
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series 必须是 pandas.Series")

    std = series.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, name=series.name)

    return (series - series.mean()) / std
