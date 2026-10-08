# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def neutralize_ols(target, controls, min_obs=30):
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
