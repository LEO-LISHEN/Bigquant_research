# -*- coding: utf-8 -*-
"""市值与可选行业 OLS 中性化。"""

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_ols import neutralize_ols
from factor_lib.common.preprocess.zscore import zscore


def neutralize_size_industry(
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

        residual = neutralize_ols(
            target=y.where(valid),
            controls=controls,
            min_obs=min_obs,
        )
        residual = residual.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if standardize_residual:
            residual = zscore(
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
