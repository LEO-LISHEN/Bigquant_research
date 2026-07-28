# -*- coding: utf-8 -*-
"""BP（Book-to-Price）估值因子。"""

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_ols import neutralize_ols
from factor_lib.common.preprocess.winsorize_mad import winsorize_mad
from factor_lib.common.preprocess.zscore import zscore


def calc_book_to_price(
    data,
    as_of_date=None,
    neutralize_industry=True,
    winsor_k=5.0,
    min_cs_count=30,
):
    """
    计算 BP 因子：BP = 1 / PB。

    处理流程：
    BP 原始值
    → 市值中性化，默认再叠加行业中性化
    → MAD 去极值
    → Z-score 标准化
    → 缺失值填 0

    参数
    ----
    data : pandas.DataFrame
        必须包含 date、instrument、pb、total_market_cap。
        当 neutralize_industry=True 时，还必须包含 industry。
    as_of_date : str 或 datetime，可选
        仅计算该日期及以前的数据，不读取未来数据。
    neutralize_industry : bool，默认 True
        True 为市值+行业中性化；False 为仅市值中性化。
    winsor_k : float，默认 5.0
        MAD 去极值倍数。
    min_cs_count : int，默认 30
        单日截面最小有效样本数。

    返回
    ----
    pandas.DataFrame
        包含 date、instrument、book_to_price 三列。
        数值越大，代表相对估值越低。
    """
    required_columns = {"date", "instrument", "pb", "total_market_cap"}
    if neutralize_industry:
        required_columns.add("industry")

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"BP 因子缺少字段：{sorted(missing_columns)}")

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    if as_of_date is not None:
        df = df[df["date"] <= pd.Timestamp(as_of_date)].copy()

    if df.empty:
        return pd.DataFrame(columns=["date", "instrument", "book_to_price"])

    result_parts = []

    for date, cross_section in df.groupby("date", sort=True):
        cross_section = cross_section.copy()

        # 原 notebook 定义：PB 必须为正，BP = 1 / PB。
        bp_raw = pd.Series(np.nan, index=cross_section.index, dtype=float)
        valid_pb = cross_section["pb"].notna() & (cross_section["pb"] > 0)
        bp_raw.loc[valid_pb] = 1.0 / cross_section.loc[valid_pb, "pb"]

        # 控制变量：log(总市值)，以及可选的行业哑变量。
        log_market_cap = np.log(
            cross_section["total_market_cap"].where(
                cross_section["total_market_cap"] > 0
            )
        ).rename("log_total_market_cap")

        controls = log_market_cap.to_frame()

        if neutralize_industry:
            industry_dummies = pd.get_dummies(
                cross_section["industry"].fillna("unknown").astype(str),
                prefix="industry",
                drop_first=True,
                dtype=float,
            )
            controls = pd.concat([controls, industry_dummies], axis=1)

        bp_neutral = neutralize_ols(
            target=bp_raw,
            controls=controls,
            min_obs=min_cs_count,
        )

        factor = zscore(winsorize_mad(bp_neutral, k=winsor_k))
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        result_parts.append(
            pd.DataFrame(
                {
                    "date": date,
                    "instrument": cross_section["instrument"].values,
                    "book_to_price": factor.values,
                }
            )
        )

    return pd.concat(result_parts, ignore_index=True)


FACTOR = {
    "name": "book_to_price",
    "func": calc_book_to_price,
    "category": "valuation",
    "direction": 1,
    "description": "BP=1/PB，经市值和可选行业中性化、MAD去极值及Z-score标准化。",
}
