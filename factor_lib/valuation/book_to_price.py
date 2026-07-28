# -*- coding: utf-8 -*-
"""BP（Book-to-Price）估值因子。"""

import time

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_ols import neutralize_ols
from factor_lib.common.preprocess.winsorize_mad import winsorize_mad
from factor_lib.common.preprocess.zscore import zscore


def calc_book_to_price(
    data,
    target_dates=None,
    as_of_date=None,
    neutralize_industry=True,
    winsor_k=5.0,
    min_cs_count=30,
    show_progress=False,
    progress_every=20,
):
    """
    计算 BP 因子：BP = 1 / PB。

    输入字段使用跨数据源的标准字段名。数据加载层负责将 BigQuant、
    其他数据库或本地文件中的实际字段映射为这些标准字段名。

    处理流程：
    BP 原始值
    → 市值中性化，默认再叠加行业中性化
    → MAD 去极值
    → Z-score 标准化
    → 缺失值填 0

    参数
    ----
    data : pandas.DataFrame
        已准备好的原始数据，可包含目标日期以外的历史日期。
        必须包含 date、instrument、pb、total_market_cap。
        当 neutralize_industry=True 时，还必须包含 industry。
    target_dates : 可迭代日期对象或单个日期，可选
        需要输出因子值的目标截面日期。
        为 None 时，对 data 中全部日期计算。
    as_of_date : str 或 datetime，可选
        全局可用数据的时间上限；只使用该日期及以前的数据。
        该参数不能替代 target_dates。
    neutralize_industry : bool，默认 True
        True 为市值+行业中性化；False 为仅市值中性化。
    winsor_k : float，默认 5.0
        MAD 去极值倍数。
    min_cs_count : int，默认 30
        单日截面最小有效样本数。
    show_progress : bool，默认 False
        是否在终端用单行刷新方式显示计算进度。
    progress_every : int，默认 20
        每处理多少个目标截面刷新一次进度。

    返回
    ----
    pandas.DataFrame
        包含 date、instrument、book_to_price 三列。
        仅输出 target_dates 对应的截面。
        数值越大，代表相对估值越低。
    """
    if progress_every <= 0:
        raise ValueError("progress_every 必须为正整数")

    required_columns = {
        "date",
        "instrument",
        "pb",
        "total_market_cap",
    }

    if neutralize_industry:
        required_columns.add("industry")

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"BP 因子缺少字段：{sorted(missing_columns)}")

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    if as_of_date is not None:
        df = df[
            df["date"] <= pd.Timestamp(as_of_date)
        ].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["date", "instrument", "book_to_price"]
        )

    if target_dates is not None:
        # 同时兼容单个日期、list、set、Series、DatetimeIndex 等输入。
        if isinstance(target_dates, (str, pd.Timestamp)):
            target_dates = [target_dates]
        else:
            try:
                target_dates = list(target_dates)
            except TypeError:
                target_dates = [target_dates]

        target_date_index = pd.DatetimeIndex(
            pd.to_datetime(target_dates)
        ).unique().sort_values()

        if target_date_index.empty:
            return pd.DataFrame(
                columns=["date", "instrument", "book_to_price"]
            )

        available_dates = pd.DatetimeIndex(
            df["date"].dropna().unique()
        )

        missing_target_dates = target_date_index.difference(
            available_dates
        )

        if not missing_target_dates.empty:
            missing_preview = [
                date.strftime("%Y-%m-%d")
                for date in missing_target_dates[:5]
            ]

            raise ValueError(
                "BP 因子缺少目标日期的原始数据："
                f"{missing_preview}"
                "。请检查 target_dates、数据日期范围和 as_of_date。"
            )

        df = df[df["date"].isin(target_date_index)].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["date", "instrument", "book_to_price"]
        )

    total_dates = df["date"].nunique()
    result_parts = []
    start_time = time.perf_counter()

    if show_progress:
        print(
            f"\r[BP 因子] 0/{total_dates} 个目标截面 | 0.0%",
            end="",
            flush=True,
        )

    try:
        for position, (date, cross_section) in enumerate(
            df.groupby("date", sort=True),
            start=1,
        ):
            cross_section = cross_section.copy()

            # 原始定义：PB 必须为正，BP = 1 / PB。
            bp_raw = pd.Series(
                np.nan,
                index=cross_section.index,
                dtype=float,
            )

            valid_pb = (
                cross_section["pb"].notna()
                & (cross_section["pb"] > 0)
            )

            bp_raw.loc[valid_pb] = (
                1.0 / cross_section.loc[valid_pb, "pb"]
            )

            # 控制变量：log(总市值)，以及可选的行业哑变量。
            log_market_cap = np.log(
                cross_section["total_market_cap"].where(
                    cross_section["total_market_cap"] > 0
                )
            ).rename("log_total_market_cap")

            controls = log_market_cap.to_frame()

            if neutralize_industry:
                industry_dummies = pd.get_dummies(
                    cross_section["industry"]
                    .fillna("unknown")
                    .astype(str),
                    prefix="industry",
                    drop_first=True,
                    dtype=float,
                )

                controls = pd.concat(
                    [controls, industry_dummies],
                    axis=1,
                )

            bp_neutral = neutralize_ols(
                target=bp_raw,
                controls=controls,
                min_obs=min_cs_count,
            )

            factor = zscore(
                winsorize_mad(bp_neutral, k=winsor_k)
            )

            factor = factor.replace(
                [np.inf, -np.inf],
                np.nan,
            ).fillna(0.0)

            result_parts.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "instrument": (
                            cross_section["instrument"].values
                        ),
                        "book_to_price": factor.values,
                    }
                )
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
                    f"[BP 因子] {position}/{total_dates} 个目标截面 "
                    f"| {position / total_dates:.1%} "
                    f"| 当前：{pd.Timestamp(date):%Y-%m-%d} "
                    f"| 已耗时：{elapsed:.1f}s "
                    f"| 预计剩余：{estimated_remaining:.1f}s",
                    end="",
                    flush=True,
                )
    finally:
        if show_progress:
            print()

    return pd.concat(result_parts, ignore_index=True)


FACTOR = {
    "name": "book_to_price",
    "func": calc_book_to_price,
    "category": "valuation",
    "direction": 1,
    "description": (
        "BP=1/PB，经市值和可选行业中性化、MAD去极值及Z-score标准化。"
    ),
    "required_columns": [
        "date",
        "instrument",
        "pb",
        "total_market_cap",
    ],
    "conditional_columns": {
        "industry": (
            "当 neutralize_industry=True 时必需；"
            "默认启用行业中性化。"
        ),
    },
    "data_window": {
        "lookback_trading_days": 0,
        "requires_current_date_data": True,
        "description": (
            "纯当日截面因子，不依赖目标日期之前的历史数据，"
            "不需要预热期。"
        ),
    },
    "output_columns": [
        "date",
        "instrument",
        "book_to_price",
    ],
}
