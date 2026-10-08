# -*- coding: utf-8 -*-
"""BP（Book-to-Price）估值因子。"""

import time

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_size_industry import (
    neutralize_size_industry,
)
from factor_lib.common.preprocess.winsorize_mad import winsorize_mad
from factor_lib.common.preprocess.zscore import zscore


OUTPUT_COLUMNS = ["date", "instrument", "book_to_price"]


def _normalize_target_dates(target_dates):
    """将单日或日期序列规范为去重、排序后的 DatetimeIndex。"""
    if isinstance(target_dates, (str, pd.Timestamp)):
        target_dates = [target_dates]
    else:
        try:
            target_dates = list(target_dates)
        except TypeError:
            target_dates = [target_dates]

    return pd.DatetimeIndex(pd.to_datetime(target_dates)).unique().sort_values()


def _empty_result():
    """返回统一结构的空因子面板。"""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


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
    """计算 BP 因子：BP = 1 / PB。

    因子模块只接收数据源无关的标准字段，不负责数据查询。数据加载器
    应根据本模块的 ``FACTOR["input_schema"]`` 和 ``FACTOR["data_window"]``
    准备数据；本因子不需要历史预热数据。

    处理流程：
    BP 原始值 → 市值中性化（可选叠加行业中性化）→ MAD 去极值
    → Z-score 标准化。无法计算的记录保留 NaN。

    参数
    ----
    data : pandas.DataFrame
        使用标准字段名的原始面板数据。必须包含 date、instrument、pb、
        total_market_cap；当 neutralize_industry=True 时还必须包含 industry。
    target_dates : 日期或可迭代日期对象，可选
        实际需要输出因子值的截面日期。为 None 时计算 data 中全部日期。
    as_of_date : str 或 datetime，可选
        全局信息截止日。仅使用该日期及以前的记录；不能替代 target_dates。
    neutralize_industry : bool，默认 True
        True：对 log(总市值) 和行业哑变量中性化；False：仅市值中性化。
    winsor_k : float，默认 5.0
        MAD 去极值倍数，必须大于 0。
    min_cs_count : int，默认 30
        单个截面的最小有效回归样本数，必须为正整数。样本不足时，
        该截面无法计算的结果保留为 NaN。
    show_progress : bool，默认 False
        是否以单行刷新形式显示截面计算进度。
    progress_every : int，默认 20
        每处理多少个截面刷新一次进度，必须为正整数。

    返回
    ----
    pandas.DataFrame
        固定包含 date、instrument、book_to_price 三列，仅含 target_dates
        对应截面。数值越大代表相对估值越低。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    if not isinstance(neutralize_industry, (bool, np.bool_)):
        raise TypeError("neutralize_industry 必须是 bool。")
    if not isinstance(min_cs_count, (int, np.integer)) or min_cs_count <= 0:
        raise ValueError("min_cs_count 必须是正整数。")
    if winsor_k <= 0:
        raise ValueError("winsor_k 必须大于 0。")
    if not isinstance(progress_every, (int, np.integer)) or progress_every <= 0:
        raise ValueError("progress_every 必须是正整数。")

    required_columns = {"date", "instrument", "pb", "total_market_cap"}
    if neutralize_industry:
        required_columns.add("industry")

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"BP 因子缺少字段：{sorted(missing_columns)}")

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("BP 因子的 date 存在无法解析的日期或缺失值。")
    if df["instrument"].isna().any():
        raise ValueError("BP 因子的 instrument 不允许缺失。")

    duplicated = df.duplicated(["date", "instrument"], keep=False)
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["date", "instrument"]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            "BP 因子输入存在重复的 date + instrument 记录："
            f"{examples}"
        )

    df["pb"] = pd.to_numeric(df["pb"], errors="coerce")
    df["total_market_cap"] = pd.to_numeric(
        df["total_market_cap"],
        errors="coerce",
    )

    if as_of_date is not None:
        as_of_timestamp = pd.Timestamp(as_of_date)
        df = df.loc[df["date"] <= as_of_timestamp].copy()

    if df.empty:
        return _empty_result()

    if target_dates is not None:
        target_date_index = _normalize_target_dates(target_dates)
        if target_date_index.empty:
            return _empty_result()

        available_dates = pd.DatetimeIndex(df["date"].unique())
        missing_target_dates = target_date_index.difference(available_dates)
        if not missing_target_dates.empty:
            missing_preview = [
                date.strftime("%Y-%m-%d")
                for date in missing_target_dates[:5]
            ]
            raise ValueError(
                "BP 因子缺少目标日期的原始数据："
                f"{missing_preview}。请检查 target_dates、数据日期范围和 "
                "as_of_date。"
            )

        # BP 是零历史窗口截面因子，只保留实际输出截面即可。
        df = df.loc[df["date"].isin(target_date_index)].copy()

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

            # PB 必须为正；BP = 1 / PB。
            bp_raw = pd.Series(np.nan, index=cross_section.index, dtype=float)
            valid_pb = cross_section["pb"].notna() & (cross_section["pb"] > 0)
            bp_raw.loc[valid_pb] = 1.0 / cross_section.loc[valid_pb, "pb"]

            # 统一调用公共市值行业中性化函数；此处只取残差，随后仍按
            # 原 BP 流程执行 MAD 去极值和 Z-score 标准化。
            industry = (
                cross_section["industry"]
                if neutralize_industry
                else None
            )
            bp_neutral = neutralize_size_industry(
                target=bp_raw,
                market_cap=cross_section["total_market_cap"],
                industry=industry,
                min_obs=min_cs_count,
                standardize_residual=False,
                zscore_ddof=0,
                show_progress=False,
            )
            factor = zscore(
                winsorize_mad(bp_neutral, k=winsor_k),
                ddof=0,
                show_progress=False,
            )
            factor = factor.replace([np.inf, -np.inf], np.nan)

            result_parts.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "instrument": cross_section["instrument"].values,
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
                estimated_remaining = elapsed / position * (total_dates - position)
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
    "name": 'book_to_price',
    "func": calc_book_to_price,
    "factor_type": "base",
    "candidate_instances": {"default": {}},
    "input_schema": {
        "required": {
            'date': {},
            'instrument': {},
            'pb': {},
            'total_market_cap': {},
        },
        "conditional": {
            'industry': {"required_when": {'neutralize_industry': True}},
        },
    },
    "parameters": {
        'target_dates': {"default": None},
        'as_of_date': {"default": None},
        'neutralize_industry': {"default": True},
        'winsor_k': {"default": 5.0},
        'min_cs_count': {"default": 30},
        'show_progress': {"default": False},
        'progress_every': {"default": 20},
    },
    "data_window": {
        "lookback_trading_days": 0,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": False,
    },
    "output_schema": {
        'date': {},
        'instrument': {},
        'book_to_price': {},
    },
}


FACTOR_INFO = """
# BP（账面市值比）

以 PB 的倒数刻画相对估值。经市值中性化，并可选择行业中性化后进行去极值和标准化；数值较高通常对应相对低估。

- **计算**：BP = 1 / PB，仅保留正 PB。
- **时点**：PB、市值和行业必须为信号日当时可得数据；收盘信息形成的信号应在下一可交易时点执行。
- **研究提示**：宜与质量、成长等因子联合检验，不将其视为绝对估值结论。
"""
