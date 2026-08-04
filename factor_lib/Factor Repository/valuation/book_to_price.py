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
    "name": "book_to_price",
    "func": calc_book_to_price,
    "category": "valuation",
    "direction": 1,
    "description": "BP=1/PB；经市值和可选行业中性化、MAD 去极值及 Z-score 标准化。",
    "formula": (
        "raw_bp = 1 / pb（仅 pb > 0）；对 raw_bp 关于 log(total_market_cap) "
        "及可选行业哑变量进行截面 OLS 中性化，再做 MAD 去极值与 Z-score 标准化。"
    ),
    "input_schema": {
        "required": {
            "date": {
                "dtype": "datetime64[ns] 或可解析日期",
                "meaning": "观测/信号截面日期。",
            },
            "instrument": {
                "dtype": "string",
                "meaning": "证券唯一标识；同一 date + instrument 不允许重复。",
            },
            "pb": {
                "dtype": "float",
                "meaning": "市净率（Price-to-Book）。仅正值参与 BP 原始值计算。",
            },
            "total_market_cap": {
                "dtype": "float",
                "meaning": "当日总市值；正值取自然对数后用于市值中性化。",
            },
        },
        "conditional": {
            "industry": {
                "dtype": "string 或分类变量",
                "meaning": "当日行业分类，用于构造行业哑变量。",
                "required_when": {"neutralize_industry": True},
                "default_behavior_when_missing": "仅在未启用行业中性化时允许不提供。",
            },
        },
    },
    "parameters": {
        "target_dates": {
            "default": None,
            "accepted_values": "单个日期或可迭代日期对象。",
            "effect": "指定实际输出的因子截面；None 表示计算 data 中全部日期。",
            "changes_data_requirements": True,
        },
        "as_of_date": {
            "default": None,
            "accepted_values": "可解析日期或 None。",
            "effect": "全局信息截止日；晚于该日的数据一律不参与计算。",
            "changes_data_requirements": False,
        },
        "neutralize_industry": {
            "default": True,
            "accepted_values": [True, False],
            "effect": "控制是否在市值中性化基础上增加行业中性化。",
            "changes_data_requirements": True,
            "additional_required_fields_when_true": ["industry"],
        },
        "winsor_k": {
            "default": 5.0,
            "accepted_values": "大于 0 的 float。",
            "effect": "MAD 去极值的阈值倍数；值越小，极值处理越严格。",
            "changes_data_requirements": False,
        },
        "min_cs_count": {
            "default": 30,
            "accepted_values": "正整数。",
            "effect": "单日中性化回归所需最小有效样本数；不足时保留 NaN。",
            "changes_data_requirements": False,
        },
        "show_progress": {
            "default": False,
            "accepted_values": [True, False],
            "effect": "仅控制终端进度显示，不改变计算结果。",
            "changes_data_requirements": False,
        },
        "progress_every": {
            "default": 20,
            "accepted_values": "正整数。",
            "effect": "进度刷新间隔（按目标截面计）。",
            "changes_data_requirements": False,
        },
    },
    "data_window": {
        "lookback_trading_days": 0,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": False,
        "insufficient_window_behavior": "不适用；这是零历史窗口的当日截面因子。",
        "insufficient_cross_section_behavior": (
            "有效样本数不足以完成中性化时，对应因子值保留为 NaN。"
        ),
    },
    "output_schema": {
        "date": {
            "dtype": "datetime64[ns]",
            "meaning": "目标因子截面日期。",
        },
        "instrument": {
            "dtype": "string",
            "meaning": "证券唯一标识。",
        },
        "book_to_price": {
            "dtype": "float64",
            "meaning": "标准化 BP 暴露；数值越大代表相对估值越低。",
        },
    },
    "usage_notes": [
        "因子模块不读取任何数据源；加载器负责将实际字段映射为本元信息中的标准字段。",
        "适用于目标日可获得 PB、总市值及可选行业分类的股票截面。",
        "direction=1 仅说明该因子的经验方向；具体选股排序由调用策略显式决定。",
    ],
    "pit_notes": [
        "pb、total_market_cap 与 industry 必须是目标日信号形成时真实可获得的点时数据。",
        "财务类 PB 的底层口径必须避免使用目标日后才发布或修订的信息。",
        "该因子仅使用目标日截面，不使用未来数据；策略如在收盘后形成信号，应在下一可交易时点执行。",
    ],
    "tags": ["valuation", "bp", "cross_sectional", "neutralized"],
    "status": "research",
    "version": "1.2.0",
}
