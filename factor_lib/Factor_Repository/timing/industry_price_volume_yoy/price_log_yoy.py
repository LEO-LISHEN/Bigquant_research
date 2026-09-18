# -*- coding: utf-8 -*-
"""申万一级行业指数价格对数同比因子。"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd


FACTOR_NAME = "industry_price_log_yoy"
OUTPUT_COLUMNS = ["date", "instrument", FACTOR_NAME]


def _normalize_target_dates(target_dates):
    if isinstance(target_dates, (str, pd.Timestamp, np.datetime64)):
        values = [target_dates]
    else:
        try:
            values = list(target_dates)
        except TypeError as exc:
            raise TypeError("target_dates 必须是日期或日期序列。") from exc

    result = pd.DatetimeIndex(
        pd.to_datetime(values, errors="raise")
    ).normalize().unique().sort_values()
    if result.empty:
        return result
    return result


def _validate_progress_arguments(show_progress, progress_every):
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if (
        not isinstance(progress_every, (int, np.integer))
        or isinstance(progress_every, (bool, np.bool_))
        or progress_every <= 0
    ):
        raise ValueError("progress_every 必须是正整数。")


def _render_progress(
    completed,
    total,
    instrument,
    started_at,
    stage,
):
    elapsed = time.perf_counter() - started_at
    percentage = completed / total if total else 1.0
    message = (
        f"\r[{FACTOR_NAME}] {stage} | "
        f"{completed}/{total} 个行业指数（{percentage:.1%}）"
    )
    if instrument is not None:
        message += f" | 当前：{instrument}"
    message += f" | 已耗时：{elapsed:.1f}s"
    if 0 < completed < total:
        remaining = elapsed / completed * (total - completed)
        message += f" | 预计剩余：{remaining:.1f}s"
    print(message.ljust(180), end="", flush=True)


def _prepare_input_data(data, target_dates, as_of_date):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")

    required_columns = {"date", "instrument", "industry_close"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"{FACTOR_NAME} 缺少输入字段：{missing_columns}。"
        )

    df = data.loc[:, ["date", "instrument", "industry_close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError(f"{FACTOR_NAME} 的 date 存在无效值或缺失值。")
    if df["instrument"].isna().any():
        raise ValueError(f"{FACTOR_NAME} 的 instrument 不允许缺失。")

    duplicated = df.duplicated(["date", "instrument"], keep=False)
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["date", "instrument"]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            f"{FACTOR_NAME} 的输入存在重复 date + instrument：{examples}。"
        )

    df["industry_close"] = pd.to_numeric(
        df["industry_close"], errors="coerce"
    )

    if as_of_date is not None:
        try:
            cutoff = pd.Timestamp(as_of_date).normalize()
        except (TypeError, ValueError) as exc:
            raise ValueError("as_of_date 必须是可解析日期。") from exc
        if pd.isna(cutoff):
            raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= cutoff].copy()

    if df.empty:
        if target_dates is None:
            return df, pd.DatetimeIndex([])
        requested = _normalize_target_dates(target_dates)
        if requested.empty:
            return df, requested
        raise ValueError(
            f"{FACTOR_NAME} 在 as_of_date 截断后没有可用原始数据。"
        )

    df = df.sort_values(["instrument", "date"], kind="mergesort")
    df["month"] = df["date"].dt.to_period("M")

    # 研报逐月计算；同一自然月仅使用该行业指数当月最后一个可得交易日。
    monthly = (
        df.groupby(["instrument", "month"], sort=False, group_keys=False)
        .tail(1)
        .copy()
    )
    monthly = monthly.sort_values(["instrument", "date"], kind="mergesort")

    global_month_ends = (
        df.groupby("month", sort=False)["date"].max().sort_values()
    )
    available_targets = pd.DatetimeIndex(global_month_ends.to_numpy())

    if target_dates is None:
        targets = available_targets
    else:
        targets = _normalize_target_dates(target_dates)
        if targets.empty:
            return monthly, targets
        missing = targets.difference(available_targets)
        if not missing.empty:
            examples = [date.strftime("%Y-%m-%d") for date in missing[:5]]
            raise ValueError(
                f"{FACTOR_NAME} 的 target_dates 必须是输入数据中每月最后一个"
                f"交易日；无效日期：{examples}。"
            )

    return monthly, targets


def _calculate_one_industry(industry_data, target_set):
    """对一个行业指数的月末价格计算严格的 12 个自然月同比。"""
    industry_data = industry_data.sort_values("date", kind="mergesort").copy()
    prior = industry_data.loc[:, ["month", "industry_close"]].copy()
    prior["month"] = prior["month"] + 12
    prior = prior.rename(columns={"industry_close": "prior_close"})

    result = industry_data.merge(
        prior,
        on="month",
        how="left",
        validate="one_to_one",
    )
    close = result["industry_close"].to_numpy(dtype=float)
    prior_close = result["prior_close"].to_numpy(dtype=float)
    valid = (
        np.isfinite(close)
        & np.isfinite(prior_close)
        & (close > 0)
        & (prior_close > 0)
    )
    values = np.full(len(result), np.nan, dtype=float)
    values[valid] = np.log(close[valid]) - np.log(prior_close[valid])

    result[FACTOR_NAME] = values
    result = result.loc[
        result["date"].isin(target_set),
        ["date", "instrument", FACTOR_NAME],
    ]
    return result


def calc_industry_price_log_yoy(
    data,
    target_dates=None,
    as_of_date=None,
    industry_indices=None,
    show_progress=False,
    progress_every=20,
):
    """计算申万一级行业指数的价格对数同比。

    对每个行业指数，在月末计算：

    ``ln(P_t) - ln(P_{t-12})``

    其中 ``P_t`` 为该自然月最后一个交易日的行业指数收盘价。该函数只
    计算独立的行业价格指标；不输出仓位、不判断价格同比是否改善，也不
    与成交量因子组合。

    ``industry_indices`` 仅供 Loader 限制数据源中的行业指数范围使用，
    不改变已传入数据上的因子公式。
    """
    del industry_indices
    _validate_progress_arguments(show_progress, progress_every)
    monthly, targets = _prepare_input_data(data, target_dates, as_of_date)
    if monthly.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    target_set = set(targets)
    grouped = list(monthly.groupby("instrument", sort=False))
    total = len(grouped)
    started_at = time.perf_counter()
    result_parts = []

    if show_progress:
        _render_progress(
            0,
            total,
            None,
            started_at,
            "开始计算月末价格对数同比",
        )

    try:
        for position, (instrument, industry_data) in enumerate(
            grouped,
            start=1,
        ):
            result_parts.append(
                _calculate_one_industry(industry_data, target_set)
            )
            should_refresh = (
                position == 1
                or position % progress_every == 0
                or position == total
            )
            if show_progress and should_refresh:
                _render_progress(
                    position,
                    total,
                    instrument,
                    started_at,
                    "已完成行业指数计算",
                )
    finally:
        if show_progress:
            print()

    result_parts = [part for part in result_parts if not part.empty]
    if not result_parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.concat(result_parts, ignore_index=True)
    duplicated = result.duplicated(["date", "instrument"], keep=False)
    if duplicated.any():
        raise RuntimeError(f"{FACTOR_NAME} 输出出现重复主键。")
    return result.sort_values(
        ["date", "instrument"], kind="mergesort"
    ).reset_index(drop=True)


FACTOR = {
    "name": FACTOR_NAME,
    "func": calc_industry_price_log_yoy,
    "factor_type": "base",
    "factor_role": "timing",
    "primary_data_domain": "industry_daily",
    "candidate_instances": {"default": {}},
    "category": "industry_price_volume_timing",
    "direction": 1,
    "description": (
        "申万一级行业指数月末收盘价相对 12 个自然月前的对数变化；"
        "数值越高表示行业价格水平的同比增幅越高。"
    ),
    "formula": "industry_price_log_yoy_t = ln(P_t) - ln(P_{t-12})",
    "input_schema": {
        "required": {
            "date": {
                "dtype": "datetime64[ns] 或可解析日期",
                "frequency": "daily",
                "data_domain": "industry_daily",
                "meaning": "行业指数交易日。",
            },
            "instrument": {
                "dtype": "string",
                "frequency": "daily",
                "data_domain": "industry_daily",
                "meaning": "申万一级行业指数代码，例如 801010.SWI。",
            },
            "industry_close": {
                "dtype": "float64",
                "frequency": "daily",
                "data_domain": "industry_daily",
                "meaning": "申万一级行业指数日收盘价。",
            },
        },
        "conditional": {},
    },
    "parameters": {
        "target_dates": {
            "default": None,
            "accepted_values": "每月最后一个交易日的日期序列或 None",
            "effect": "限定输出的月末行业截面。",
            "changes_data_requirements": False,
        },
        "as_of_date": {
            "default": None,
            "accepted_values": "可解析日期或 None",
            "effect": "全局信息截止日。",
            "changes_data_requirements": False,
        },
        "industry_indices": {
            "default": None,
            "accepted_values": "行业指数代码、代码序列或 None",
            "effect": "供 industry_daily 适配器限制原始行业指数范围。",
            "changes_data_requirements": False,
        },
        "show_progress": {
            "default": False,
            "accepted_values": "bool",
            "effect": "显示按行业指数计数的单行进度。",
            "changes_data_requirements": False,
        },
        "progress_every": {
            "default": 20,
            "accepted_values": "正整数",
            "effect": "每处理多少个行业指数刷新一次进度。",
            "changes_data_requirements": False,
        },
    },
    "data_window": {
        "default": {
            "lookback_trading_days": 270,
            "requires_target_date_data": True,
            "minimum_history_observations": 13,
            "preheating_required": True,
            "insufficient_window_behavior": (
                "缺少目标月或严格 12 个自然月前月末价格时输出 NaN，"
                "不以前值或未来值补齐。"
            ),
        },
        "resolver_notes": "270 个交易日仅用于覆盖 12 个自然月预热，非公式参数。",
    },
    "output_schema": {
        "date": {
            "dtype": "datetime64[ns]",
            "meaning": "目标月最后一个交易日。",
        },
        "instrument": {
            "dtype": "string",
            "meaning": "申万一级行业指数代码。",
        },
        FACTOR_NAME: {
            "dtype": "float64",
            "meaning": "行业指数收盘价的 12 个月对数同比。",
        },
    },
    "usage_notes": [
        "这是行业级择时组件，不是个股横截面选股因子。",
        "target_dates 必须为输入数据中每月最后一个交易日。",
        "industry_indices=None 时由适配器读取所选日期内全部可得行业指数。",
    ],
    "pit_notes": [
        "仅使用目标月最后一个交易日及以前的行业指数收盘价。",
        "T 日收盘价计算出的指标最早在 T 日收盘后可得；交易执行时点由策略层另行定义。",
    ],
    "references": ["华泰证券《周期择时1：华泰价量择时模型》，2016-09-12"],
    "status": "research",
    "version": "1.0.0",
}


FACTOR_INFO = """
# 行业价格对数同比

使用申万一级行业指数的月末收盘价计算 `ln(P_t) - ln(P_{t-12})`。
该模块只输出独立行业指标，不在因子层形成仓位或与成交量指标组合。
"""
