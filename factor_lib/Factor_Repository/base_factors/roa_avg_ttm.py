# -*- coding: utf-8 -*-
"""平均总资产净利率（TTM）。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "roa_avg_ttm"]
REQUIRED_COLUMNS = {"date", "instrument", "ttm_roa_avg"}


def _normalize_target_dates(target_dates, available_dates):
    if target_dates is None:
        return pd.DatetimeIndex(available_dates).unique().sort_values()
    if isinstance(target_dates, (str, pd.Timestamp)):
        target_dates = [target_dates]
    else:
        try:
            target_dates = list(target_dates)
        except TypeError:
            target_dates = [target_dates]
    target_index = pd.DatetimeIndex(
        pd.to_datetime(target_dates, errors="raise")
    ).normalize().unique().sort_values()
    if target_index.empty:
        return target_index
    missing = target_index.difference(
        pd.DatetimeIndex(available_dates).unique()
    )
    if not missing.empty:
        preview = [value.strftime("%Y-%m-%d") for value in missing[:5]]
        raise ValueError("roa_avg_ttm 缺少目标日期的原始数据：" + str(preview))
    return target_index


def calc_roa_avg_ttm(
    data,
    target_dates=None,
    as_of_date=None,
    show_progress=False,
    progress_every=20,
):
    """计算 roa_avg_ttm。

    factor_t = ttm_roa_avg_t。
    因子层只计算因子暴露，不读取数据、不构造标签，也不处理交易逻辑。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if (
        not isinstance(progress_every, (int, np.integer))
        or isinstance(progress_every, (bool, np.bool_))
        or progress_every <= 0
    ):
        raise ValueError("progress_every 必须是正整数。")

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError("roa_avg_ttm 因子缺少字段：" + str(sorted(missing_columns)))

    df = data.loc[:, ["date", "instrument", "ttm_roa_avg"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("roa_avg_ttm 的 date 存在无法解析的日期或缺失值。")
    if df["instrument"].isna().any():
        raise ValueError("roa_avg_ttm 的 instrument 不允许缺失。")
    duplicated = df.duplicated(["date", "instrument"], keep=False)
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["date", "instrument"]]
            .head(5).astype(str).to_dict("records")
        )
        raise ValueError("roa_avg_ttm 输入存在重复 date + instrument：" + str(examples))

    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        if pd.isna(cutoff):
            raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= cutoff.normalize()].copy()
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    target_index = _normalize_target_dates(target_dates, df["date"].unique())
    if target_index.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    started_at = time.perf_counter()
    if show_progress:
        print(
            "\r[roa_avg_ttm] 计算目标截面 | "
            + str(len(target_index)) + " 个日期 | 已耗时 0.0s",
            end="", flush=True,
        )
    try:
        values = pd.to_numeric(df["ttm_roa_avg"], errors="coerce")
        result = pd.DataFrame({
            "date": df["date"].to_numpy(),
            "instrument": df["instrument"].to_numpy(),
            "roa_avg_ttm": values.to_numpy(dtype=float),
        })
        result = result.loc[result["date"].isin(target_index)]
        return result.sort_values(
            ["date", "instrument"], kind="mergesort"
        ).reset_index(drop=True)
    finally:
        if show_progress:
            elapsed = time.perf_counter() - started_at
            print(
                "\r[roa_avg_ttm] 计算完成 | "
                + str(len(target_index))
                + " 个目标截面 | 已耗时 " + format(elapsed, ".1f") + "s"
            )


FACTOR = {
    "name": "roa_avg_ttm",
    "func": calc_roa_avg_ttm,
    "factor_type": "base",
    "category": "quality",
    "direction": 1,
    "description": "平均总资产净利率（TTM）。",
    "formula": "factor_t = ttm_roa_avg_t。",
    "input_schema": {
        "required": {
            "date": {"dtype": "datetime64[ns] 或可解析日期", "frequency": "daily", "meaning": "日频点时观测日期及目标因子截面日期。"},
            "instrument": {"dtype": "string", "frequency": "daily", "meaning": "证券唯一标识；同一 date + instrument 不允许重复。"},
            "ttm_roa_avg": {"dtype": "float", "frequency": "financial", "meaning": "按点时口径可得的平均总资产净利率（TTM）。"},
        },
        "conditional": {},
    },
    "parameters": {
        "target_dates": {
            "default": None,
            "accepted_values": "单个日期、日期序列或 None。",
            "effect": "指定实际输出截面；None 表示输出 data 中全部日期。",
            "changes_data_requirements": True,
        },
        "as_of_date": {
            "default": None,
            "accepted_values": "可解析日期或 None。",
            "effect": "全局信息截止日，晚于该日的数据不参与计算。",
            "changes_data_requirements": False,
        },
        "show_progress": {
            "default": False,
            "accepted_values": [True, False],
            "effect": "仅控制进度显示，不改变因子结果。",
            "changes_data_requirements": False,
        },
        "progress_every": {
            "default": 20,
            "accepted_values": "正整数。",
            "effect": "与全库公共接口保持一致；本因子为向量化计算。",
            "changes_data_requirements": False,
        },
    },
    "data_window": {
        "lookback_trading_days": 0,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": False,
        "insufficient_window_behavior": "不适用；只需要目标日点时数据。",
    },
    "output_schema": {
        "date": {"dtype": "datetime64[ns]", "meaning": "目标因子截面日期。"},
        "instrument": {"dtype": "string", "meaning": "证券唯一标识。"},
        "roa_avg_ttm": {"dtype": "float64", "meaning": "平均总资产净利率（TTM）。"},
    },
    "candidate_instances": [{"id": "default", "params": {}}],
    "usage_notes": [
        "因子层不负责截面去极值、标准化、中性化、股票池过滤或选股。",
        "研究层应按目标日点时股票池过滤，并自行处理缺失与极端值。",
    ],
    "pit_notes": [
        "财务字段必须由日频点时财务适配器提供，不能按报告期末日期回填。",
        "若目标日收盘后才形成信号，最早应在下一可交易时点执行订单。",
    ],
    "references": ["BigQuant 日频点时财务科目与财务指标字段。"],
    "tags": ["quality", "financial", "base_factor"],
    "status": "research",
    "version": "1.0.0",
}


FACTOR_INFO = """
## 计算逻辑

直接使用 TTM 平均 ROA，降低单季财务波动对质量评估的影响。

## 使用提示

传入包含目标日点时财务字段的数据，并可用 target_dates 限定输出截面。
该脚本不查询数据，也不进行中性化、标准化或行业处理。
""".strip()

