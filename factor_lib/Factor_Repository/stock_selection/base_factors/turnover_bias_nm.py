# -*- coding: utf-8 -*-
"""N 月换手率偏离因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "turnover_bias_nm"]


def _resolve_data_window(params):
    """根据窗口参数声明因子所需的历史交易日数。"""
    n_months = params.get("n_months", 1)
    trading_days_per_month = params.get("trading_days_per_month", 21)

    if (
        not isinstance(n_months, int)
        or isinstance(n_months, bool)
        or n_months < 1
    ):
        raise ValueError("n_months 必须是正整数。")
    if (
        not isinstance(trading_days_per_month, int)
        or isinstance(trading_days_per_month, bool)
        or trading_days_per_month < 1
    ):
        raise ValueError("trading_days_per_month 必须是正整数。")

    return {
        "lookback_trading_days": n_months * trading_days_per_month,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": True,
        "insufficient_window_behavior": (
            "不设置人为最低有效日门槛；目标日前窗口中不存在任何"
            "有效正换手率基准，或目标日换手率无效/非正时输出 NaN。"
        ),
    }


def _prepare(data, target_dates, as_of_date):
    """校验并整理与数据源无关的标准输入面板。"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")

    required = ["date", "instrument", "turn"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"turnover_bias_nm 缺少字段：{missing}。")

    df = data.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any() or df["instrument"].isna().any():
        raise ValueError("date 或 instrument 包含无效值。")

    df["instrument"] = df["instrument"].astype(str)
    if df.duplicated(["date", "instrument"], keep=False).any():
        raise ValueError("data 存在重复 date + instrument。")

    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        if pd.isna(cutoff):
            raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= cutoff.normalize()].copy()

    available = pd.DatetimeIndex(df["date"].unique()).sort_values()
    if target_dates is None:
        targets = available
    else:
        values = (
            [target_dates]
            if isinstance(target_dates, (str, pd.Timestamp))
            else list(target_dates)
        )
        targets = (
            pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
            .normalize()
            .unique()
            .sort_values()
        )
        missing_dates = targets.difference(available)
        if not missing_dates.empty:
            preview = missing_dates[:5].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"缺少目标日原始数据：{preview}。")

    return (
        df.sort_values(["instrument", "date"], kind="mergesort")
        .reset_index(drop=True),
        targets,
    )


def calc_turnover_bias_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=1,
    trading_days_per_month=21,
    show_progress=False,
    progress_every=20,
):
    """计算当前换手率相对过去 N 月平均换手率的偏离。

    ``turnover_bias_t = turn_t / mean(turn_{t-W:t-1}) - 1``。

    因子层只依据换手率本身是否有效；停牌、涨跌停及成交限制属于
    策略执行层约束，不再作为因子缺失的判定条件。历史基准不含目标日，
    避免当天异常换手同时进入分子和分母。
    """
    del progress_every

    window_info = _resolve_data_window(
        {
            "n_months": n_months,
            "trading_days_per_month": trading_days_per_month,
        }
    )
    window = window_info["lookback_trading_days"]
    started_at = time.perf_counter()

    df, targets = _prepare(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if show_progress:
        print(
            "\r"
            f"[turnover_bias_nm] [1/2] 计算 {window} 日历史换手率基准...",
            end="",
            flush=True,
        )

    try:
        current_turn = pd.to_numeric(df["turn"], errors="coerce")
        valid_turn = np.isfinite(current_turn) & (current_turn > 0)

        # 无效换手率不参与历史均值；窗口允许使用截至目标日实际可得的
        # 有效历史观测，因此不把停牌/涨跌停等执行状态混入因子定义。
        historical_turn = current_turn.where(valid_turn)
        baseline = historical_turn.groupby(
            df["instrument"],
            sort=False,
        ).transform(
            lambda series: series.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

        calculable = valid_turn & np.isfinite(baseline) & (baseline > 0)
        values = pd.Series(np.nan, index=df.index, dtype=float)
        values.loc[calculable] = (
            current_turn.loc[calculable] / baseline.loc[calculable] - 1.0
        )
        df["turnover_bias_nm"] = values

        result = df.loc[df["date"].isin(targets), OUTPUT_COLUMNS].copy()
        if show_progress:
            print(
                "\r"
                f"[turnover_bias_nm] [2/2] 完成 | {len(result):,} 条输出 "
                f"| 耗时 {time.perf_counter() - started_at:.1f}s",
                end="",
                flush=True,
            )
        return result.sort_values(
            ["date", "instrument"],
            kind="mergesort",
        ).reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "turnover_bias_nm",
    "func": calc_turnover_bias_nm,
    "factor_type": "base",
    "candidate_instances": {
        "1m": {"n_months": 1, "trading_days_per_month": 21},
        "3m": {"n_months": 3, "trading_days_per_month": 21},
        "6m": {"n_months": 6, "trading_days_per_month": 21},
        "12m": {"n_months": 12, "trading_days_per_month": 21},
    },
    "category": "liquidity",
    "direction": 0,
    "description": "当前有效换手率相对过去 N 月平均换手率的偏离。",
    "formula": "turnover_bias_t = turn_t / mean(turn_{t-W:t-1}) - 1。",
    "input_schema": {
        "required": {
            "date": {},
            "instrument": {},
            "turn": {},
        },
        "conditional": {},
    },
    "parameters": {
        "n_months": {
            "default": 1,
            "range": "正整数",
            "meaning": "历史基准窗口月数。",
        },
        "trading_days_per_month": {
            "default": 21,
            "range": "正整数",
            "meaning": "月数到交易日窗口的换算口径。",
        },
        "target_dates": {
            "default": None,
            "meaning": "实际输出截面；None 为全部输入日期。",
        },
        "as_of_date": {"default": None, "meaning": "全局信息截止日。"},
        "show_progress": {
            "default": False,
            "meaning": "是否显示单行计算进度。",
        },
        "progress_every": {
            "default": 20,
            "meaning": "兼容统一接口；该计算为向量化滚动操作。",
        },
    },
    "data_window": {
        "resolver": _resolve_data_window,
        "default": _resolve_data_window({}),
    },
    "output_schema": {
        "date": {},
        "instrument": {},
        "turnover_bias_nm": {
            "dtype": "float64",
            "meaning": "当前换手率相对历史平均换手率的偏离率。",
        },
    },
    "usage_notes": (
        "历史基准不含目标日。仅当目标日换手率有限且大于 0、且此前存在"
        "有效正换手率基准时输出值；停牌、涨跌停等交易执行约束不参与"
        "因子缺失判定，应由策略层处理。"
    ),
    "pit_notes": "只使用目标日及以前的换手率；历史基准显式排除目标日自身。",
}


FACTOR_INFO = """# N 月换手率偏离

该因子衡量当前换手率相对自身近期平均水平的异常程度。正值代表成交活跃度高于历史常态，负值代表低于常态。

历史基准不包含当天，避免当天换手率同时进入分子和分母。因子只依赖换手率本身；停牌、涨跌停和下单限制由策略执行层单独处理。若目标日换手率无效，或此前没有任何有效正换手率基准，则输出缺失值。
"""
