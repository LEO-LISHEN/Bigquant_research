# -*- coding: utf-8 -*-
"""N 月平均换手率因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "turnover_mean_nm"]


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

    window = n_months * trading_days_per_month
    return {
        "lookback_trading_days": window - 1,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": window > 1,
        "insufficient_window_behavior": (
            "不设置人为最低有效日门槛；目标日换手率无效/非正，或截至"
            "目标日窗口内不存在有效正换手率时输出 NaN。"
        ),
    }


def _prepare(data, target_dates, as_of_date):
    """校验并整理与数据源无关的标准输入面板。"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")

    required = ["date", "instrument", "turn"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"turnover_mean_nm 缺少字段：{missing}。")

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


def calc_turnover_mean_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=1,
    trading_days_per_month=21,
    show_progress=False,
    progress_every=20,
):
    """计算过去 N 月平均日换手率。

    历史窗口包含目标日。因子层只依据换手率本身是否有效；停牌、涨跌停
    等交易限制属于策略执行层约束，不再造成因子暴露缺失。
    """
    del progress_every

    window_info = _resolve_data_window(
        {
            "n_months": n_months,
            "trading_days_per_month": trading_days_per_month,
        }
    )
    window = window_info["lookback_trading_days"] + 1
    started_at = time.perf_counter()

    df, targets = _prepare(data, target_dates, as_of_date)
    if df.empty or targets.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if show_progress:
        print(
            "\r"
            f"[turnover_mean_nm] [1/2] 计算 {window} 日滚动平均换手率...",
            end="",
            flush=True,
        )

    try:
        turn = pd.to_numeric(df["turn"], errors="coerce")
        valid_turn = np.isfinite(turn) & (turn > 0)
        rolling_mean = turn.where(valid_turn).groupby(
            df["instrument"],
            sort=False,
        ).transform(
            lambda series: series.rolling(
                window,
                min_periods=1,
            ).mean()
        )

        # 保持“目标日必须有有效换手率”的口径，但不引入任何交易限制字段。
        df["turnover_mean_nm"] = rolling_mean.where(valid_turn)

        result = df.loc[df["date"].isin(targets), OUTPUT_COLUMNS].copy()
        if show_progress:
            print(
                "\r"
                f"[turnover_mean_nm] [2/2] 完成 | {len(result):,} 条输出 "
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
    "name": "turnover_mean_nm",
    "func": calc_turnover_mean_nm,
    "factor_type": "base",
    "candidate_instances": {
        "1m": {"n_months": 1, "trading_days_per_month": 21},
        "3m": {"n_months": 3, "trading_days_per_month": 21},
        "6m": {"n_months": 6, "trading_days_per_month": 21},
        "12m": {"n_months": 12, "trading_days_per_month": 21},
    },
    "category": "liquidity",
    "direction": 0,
    "description": "过去 N 月的平均日换手率。",
    "formula": "turnover_mean_t = mean(turn_{t-W+1:t})。",
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
            "meaning": "滚动窗口月数。",
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
        "turnover_mean_nm": {
            "dtype": "float64",
            "meaning": "过去 N 月平均日换手率。",
        },
    },
    "usage_notes": (
        "窗口不足时使用截至目标日实际可得的有效换手率，不设置人为最低"
        "观察日门槛。仅当目标日换手率有限且大于 0 时输出值；停牌、"
        "涨跌停和下单限制不参与因子缺失判定，应由策略层处理。"
    ),
    "pit_notes": "只使用目标日及以前的换手率，滚动窗口包含目标日。",
}


FACTOR_INFO = """# N 月平均换手率

该因子衡量股票在最近 N 个月的平均交易活跃度。窗口包含目标日，并使用窗口内实际可得的有效正换手率，不以固定最少观察天数排除新股。

因子只依赖换手率本身；停牌、涨跌停和下单限制属于策略执行层。若目标日换手率无效或非正，因子输出缺失值。
"""
