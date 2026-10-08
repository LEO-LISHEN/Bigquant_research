# -*- coding: utf-8 -*-
"""华泰 N 月区间收益因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "return_nm"]


def _resolve_return_nm_data_window(resolved_params):
    """根据本次完整因子参数返回确定的数据窗口。"""
    n_months = resolved_params.get("n_months", 1)
    trading_days_per_month = resolved_params.get(
        "trading_days_per_month",
        21,
    )
    if (
        not isinstance(n_months, (int, np.integer))
        or isinstance(n_months, (bool, np.bool_))
        or n_months <= 0
    ):
        raise ValueError("n_months 必须是正整数。")
    if (
        not isinstance(trading_days_per_month, (int, np.integer))
        or isinstance(trading_days_per_month, (bool, np.bool_))
        or trading_days_per_month <= 0
    ):
        raise ValueError("trading_days_per_month 必须是正整数。")

    lookback_days = int(n_months * trading_days_per_month)
    return {
        "lookback_trading_days": lookback_days,
        "requires_target_date_data": True,
        "minimum_history_observations": lookback_days,
        "preheating_required": True,
        "insufficient_window_behavior": (
            "单只股票在目标日前不足 L 个历史交易日时，"
            "该股票目标日因子值输出 NaN，不使用未来数据补足。"
        ),
    }


def calc_return_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=1,
    trading_days_per_month=21,
    show_progress=False,
    progress_every=200,
):
    """计算 N 月区间收益因子。

    ``factor_t = close_t / close_{t-L} - 1``

    其中 ``L = n_months * trading_days_per_month``。原华泰
    ``return_1m`` 对应 ``n_months=1``。数值越低的股票在原华泰
    研究中表现越好。本函数不负责数据查询或任何策略逻辑。

    参数
    ----
    data : pandas.DataFrame
        必须包含 date、instrument、close，并包含历史预热数据。
    target_dates : 日期或日期序列，可选
        只输出这些目标截面。为 None 时输出 data 中全部日期。
    as_of_date : 日期，可选
        全局信息截止日；晚于该日期的数据不会参与计算。
    n_months : int，默认 1
        回看月数，必须为正整数。
    trading_days_per_month : int，默认 21
        每月折算的交易日数量，必须为正整数。
    show_progress : bool，默认 False
        是否使用终端单行刷新显示计算进度。
    progress_every : int，默认 200
        每处理多少只股票刷新一次进度。

    返回
    ----
    pandas.DataFrame
        date、instrument、return_nm 三列。历史不足或价格无效的记录
        保留为 NaN。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    resolved_window = _resolve_return_nm_data_window(
        {
            "n_months": n_months,
            "trading_days_per_month": trading_days_per_month,
        }
    )
    lookback_days = resolved_window["lookback_trading_days"]
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if (
        not isinstance(progress_every, (int, np.integer))
        or isinstance(progress_every, (bool, np.bool_))
        or progress_every <= 0
    ):
        raise ValueError("progress_every 必须是正整数。")

    required_columns = {"date", "instrument", "close"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "return_nm 因子缺少字段："
            f"{sorted(missing_columns)}"
        )

    df = data.loc[:, ["date", "instrument", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(
            "return_nm 因子的 date "
            "存在无法解析的日期或缺失值。"
        )
    if df["instrument"].isna().any():
        raise ValueError(
            "return_nm 因子的 instrument 不允许缺失。"
        )

    duplicated = df.duplicated(
        ["date", "instrument"],
        keep=False,
    )
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["date", "instrument"]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            "return_nm 因子输入存在重复的 "
            f"date + instrument 记录：{examples}"
        )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    if as_of_date is not None:
        as_of_timestamp = pd.Timestamp(as_of_date)
        if pd.isna(as_of_timestamp):
            raise ValueError("as_of_date 必须是可解析日期。")
        df = df.loc[df["date"] <= as_of_timestamp].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if target_dates is None:
        target_date_index = pd.DatetimeIndex(
            df["date"].unique()
        ).sort_values()
    else:
        if isinstance(target_dates, (str, pd.Timestamp)):
            target_dates = [target_dates]
        else:
            try:
                target_dates = list(target_dates)
            except TypeError:
                target_dates = [target_dates]

        target_date_index = pd.DatetimeIndex(
            pd.to_datetime(target_dates, errors="raise")
        ).unique().sort_values()
        if target_date_index.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        available_dates = pd.DatetimeIndex(df["date"].unique())
        missing_target_dates = target_date_index.difference(
            available_dates
        )
        if not missing_target_dates.empty:
            preview = [
                date.strftime("%Y-%m-%d")
                for date in missing_target_dates[:5]
            ]
            raise ValueError(
                "return_nm 因子缺少目标日期的原始数据："
                f"{preview}。请检查数据范围和 as_of_date。"
            )

    df = df.sort_values(
        ["instrument", "date"],
        kind="mergesort",
    ).reset_index(drop=True)
    total_instruments = df["instrument"].nunique()
    result_parts = []
    started_at = time.perf_counter()

    if show_progress:
        print(
            "\r"
            f"[return_nm] 0/{total_instruments} 只股票 | 0.0%",
            end="",
            flush=True,
        )

    try:
        grouped = df.groupby("instrument", sort=False)
        for position, (instrument, stock_data) in enumerate(
            grouped,
            start=1,
        ):
            stock_data = stock_data.sort_values(
                "date",
                kind="mergesort",
            ).copy()
            close = stock_data["close"].to_numpy(dtype=float)
            row_count = len(stock_data)

            historical_close = np.full(
                row_count,
                np.nan,
                dtype=float,
            )
            if row_count > lookback_days:
                historical_close[lookback_days:] = (
                    close[:-lookback_days]
                )

            factor_values = np.full(
                row_count,
                np.nan,
                dtype=float,
            )
            calculable = (
                np.isfinite(close)
                & np.isfinite(historical_close)
                & (historical_close != 0)
            )
            factor_values[calculable] = (
                close[calculable]
                / historical_close[calculable]
                - 1.0
            )

            stock_result = pd.DataFrame(
                {
                    "date": stock_data["date"].to_numpy(),
                    "instrument": instrument,
                    "return_nm": factor_values,
                }
            )
            stock_result = stock_result.loc[
                stock_result["date"].isin(target_date_index)
            ]
            if not stock_result.empty:
                result_parts.append(stock_result)

            should_refresh = (
                position == 1
                or position % progress_every == 0
                or position == total_instruments
            )
            if show_progress and should_refresh:
                elapsed = time.perf_counter() - started_at
                remaining = (
                    elapsed / position
                    * (total_instruments - position)
                )
                print(
                    "\r"
                    f"[return_nm] "
                    f"{position}/{total_instruments} 只股票 "
                    f"| {position / total_instruments:.1%} "
                    f"| 当前：{instrument} "
                    f"| 已耗时：{elapsed:.1f}s "
                    f"| 预计剩余：{remaining:.1f}s",
                    end="",
                    flush=True,
                )
    finally:
        if show_progress:
            print()

    if not result_parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.concat(result_parts, ignore_index=True)
    return result.sort_values(
        ["date", "instrument"],
        kind="mergesort",
    ).reset_index(drop=True)


FACTOR = {
    "name": 'return_nm',
    "func": calc_return_nm,
    "factor_type": "base",
    "candidate_instances": {"1m": {"n_months": 1, "trading_days_per_month": 21}},
    "input_schema": {
        "required": {
            'date': {},
            'instrument': {},
            'close': {},
        },
        "conditional": {
        },
    },
    "parameters": {
        'target_dates': {"default": None},
        'as_of_date': {"default": None},
        'n_months': {"default": 1},
        'trading_days_per_month': {"default": 21},
        'show_progress': {"default": False},
        'progress_every': {"default": 200},
    },
    "data_window": {
        "resolver": _resolve_return_nm_data_window,
        "default": {
            "lookback_trading_days": 21,
            "requires_target_date_data": True,
            "minimum_history_observations": 21,
            "preheating_required": True,
        },
    },
    "output_schema": {
        'date': {},
        'instrument': {},
        'return_nm': {},
    },
}


FACTOR_INFO = """
# 区间收益（N 月）

计算近 N 个月复权收盘价的区间收益，保留原研究中的排序口径：数值较低通常更优。

- **计算**：收益 = 目标日收盘价 / 窗口起点收盘价 - 1。
- **时点**：整个窗口必须使用一致的复权价格口径，且不使用未来行情。
- **推荐实例**：`n_months=1`，即原 1 个月版本。
"""
