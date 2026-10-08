# -*- coding: utf-8 -*-
"""华泰指数衰减换手率加权 N 月收益因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "exp_wgt_return_nm"]


def _resolve_exp_wgt_return_nm_data_window(resolved_params):
    """根据本次完整因子参数返回确定的数据窗口。"""
    n_months = resolved_params.get("n_months", 6)
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


def calc_exp_wgt_return_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=6,
    trading_days_per_month=21,
    show_progress=False,
    progress_every=200,
):
    """计算指数衰减换手率加权 N 月收益因子。

    对最近 ``n_months * trading_days_per_month`` 个交易日的每日收益率，
    使用“当日换手率 × 指数衰减项”加权：

    ``decay_i = exp(-i / n_months / 4)``

    ``factor_t = sum(ret_{t-i} * turn_{t-i} * decay_i)
                  / sum(turn_{t-i} * decay_i)``

    其中 ``i=0`` 表示目标日，数值越低的股票在原华泰研究中表现越好。
    本函数只计算因子，不负责取数、股票池筛选、中性化、选股或回测。

    参数
    ----
    data : pandas.DataFrame
        必须包含 date、instrument、close、turn。应包含目标日以及计算窗口
        所需的历史预热数据。
    target_dates : 日期或日期序列，可选
        只输出这些目标截面。为 None 时输出 data 中全部日期。
    as_of_date : 日期，可选
        全局信息截止日；晚于该日期的数据不会参与计算。
    n_months : int，默认 6
        回看月数，必须为正整数。传入 3 即复现 exp_wgt_return_3m，
        传入 6 即复现 exp_wgt_return_6m。
    trading_days_per_month : int，默认 21
        每月折算的交易日数量，必须为正整数。
    show_progress : bool，默认 False
        是否使用终端单行刷新显示计算进度。
    progress_every : int，默认 200
        每处理多少只股票刷新一次进度，必须为正整数。

    返回
    ----
    pandas.DataFrame
        date、instrument、exp_wgt_return_nm 三列。历史不足或分母为 0
        的记录保留为 NaN，交由研究或策略层决定是否剔除。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    resolved_window = _resolve_exp_wgt_return_nm_data_window(
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

    required_columns = {"date", "instrument", "close", "turn"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "exp_wgt_return_nm 因子缺少字段："
            f"{sorted(missing_columns)}"
        )

    df = data.loc[:, ["date", "instrument", "close", "turn"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(
            "exp_wgt_return_nm 因子的 date "
            "存在无法解析的日期或缺失值。"
        )
    if df["instrument"].isna().any():
        raise ValueError(
            "exp_wgt_return_nm 因子的 instrument 不允许缺失。"
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
            "exp_wgt_return_nm 因子输入存在重复的 "
            f"date + instrument 记录：{examples}"
        )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")

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
                "exp_wgt_return_nm 因子缺少目标日期的原始数据："
                f"{preview}。请检查数据范围和 as_of_date。"
            )

    decay = np.exp(
        -np.arange(lookback_days, dtype=float)
        / float(n_months)
        / 4.0
    )

    df = df.sort_values(
        ["instrument", "date"],
        kind="mergesort",
        inplace=False,
    ).reset_index(drop=True)
    total_instruments = df["instrument"].nunique()
    result_parts = []
    started_at = time.perf_counter()

    if show_progress:
        print(
            "\r"
            f"[exp_wgt_return_nm] 0/{total_instruments} 只股票 "
            "| 0.0%",
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
                inplace=False,
            ).copy()
            close = stock_data["close"].to_numpy(dtype=float)
            turn = stock_data["turn"].to_numpy(dtype=float)
            row_count = len(stock_data)

            previous_close = np.empty(row_count, dtype=float)
            previous_close[:] = np.nan
            if row_count > 1:
                previous_close[1:] = close[:-1]

            daily_return = np.full(row_count, np.nan, dtype=float)
            valid_price = (
                np.isfinite(close)
                & np.isfinite(previous_close)
                & (previous_close != 0)
            )
            daily_return[valid_price] = (
                close[valid_price]
                / previous_close[valid_price]
                - 1.0
            )

            valid_turn = np.isfinite(turn)
            denominator_input = np.where(
                valid_turn,
                turn,
                0.0,
            )
            numerator_input = np.where(
                valid_turn & np.isfinite(daily_return),
                turn * daily_return,
                0.0,
            )

            # np.convolve 的核顺序正好对应：
            # 当前日权重 decay[0]、前一日 decay[1]，依次向前。
            numerator = np.convolve(
                numerator_input,
                decay,
                mode="full",
            )[:row_count]
            denominator = np.convolve(
                denominator_input,
                decay,
                mode="full",
            )[:row_count]

            factor_values = np.full(
                row_count,
                np.nan,
                dtype=float,
            )
            # L 日收益需要目标日前至少 L 个历史收盘价，
            # 因此第一个可计算位置是 position=L。
            history_ready = (
                np.arange(row_count) >= lookback_days
            )
            valid_denominator = (
                np.isfinite(denominator)
                & (denominator != 0)
            )
            calculable = history_ready & valid_denominator
            factor_values[calculable] = (
                numerator[calculable]
                / denominator[calculable]
            )

            stock_result = pd.DataFrame(
                {
                    "date": stock_data["date"].to_numpy(),
                    "instrument": instrument,
                    "exp_wgt_return_nm": factor_values,
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
                    f"[exp_wgt_return_nm] "
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
        inplace=False,
    ).reset_index(drop=True)


FACTOR = {
    "name": 'exp_wgt_return_nm',
    "func": calc_exp_wgt_return_nm,
    "factor_type": "base",
    "candidate_instances": {
        "3m": {"n_months": 3, "trading_days_per_month": 21},
        "6m": {"n_months": 6, "trading_days_per_month": 21},
    },
    "input_schema": {
        "required": {
            'date': {},
            'instrument': {},
            'close': {},
            'turn': {},
        },
        "conditional": {
        },
    },
    "parameters": {
        'target_dates': {"default": None},
        'as_of_date': {"default": None},
        'n_months': {"default": 6},
        'trading_days_per_month': {"default": 21},
        'show_progress': {"default": False},
        'progress_every': {"default": 200},
    },
    "data_window": {
        "resolver": _resolve_exp_wgt_return_nm_data_window,
        "default": {
            "lookback_trading_days": 126,
            "requires_target_date_data": True,
            "minimum_history_observations": 126,
            "preheating_required": True,
        },
    },
    "output_schema": {
        'date': {},
        'instrument': {},
        'exp_wgt_return_nm': {},
    },
}


FACTOR_INFO = """
# 指数衰减换手率加权收益（N 月）

以近 N 个月日收益率为基础，并按换手率和指数衰减权重加权。该因子沿用原研究口径：数值较低通常更优。

- **计算**：窗口由 `n_months × trading_days_per_month` 决定。
- **时点**：仅使用目标日及此前的复权收盘价和换手率；信号最早在目标日收盘后形成。
- **推荐实例**：`n_months=6`，即原 6 个月版本。
"""
