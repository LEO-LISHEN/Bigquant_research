# -*- coding: utf-8 -*-
"""华泰换手率加权 N 月收益因子。"""

import time

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "wgt_return_nm"]


def _resolve_wgt_return_nm_data_window(resolved_params):
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


def calc_wgt_return_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=1,
    trading_days_per_month=21,
    show_progress=False,
    progress_every=200,
):
    """计算换手率加权 N 月收益因子。

    对最近 ``n_months * trading_days_per_month`` 个交易日的每日收益率，
    使用当日换手率加权：

    ``factor_t = sum(ret_{t-i} * turn_{t-i})
                  / sum(turn_{t-i})``

    原华泰 ``wgt_return_1m`` 对应 ``n_months=1``。数值越低的股票在
    原华泰研究中表现越好。本函数不负责数据查询或任何策略逻辑。

    参数
    ----
    data : pandas.DataFrame
        必须包含 date、instrument、close、turn，并包含历史预热数据。
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
        date、instrument、wgt_return_nm 三列。历史不足或权重和为 0
        的记录保留为 NaN。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    resolved_window = _resolve_wgt_return_nm_data_window(
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
            "wgt_return_nm 因子缺少字段："
            f"{sorted(missing_columns)}"
        )

    df = data.loc[:, ["date", "instrument", "close", "turn"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(
            "wgt_return_nm 因子的 date "
            "存在无法解析的日期或缺失值。"
        )
    if df["instrument"].isna().any():
        raise ValueError(
            "wgt_return_nm 因子的 instrument 不允许缺失。"
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
            "wgt_return_nm 因子输入存在重复的 "
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
                "wgt_return_nm 因子缺少目标日期的原始数据："
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
            f"[wgt_return_nm] 0/{total_instruments} 只股票 "
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

            numerator_prefix = np.concatenate(
                ([0.0], np.cumsum(numerator_input))
            )
            denominator_prefix = np.concatenate(
                ([0.0], np.cumsum(denominator_input))
            )
            numerator = numerator_prefix[1:].copy()
            denominator = denominator_prefix[1:].copy()

            if row_count > lookback_days:
                numerator[lookback_days:] = (
                    numerator_prefix[lookback_days + 1 :]
                    - numerator_prefix[
                        1 : row_count - lookback_days + 1
                    ]
                )
                denominator[lookback_days:] = (
                    denominator_prefix[lookback_days + 1 :]
                    - denominator_prefix[
                        1 : row_count - lookback_days + 1
                    ]
                )

            factor_values = np.full(
                row_count,
                np.nan,
                dtype=float,
            )
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
                    "wgt_return_nm": factor_values,
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
                    f"[wgt_return_nm] "
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
    "name": "wgt_return_nm",
    "func": calc_wgt_return_nm,
    "category": "momentum",
    "direction": -1,
    "description": (
        "最近 N 个月每日收益率以当日换手率加权；"
        "原华泰研究中因子值越低越好。"
    ),
    "formula": (
        "L=n_months*trading_days_per_month；"
        "ret_t=close_t/close_{t-1}-1；"
        "factor_t=sum_{i=0}^{L-1}"
        "(ret_{t-i}*turn_{t-i})"
        "/sum_{i=0}^{L-1}(turn_{t-i})。"
    ),
    "input_schema": {
        "required": {
            "date": {
                "dtype": "datetime64[ns] 或可解析日期",
                "meaning": "日频观测日期及目标因子截面日期。",
            },
            "instrument": {
                "dtype": "string",
                "meaning": (
                    "证券唯一标识；同一 date + instrument "
                    "不允许重复。"
                ),
            },
            "close": {
                "dtype": "float",
                "meaning": (
                    "与因子定义一致的复权收盘价；"
                    "整个回看窗口必须使用同一复权口径。"
                ),
            },
            "turn": {
                "dtype": "float",
                "meaning": "当日换手率，用作每日收益率权重。",
            },
        },
        "conditional": {},
    },
    "parameters": {
        "target_dates": {
            "default": None,
            "accepted_values": "单个日期或日期序列。",
            "effect": (
                "指定实际输出截面；None 表示输出 data 中全部日期。"
            ),
            "changes_data_requirements": True,
        },
        "as_of_date": {
            "default": None,
            "accepted_values": "可解析日期或 None。",
            "effect": "全局信息截止日，晚于该日的数据不参与计算。",
            "changes_data_requirements": False,
        },
        "n_months": {
            "default": 1,
            "accepted_values": "正整数。",
            "effect": "改变收益回看长度。",
            "changes_data_requirements": True,
        },
        "trading_days_per_month": {
            "default": 21,
            "accepted_values": "正整数。",
            "effect": (
                "将月份换算为交易日；改变实际回看交易日数。"
            ),
            "changes_data_requirements": True,
        },
        "show_progress": {
            "default": False,
            "accepted_values": [True, False],
            "effect": "仅控制进度显示，不改变计算结果。",
            "changes_data_requirements": False,
        },
        "progress_every": {
            "default": 200,
            "accepted_values": "正整数。",
            "effect": "进度刷新间隔，单位为股票数。",
            "changes_data_requirements": False,
        },
    },
    "data_window": {
        "resolver": _resolve_wgt_return_nm_data_window,
        "default": {
            "lookback_trading_days": 21,
            "requires_target_date_data": True,
            "minimum_history_observations": 21,
            "preheating_required": True,
            "insufficient_window_behavior": (
                "单只股票在目标日前不足 21 个历史交易日时，"
                "该股票目标日因子值输出 NaN。"
            ),
        },
        "resolver_notes": (
            "实际窗口由 n_months * trading_days_per_month 决定；"
            "loader 必须使用本次 resolved_factor_params 调用 resolver。"
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
        "wgt_return_nm": {
            "dtype": "float64",
            "meaning": (
                "换手率加权收益；数值越低，"
                "按原华泰研究定义越优。"
            ),
        },
    },
    "usage_notes": [
        "n_months=1 对应原始 wgt_return_1m。",
        "因子层不负责市值分组、行业剔除、ST/停牌过滤或选股。",
        (
            "策略和研究层应剔除 NaN 因子值；"
            "不应将预热不足的结果填为 0。"
        ),
    ],
    "best_practice": {
        "instance_name": "wgt_return_1m",
        "parameters": {
            "n_months": 1,
            "trading_days_per_month": 21,
        },
        "description": (
            "当前最佳实践实例为 1 个月换手率加权收益因子。"
        ),
    },
    "pit_notes": [
        (
            "只使用目标日及以前的 close 和 turn，"
            "不使用任何未来行情。"
        ),
        (
            "若使用目标日收盘价形成信号，最早应在下一可交易时点执行。"
        ),
        (
            "不同数据源的复权方式可能不同；迁移和结果对照时必须"
            "保持相同的价格复权口径。"
        ),
    ],
    "references": [
        "华泰动量类因子研究：wgt_return_1m 旧 notebook。",
    ],
    "tags": [
        "momentum",
        "turnover_weighted",
        "parameterized_window",
    ],
    "status": "research",
    "version": "1.1.0",
}
