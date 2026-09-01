# -*- coding: utf-8 -*-
"""BigQuant N 日频市值分组因子分位等权回测。

公开接口只有 ``run_defensive_market_cap_group_backtest``。策略负责：

1. 根据交易日历构造信号日、执行日和因子预热日期；
2. 通过 BigQuant 数据适配器一次性预存原始数据；
3. 在每个信号日动态调用因子函数计算单日截面；
4. 在各市值组内按原始因子值分位区间独立选股；
5. 使用 BigTrader 在下一交易日按指定价格等权调仓；
6. 保留信号、调仓、订单和成交审计对象，但默认不打印它们。

默认显式输出仅为 ``M.bigtrader.v35`` 生成的 BigQuant 回测图表。
"""

# 对外仅使用文件底部的 run_defensive_market_cap_group_backtest()。

from __future__ import annotations

import math
import time
import threading
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


_DATE_KEY_COLUMNS = ["date", "instrument"]
_RESERVED_FACTOR_PARAMS = {
    "data",
    "target_dates",
    "as_of_date",
    "show_progress",
    "progress_every",
}
_SUPPORTED_ORDER_PRICE_FIELDS = {"open", "close", "vwap"}


def _normalize_timestamp(value, parameter_name):
    try:
        timestamp = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{parameter_name} 必须是可解析日期：{value!r}"
        ) from exc
    if pd.isna(timestamp):
        raise ValueError(f"{parameter_name} 不允许为空。")
    return timestamp


def _normalize_positive_integer(value, parameter_name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{parameter_name} 必须是正整数。")
    return int(value)


def _normalize_factor_params(factor_params):
    if factor_params is None:
        return {}
    if not isinstance(factor_params, Mapping):
        raise TypeError("factor_params 必须是字典或 None。")

    result = dict(factor_params)
    conflicts = sorted(_RESERVED_FACTOR_PARAMS.intersection(result))
    if conflicts:
        raise ValueError(
            "factor_params 包含由策略统一控制的保留参数："
            f"{conflicts}。请从 factor_params 中移除。"
        )
    return result


def _normalize_selected_groups(selected_groups, group_count):
    if selected_groups is None:
        return list(range(1, group_count + 1))
    if isinstance(selected_groups, (str, bytes)) or not isinstance(
        selected_groups,
        Iterable,
    ):
        raise TypeError(
            "selected_market_cap_groups 必须是市值组编号序列或 None。"
        )

    normalized = []
    for value in selected_groups:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("市值组编号不能是 bool。")
        try:
            group_number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无效市值组编号：{value!r}") from exc
        if group_number != value and not (
            isinstance(value, str) and value.strip() == str(group_number)
        ):
            raise ValueError(f"市值组编号必须是整数：{value!r}")
        normalized.append(group_number)

    normalized = sorted(set(normalized))
    if not normalized:
        raise ValueError("selected_market_cap_groups 不能为空。")
    if normalized[0] < 1 or normalized[-1] > group_count:
        raise ValueError(
            "selected_market_cap_groups 必须位于 "
            f"1 至 {group_count} 之间。"
        )
    return normalized


def _normalize_quantile_range(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        raise ValueError(
            "factor_quantile_range 必须是 (lower, upper) 形式。"
        )
    lower = float(value[0])
    upper = float(value[1])
    if not 0 <= lower < upper <= 1:
        raise ValueError(
            "factor_quantile_range 必须满足 0 <= lower < upper <= 1。"
        )
    return lower, upper


def _normalize_price_field(value, parameter_name):
    if not isinstance(value, str):
        raise TypeError(f"{parameter_name} 必须是字符串。")
    normalized = value.strip().lower()
    if normalized not in _SUPPORTED_ORDER_PRICE_FIELDS:
        raise ValueError(
            f"{parameter_name} 仅支持 "
            f"{sorted(_SUPPORTED_ORDER_PRICE_FIELDS)}。"
        )
    return normalized


def _normalize_trading_costs(trading_costs):
    defaults = {
        "buy_cost": 0.0003,
        "sell_cost": 0.0003,
        "min_cost": 5.0,
        "tax_ratio": 0.0005,
    }
    if trading_costs is None:
        return defaults
    if not isinstance(trading_costs, Mapping):
        raise TypeError("trading_costs 必须是字典或 None。")

    result = dict(trading_costs)
    missing = sorted(set(defaults) - set(result))
    if missing:
        raise ValueError(f"trading_costs 缺少字段：{missing}")

    normalized = {}
    for name in defaults:
        value = float(result[name])
        if value < 0:
            raise ValueError(f"trading_costs[{name!r}] 不能为负数。")
        normalized[name] = value
    return normalized


def _normalize_instruments(values):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable):
        raise TypeError("股票代码必须是字符串或字符串序列。")

    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"无效股票代码：{value!r}")
        instrument = value.strip()
        if instrument not in normalized:
            normalized.append(instrument)
    if not normalized:
        raise ValueError("自定义股票集合不能为空。")
    return normalized


def _normalize_defensive_config(
    defensive_benchmark_index,
    defensive_ma_window,
    defensive_strategy_weight,
    defensive_compensation_instruments,
    market_index_code_mapping,
):
    """校验防御开关，并把指数代码统一解析成市场适配器别名。"""
    supplied_values = (
        defensive_benchmark_index,
        defensive_ma_window,
        defensive_strategy_weight,
        defensive_compensation_instruments,
    )
    if all(value is None for value in supplied_values):
        return None
    if any(value is None for value in supplied_values):
        raise ValueError(
            "启用防御配置时，defensive_benchmark_index、"
            "defensive_ma_window、defensive_strategy_weight 和 "
            "defensive_compensation_instruments 必须同时传入。"
        )

    if not isinstance(defensive_benchmark_index, str):
        raise TypeError("defensive_benchmark_index 必须是指数别名或指数代码字符串。")
    requested_index = defensive_benchmark_index.strip()
    if not requested_index:
        raise ValueError("defensive_benchmark_index 不能为空。")

    normalized_mapping = {
        str(name).strip().lower(): str(code).strip()
        for name, code in market_index_code_mapping.items()
    }
    code_to_name = {
        code.upper(): name
        for name, code in normalized_mapping.items()
    }
    market_index = normalized_mapping.get(requested_index.lower())
    if market_index is None:
        market_index = code_to_name.get(requested_index.upper())
    if market_index is None:
        supported = sorted(normalized_mapping)
        raise ValueError(
            "defensive_benchmark_index 不受市场数据适配器支持："
            f"{requested_index!r}。可传指数别名 {supported}，"
            "或其对应指数代码。"
        )

    ma_window = _normalize_positive_integer(
        defensive_ma_window,
        "defensive_ma_window",
    )
    strategy_weight = float(defensive_strategy_weight)
    if not np.isfinite(strategy_weight) or not 0 <= strategy_weight <= 1:
        raise ValueError(
            "defensive_strategy_weight 必须是 0 至 1 之间的仓位小数，"
            "例如 0.6 表示保留 60% 原策略仓位。"
        )

    return {
        "market_index": market_index,
        "market_index_code": normalized_mapping[market_index],
        "ma_window": ma_window,
        "strategy_weight": strategy_weight,
        "compensation_instruments": _normalize_instruments(
            defensive_compensation_instruments
        ),
    }


def _normalize_universe(universe):
    if universe == "all_a":
        return {"type": "all_a"}

    if isinstance(universe, (list, tuple, set, frozenset)):
        return {
            "type": "custom",
            "instruments": _normalize_instruments(universe),
        }

    if not isinstance(universe, Mapping):
        raise TypeError(
            "universe 必须是 'all_a'、股票代码集合或配置字典。"
        )

    universe_type = str(universe.get("type", "")).strip().lower()
    if universe_type == "all_a":
        return {"type": "all_a"}

    if universe_type in {"custom", "custom_list"}:
        return {
            "type": "custom",
            "instruments": _normalize_instruments(
                universe.get("instruments", [])
            ),
        }

    if universe_type == "index":
        index_codes = universe.get("index_codes")
        if index_codes is None:
            index_codes = universe.get("code")
        return {
            "type": "index",
            "index_codes": _normalize_instruments(index_codes),
        }

    raise ValueError(
        "universe['type'] 仅支持 all_a、index、custom。"
    )


def _quote_sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _query_trading_calendar(
    end_date,
    show_progress=False,
    started_at=None,
):
    """读取截至回测结束日的完整 A 股交易日历。"""
    import dai

    sql = f"""
    SELECT DISTINCT date
    FROM cn_stock_bar1d
    WHERE date <= '{end_date:%Y-%m-%d}'
    ORDER BY date
    """
    if started_at is None:
        started_at = time.perf_counter()
    calendar = _run_with_stage_heartbeat(
        lambda: dai.query(sql).df(),
        1,
        8,
        "读取A股交易日历",
        started_at,
        show_progress,
        current=f"截止{end_date:%Y-%m-%d}",
    )
    if calendar.empty or "date" not in calendar.columns:
        raise ValueError("未读取到有效的 A 股交易日历。")

    dates = pd.DatetimeIndex(
        pd.to_datetime(calendar["date"], errors="coerce")
    )
    if dates.isna().any():
        raise ValueError("交易日历中存在无法解析的日期。")
    dates = dates.normalize().unique().sort_values()
    if dates.empty:
        raise ValueError("A 股交易日历为空。")
    return dates


def _build_schedule(
    trading_calendar,
    start_date,
    end_date,
    rebalance_interval,
):
    """构造执行日及其前一交易日信号日。"""
    execution_candidates = trading_calendar[
        (trading_calendar >= start_date)
        & (trading_calendar <= end_date)
    ]
    if execution_candidates.empty:
        raise ValueError("回测区间内没有交易日。")

    execution_dates = execution_candidates[::rebalance_interval]
    calendar_positions = {
        date: position
        for position, date in enumerate(trading_calendar)
    }

    records = []
    for rebalance_number, execution_date in enumerate(
        execution_dates,
        start=1,
    ):
        position = calendar_positions[execution_date]
        if position == 0:
            raise ValueError(
                "首次执行日前没有可用交易日，无法形成前一日收盘信号。"
            )
        records.append(
            {
                "rebalance_number": rebalance_number,
                "signal_date": trading_calendar[position - 1],
                "execution_date": execution_date,
            }
        )

    schedule = pd.DataFrame(records)
    if schedule.empty:
        raise ValueError("未生成任何调仓计划。")
    return schedule


def _resolve_data_window(requirements):
    data_window = requirements.get("data_window", {})
    if not isinstance(data_window, Mapping):
        raise ValueError("FACTOR['data_window'] 必须是字典。")

    lookback = data_window.get("lookback_trading_days", 0)
    minimum_history = data_window.get("minimum_history_observations", 0)

    try:
        lookback = int(lookback)
        minimum_history = int(minimum_history)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "data_window 的历史窗口必须是非负整数。"
        ) from exc

    if lookback < 0 or minimum_history < 0:
        raise ValueError("因子历史窗口不能为负数。")
    return max(lookback, minimum_history)


def _build_factor_date_windows(
    schedule,
    trading_calendar,
    history_days,
):
    """为每个信号日构造包含目标日在内的原始数据日期窗口。"""
    calendar_positions = {
        date: position
        for position, date in enumerate(trading_calendar)
    }
    windows = {}
    all_dates = set()

    for signal_date in schedule["signal_date"]:
        position = calendar_positions.get(signal_date)
        if position is None:
            raise ValueError(
                f"信号日 {signal_date:%Y-%m-%d} 不在交易日历中。"
            )
        start_position = position - history_days
        if start_position < 0:
            raise ValueError(
                f"信号日 {signal_date:%Y-%m-%d} 前的历史交易日不足 "
                f"{history_days} 天。"
            )
        window = trading_calendar[start_position : position + 1]
        windows[signal_date] = window
        all_dates.update(window.tolist())

    return windows, pd.DatetimeIndex(sorted(all_dates))


def _query_index_universe(
    index_codes,
    signal_dates,
    show_progress=False,
    started_at=None,
):
    """读取一个或多个指数在各信号日的历史成分股。"""
    import dai

    index_sql = ", ".join(
        _quote_sql_literal(code) for code in index_codes
    )
    date_sql = ", ".join(
        _quote_sql_literal(date.strftime("%Y-%m-%d"))
        for date in signal_dates
    )
    sql = f"""
    SELECT
        date,
        instrument AS index_code,
        member_code AS instrument
    FROM cn_stock_index_component
    WHERE instrument IN ({index_sql})
      AND date IN ({date_sql})
    ORDER BY date, instrument, member_code
    """
    # BigQuant 对分区表要求不仅在 SQL 的 WHERE 中限定日期，还必须通过
    # filters 显式声明分区范围。SQL 中的 IN 仍负责只返回真正需要的信号日，
    # filters 只负责把底层扫描约束在首尾信号日之间。
    partition_filters = {
        "date": [
            signal_dates.min().strftime("%Y-%m-%d"),
            signal_dates.max().strftime("%Y-%m-%d"),
        ]
    }
    if started_at is None:
        started_at = time.perf_counter()
    panel = _run_with_stage_heartbeat(
        lambda: dai.query(sql, filters=partition_filters).df(),
        1,
        8,
        "读取历史指数成分股",
        started_at,
        show_progress,
        current=",".join(index_codes),
        detail=f"{len(signal_dates)}个信号日",
    )
    if panel.empty:
        raise ValueError(
            f"未读取到指数 {index_codes} 在信号日的历史成分股。"
        )

    required = {"date", "index_code", "instrument"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            f"指数成分股查询结果缺少字段：{sorted(missing)}"
        )

    panel = panel.copy()
    panel["date"] = pd.to_datetime(
        panel["date"],
        errors="coerce",
    ).dt.normalize()
    if panel["date"].isna().any():
        raise ValueError("指数成分股数据中存在无效日期。")
    if panel["instrument"].isna().any():
        raise ValueError("指数成分股数据中存在空股票代码。")

    panel = (
        panel[["date", "instrument"]]
        .drop_duplicates()
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    return panel


def _build_universe_panel(
    universe_config,
    signal_dates,
    show_progress=False,
    started_at=None,
):
    """返回动态股票池面板，以及可用于减少查询量的静态代码并集。"""
    universe_type = universe_config["type"]

    if universe_type == "all_a":
        return None, None

    if universe_type == "custom":
        instruments = universe_config["instruments"]
        panel = pd.MultiIndex.from_product(
            [signal_dates, instruments],
            names=["date", "instrument"],
        ).to_frame(index=False)
        return panel, instruments

    if universe_type == "index":
        panel = _query_index_universe(
            universe_config["index_codes"],
            signal_dates,
            show_progress=show_progress,
            started_at=started_at,
        )
        covered_dates = pd.DatetimeIndex(panel["date"].unique())
        missing_dates = signal_dates.difference(covered_dates)
        if not missing_dates.empty:
            preview = [
                date.strftime("%Y-%m-%d")
                for date in missing_dates[:5]
            ]
            raise ValueError(
                "部分信号日没有读取到指数历史成分股："
                f"{preview}"
            )
        instruments = sorted(panel["instrument"].unique())
        return panel, instruments

    raise RuntimeError(f"未处理的股票池类型：{universe_type}")


def _validate_panel(panel, panel_name, required_columns):
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{panel_name} 必须是 pandas.DataFrame。")
    missing = set(required_columns) - set(panel.columns)
    if missing:
        raise ValueError(
            f"{panel_name} 缺少字段：{sorted(missing)}"
        )

    result = panel.copy()
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    ).dt.normalize()
    if result["date"].isna().any():
        raise ValueError(f"{panel_name} 中存在无效日期。")
    if result["instrument"].isna().any():
        raise ValueError(f"{panel_name} 中存在空 instrument。")

    duplicated = result.duplicated(_DATE_KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = (
            result.loc[duplicated, _DATE_KEY_COLUMNS]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            f"{panel_name} 中存在重复 date + instrument：{examples}"
        )
    return result


def _to_bool(series, default=False):
    """将常见布尔、数值和文本标志统一为 bool。"""
    text = series.astype("string").str.strip().str.lower()
    result = text.isin({"1", "true", "t", "yes", "y", "是"})

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].ne(0)

    missing = series.isna() | text.isin({"", "nan", "none", "<na>"})
    result.loc[missing] = bool(default)
    return result.astype(bool)


def _resolve_factor_column(metadata, factor_name):
    output_schema = metadata.get("output_schema", {})
    if isinstance(output_schema, Mapping):
        factor_columns = [
            column
            for column in output_schema
            if column not in {"date", "instrument"}
        ]
        if factor_name in factor_columns:
            return factor_name
        if len(factor_columns) == 1:
            return factor_columns[0]
        if len(factor_columns) > 1:
            raise ValueError(
                f"因子 {factor_name!r} 声明了多个数值输出字段 "
                f"{factor_columns}，策略无法自动确定排序字段。"
            )
    return factor_name


def _prepare_signal_state(signal_panel, universe_panel):
    panel = signal_panel.copy()
    numeric_columns = [
        "total_market_cap",
        "is_risk_warning",
        "suspended",
        "volume",
    ]
    for column in numeric_columns:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="coerce",
        )

    panel["_is_st"] = panel["is_risk_warning"].fillna(1).ne(0)
    panel["_is_suspended"] = (
        panel["suspended"].fillna(1).ne(0)
        | panel["volume"].fillna(0).le(0)
    )

    if universe_panel is None:
        panel["in_universe"] = True
    else:
        panel = panel.merge(
            universe_panel.assign(in_universe=True),
            on=["date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        panel["in_universe"] = panel["in_universe"].fillna(False)

    return panel


def _execution_price(panel, field):
    if field == "vwap":
        amount = pd.to_numeric(panel["amount"], errors="coerce")
        volume = pd.to_numeric(panel["volume"], errors="coerce")
        return amount.where(volume > 0) / volume.where(volume > 0)
    return pd.to_numeric(panel[field], errors="coerce")


def _prepare_execution_state(
    execution_panel,
    buy_price_field,
    sell_price_field,
):
    panel = execution_panel.copy()
    for column in [
        "volume",
        "upper_limit",
        "lower_limit",
        "is_risk_warning",
        "suspended",
    ]:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="coerce",
        )

    panel["_buy_price"] = _execution_price(panel, buy_price_field)
    panel["_sell_price"] = _execution_price(panel, sell_price_field)
    panel["_is_st"] = panel["is_risk_warning"].fillna(1).ne(0)
    panel["_is_suspended"] = (
        panel["suspended"].fillna(1).ne(0)
        | panel["volume"].fillna(0).le(0)
    )

    valid_buy_price = panel["_buy_price"].notna() & panel["_buy_price"].gt(0)
    valid_sell_price = (
        panel["_sell_price"].notna()
        & panel["_sell_price"].gt(0)
    )
    valid_upper = (
        panel["upper_limit"].notna()
        & panel["upper_limit"].gt(0)
    )
    valid_lower = (
        panel["lower_limit"].notna()
        & panel["lower_limit"].gt(0)
    )

    at_or_above_upper = (
        panel["_buy_price"] > panel["upper_limit"]
    ) | np.isclose(
        panel["_buy_price"],
        panel["upper_limit"],
        rtol=0.0,
        atol=1e-8,
        equal_nan=False,
    )
    at_or_below_lower = (
        panel["_sell_price"] < panel["lower_limit"]
    ) | np.isclose(
        panel["_sell_price"],
        panel["lower_limit"],
        rtol=0.0,
        atol=1e-8,
        equal_nan=False,
    )

    panel["can_buy"] = (
        ~panel["_is_st"]
        & ~panel["_is_suspended"]
        & valid_buy_price
        & valid_upper
        & ~at_or_above_upper
    )
    panel["can_sell"] = (
        ~panel["_is_suspended"]
        & valid_sell_price
        & valid_lower
        & ~at_or_below_lower
    )

    def buy_reason(row):
        reasons = []
        if row["_is_st"]:
            reasons.append("st_or_risk_warning")
        if row["_is_suspended"]:
            reasons.append("suspended_or_zero_volume")
        if not (
            pd.notna(row["_buy_price"])
            and row["_buy_price"] > 0
        ):
            reasons.append("invalid_buy_price")
        if not (
            pd.notna(row["upper_limit"])
            and row["upper_limit"] > 0
        ):
            reasons.append("invalid_upper_limit")
        elif (
            pd.notna(row["_buy_price"])
            and row["_buy_price"] >= row["upper_limit"] - 1e-8
        ):
            reasons.append("at_or_above_upper_limit")
        return "|".join(reasons)

    def sell_reason(row):
        reasons = []
        if row["_is_suspended"]:
            reasons.append("suspended_or_zero_volume")
        if not (
            pd.notna(row["_sell_price"])
            and row["_sell_price"] > 0
        ):
            reasons.append("invalid_sell_price")
        if not (
            pd.notna(row["lower_limit"])
            and row["lower_limit"] > 0
        ):
            reasons.append("invalid_lower_limit")
        elif (
            pd.notna(row["_sell_price"])
            and row["_sell_price"] <= row["lower_limit"] + 1e-8
        ):
            reasons.append("at_or_below_lower_limit")
        return "|".join(reasons)

    panel["buy_blocked_reason"] = panel.apply(buy_reason, axis=1)
    panel["sell_blocked_reason"] = panel.apply(sell_reason, axis=1)

    return panel


def _select_cross_section(
    factor_cross_section,
    signal_state,
    factor_column,
    group_count,
    selected_groups,
    quantile_lower,
    quantile_upper,
):
    """完成一个信号日的过滤、分组和组内分位选股。"""
    cross_section = signal_state.merge(
        factor_cross_section[
            ["date", "instrument", factor_column]
        ],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    cross_section[factor_column] = pd.to_numeric(
        cross_section[factor_column],
        errors="coerce",
    )
    cross_section["_eligible"] = (
        cross_section["in_universe"]
        & ~cross_section["_is_st"]
        & ~cross_section["_is_suspended"]
        & cross_section["total_market_cap"].notna()
        & cross_section["total_market_cap"].gt(0)
        & cross_section[factor_column].notna()
        & np.isfinite(cross_section[factor_column])
    )
    eligible = cross_section.loc[cross_section["_eligible"]].copy()

    group_populations = {}
    selected_counts = {}
    if eligible.empty:
        return eligible, {
            "candidate_count": len(cross_section),
            "eligible_count": 0,
            "actual_group_count": 0,
            "group_populations": group_populations,
            "selected_counts": selected_counts,
        }

    actual_group_count = min(group_count, len(eligible))
    eligible = eligible.sort_values(
        ["total_market_cap", "instrument"],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible["_market_cap_rank"] = np.arange(1, len(eligible) + 1)
    eligible["market_cap_group"] = (
        pd.qcut(
            eligible["_market_cap_rank"],
            q=actual_group_count,
            labels=False,
        ).astype(int) + 1
    )

    selected_parts = []
    for group_number in range(1, actual_group_count + 1):
        group = eligible.loc[
            eligible["market_cap_group"] == group_number
        ].sort_values(
            [factor_column, "instrument"],
            kind="mergesort",
        ).reset_index(drop=True)

        group_populations[group_number] = len(group)
        group["factor_quantile"] = (
            np.arange(1, len(group) + 1) / len(group)
        )

        selected = group.iloc[0:0].copy()
        if group_number in selected_groups:
            start_index = int(math.floor(len(group) * quantile_lower))
            end_index = int(math.ceil(len(group) * quantile_upper))
            end_index = min(
                len(group),
                max(end_index, start_index + 1),
            )
            selected = group.iloc[start_index:end_index].copy()
            if not selected.empty:
                selected_parts.append(selected)
        selected_counts[group_number] = len(selected)

    if selected_parts:
        selected = (
            pd.concat(selected_parts, ignore_index=True)
            .drop_duplicates("instrument")
            .reset_index(drop=True)
        )
        selected["target_weight"] = 1.0 / len(selected)
    else:
        selected = eligible.iloc[0:0].copy()
        selected["factor_quantile"] = pd.Series(dtype=float)
        selected["target_weight"] = pd.Series(dtype=float)

    return selected, {
        "candidate_count": len(cross_section),
        "eligible_count": len(eligible),
        "actual_group_count": actual_group_count,
        "group_populations": group_populations,
        "selected_counts": selected_counts,
    }


def _get_position_quantity(position):
    for name in ("current_qty", "amount", "quantity"):
        value = getattr(position, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _get_position_price(position):
    for name in (
        "last_price",
        "last_sale_price",
        "market_price",
        "cost_price",
    ):
        value = getattr(position, name, None)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                return value
    return np.nan


def _get_portfolio_value(context):
    value = getattr(context.portfolio, "portfolio_value", None)
    if value is None and hasattr(context, "get_portfolio_value"):
        value = context.get_portfolio_value()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value


def _current_position_weights(context):
    positions = context.get_account_positions()
    portfolio_value = _get_portfolio_value(context)
    weights = {}
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        return positions, weights

    for instrument, position in positions.items():
        quantity = _get_position_quantity(position)
        if quantity <= 0:
            continue
        price = _get_position_price(position)
        if np.isfinite(price) and price > 0:
            weights[instrument] = quantity * price / portfolio_value
        else:
            weights[instrument] = np.nan
    return positions, weights


def _get_first_attribute(obj, names, default=None):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _render_progress(
    stage_number,
    stage_total,
    stage,
    started_at,
    completed=None,
    total=None,
    current=None,
    detail="",
):
    elapsed = time.perf_counter() - started_at
    parts = [f"[市值分组回测] [{stage_number}/{stage_total}] {stage}"]
    if completed is not None and total:
        parts.append(f"{completed}/{total} ({completed / total:.1%})")
        if 0 < completed < total:
            remaining = elapsed / completed * (total - completed)
            parts.append(f"预计剩余 {remaining:.1f}s")
    if current is not None:
        parts.append(f"当前 {current}")
    if detail:
        parts.append(str(detail))
    parts.append(f"已耗时 {elapsed:.1f}s")
    print(
        "\r" + " | ".join(parts).ljust(220),
        end="",
        flush=True,
    )


def _run_with_stage_heartbeat(
    action,
    stage_number,
    stage_total,
    stage,
    started_at,
    show_progress,
    current=None,
    detail="",
    interval_seconds=2.0,
):
    """单个阻塞阶段运行时定时刷新存活状态。"""
    if not show_progress:
        return action()

    stop_event = threading.Event()

    def heartbeat():
        while not stop_event.wait(interval_seconds):
            _render_progress(
                stage_number,
                stage_total,
                f"{stage}（仍在运行）",
                started_at,
                current=current,
                detail=detail,
            )

    worker = threading.Thread(
        target=heartbeat,
        name="market-cap-backtest-progress",
        daemon=True,
    )
    worker.start()
    try:
        return action()
    finally:
        stop_event.set()
        worker.join(timeout=max(interval_seconds, 0.1))


def run_defensive_market_cap_group_backtest(
    start_date,
    end_date,
    rebalance_interval,
    universe,
    factor_name,
    market_cap_group_count=15,
    selected_market_cap_groups=None,
    factor_quantile_range=(0.0, 0.1),
    factor_params=None,
    factor_panel_provider=None,
    defensive_benchmark_index=None,
    defensive_ma_window=None,
    defensive_strategy_weight=None,
    defensive_compensation_instruments=None,
    order_price_field_buy="open",
    order_price_field_sell="open",
    initial_cash=1_000_000,
    benchmark="000300.SH",
    trading_costs=None,
    slippage_value=None,
    volume_limit=0.025,
    weight_tolerance=1e-4,
    show_progress=False,
    progress_every=20,
):
    """运行 BigQuant N 日频市值分组因子分位等权回测。

    参数
    ----
    start_date, end_date : str 或 datetime
        回测日期范围。回测区间内第一个交易日是首次执行日；若 start_date
        不是交易日，则顺延至下一个交易日。
    rebalance_interval : int
        调仓间隔，按交易日间隔计算。首次执行日下标为 0，之后为
        0、N、2N、3N……
    universe : str、sequence[str] 或 dict
        ``"all_a"`` 表示全部 A 股；
        ``{"type": "index", "index_codes": [...]}`` 表示一个或多个指数的
        历史成分股并集；
        ``{"type": "custom", "instruments": [...]}`` 或代码集合表示固定
        股票范围。
    factor_name : str
        因子中心中登记的 FACTOR 名称。
    market_cap_group_count : int，默认 15
        按信号日总市值从小到大划分的等数量组数；第 1 组市值最小。
    selected_market_cap_groups : sequence[int] 或 None
        参与选股的市值组；None 表示全部市值组。
    factor_quantile_range : tuple[float, float]，默认 (0.0, 0.1)
        每个市值组内部按原始因子值从小到大的位置区间。区间不根据
        FACTOR['direction'] 自动翻转。
    factor_params : dict 或 None
        传给因子计算函数的内部参数。target_dates、as_of_date 和进度参数
        由策略统一控制，不能在这里覆盖。
    factor_panel_provider : callable 或 None
        可选的因子面板提供器。接收本次全部 ``signal_dates``，返回包含
        ``date``、``instrument`` 与因子输出列的 DataFrame。传入后，策略
        不再批量预存该因子的原始依赖数据，适用于需按信号日流式推理的
        机器学习因子；市值分组、选股、交易约束和 BigTrader 流程不变。
    defensive_benchmark_index : str 或 None
        防御开关监控的市场指数，可传市场适配器指数别名（如 ``csi_300``）
        或对应指数代码（如 ``000300.SH``）。四个 defensive 参数都为 None
        时，完全沿用原始市值分组策略。
    defensive_ma_window : int 或 None
        防御均线的交易日窗口；例如 23 表示使用信号日当日及此前 22 个
        交易日收盘价计算 MA23。
    defensive_strategy_weight : float 或 None
        当指数收盘价低于 MA 时保留给原市值分组组合的目标仓位，取值 0 至 1。
        未分配部分等权配置给 defensive_compensation_instruments。
    defensive_compensation_instruments : sequence[str] 或 None
        防御触发时用于承接降下仓位的自定义股票代码列表，列表内股票等权。
        不设置具体股票的默认值，必须由调用方显式传入。
    order_price_field_buy, order_price_field_sell : str
        BigTrader Bar 撮合买卖参考价，支持 open、close、vwap。
    initial_cash : float
        初始资金。
    benchmark : str 或 None
        BigQuant 回测基准。
    trading_costs : dict 或 None
        至少包含 buy_cost、sell_cost、min_cost、tax_ratio。
    slippage_value : float 或 None
        百分比滑点；None 表示不覆盖平台默认滑点设置。
    volume_limit : float
        单个订单最多占执行 Bar 成交量的比例。
    weight_tolerance : float
        目标权重和估算当前权重差值不超过该值时不提交调整订单。
    show_progress : bool
        是否显示单行进度。默认 False，不打印表格或审计结果。
    progress_every : int
        每处理多少个信号日刷新一次进度。

    返回
    ----
    dict
        ``performance`` 是 BigTrader 原始回测对象；其余 schedule、
        signals、rebalance_audit、execution_audit、order_audit、
        trade_audit 和 data_diagnostics 默认只保存在返回对象中。

    时序说明
    --------
    信号在执行日前一个交易日收盘后形成，因子调用固定使用
    ``target_dates=[signal_date]`` 和 ``as_of_date=signal_date``。
    BigTrader 在下一交易日按指定 Bar 价格撮合。执行日涨跌停、停牌、
    ST 和成交量状态仅作为订单可成交约束，不参与前一日选股，也不会触发
    替补股票或权重再分配。
    """
    started_at = time.perf_counter()

    start_date = _normalize_timestamp(start_date, "start_date")
    end_date = _normalize_timestamp(end_date, "end_date")
    if start_date > end_date:
        raise ValueError("start_date 不能晚于 end_date。")

    rebalance_interval = _normalize_positive_integer(
        rebalance_interval,
        "rebalance_interval",
    )
    market_cap_group_count = _normalize_positive_integer(
        market_cap_group_count,
        "market_cap_group_count",
    )
    progress_every = _normalize_positive_integer(
        progress_every,
        "progress_every",
    )
    selected_groups = _normalize_selected_groups(
        selected_market_cap_groups,
        market_cap_group_count,
    )
    quantile_lower, quantile_upper = _normalize_quantile_range(
        factor_quantile_range
    )
    factor_params = _normalize_factor_params(factor_params)
    if factor_panel_provider is not None and not callable(factor_panel_provider):
        raise TypeError("factor_panel_provider 必须是可调用对象或 None。")
    universe_config = _normalize_universe(universe)
    buy_price_field = _normalize_price_field(
        order_price_field_buy,
        "order_price_field_buy",
    )
    sell_price_field = _normalize_price_field(
        order_price_field_sell,
        "order_price_field_sell",
    )
    trading_costs = _normalize_trading_costs(trading_costs)

    if not isinstance(factor_name, str) or not factor_name.strip():
        raise ValueError("factor_name 必须是非空字符串。")
    factor_name = factor_name.strip()

    initial_cash = float(initial_cash)
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash 必须大于 0。")
    volume_limit = float(volume_limit)
    if not 0 < volume_limit <= 1:
        raise ValueError("volume_limit 必须满足 0 < volume_limit <= 1。")
    weight_tolerance = float(weight_tolerance)
    if weight_tolerance < 0:
        raise ValueError("weight_tolerance 不能为负数。")
    if slippage_value is not None:
        slippage_value = float(slippage_value)
        if slippage_value < 0:
            raise ValueError("slippage_value 不能为负数。")

    from factor_lib.common.data_adapters.bigquant_adapters.daily import (
        load_daily_raw_data,
    )
    from factor_lib.common.data_adapters.bigquant_adapters.market_daily import (
        MARKET_INDEX_CODE_MAPPING,
        load_market_daily_raw_data,
    )
    from factor_lib.common.data_adapters.bigquant_adapters.loader import (
        get_factor_data_requirements,
        get_factor_metadata,
        load_factor_raw_data,
    )
    from factor_lib.factor_hub.get_factor import get_factor

    defensive_config = _normalize_defensive_config(
        defensive_benchmark_index=defensive_benchmark_index,
        defensive_ma_window=defensive_ma_window,
        defensive_strategy_weight=defensive_strategy_weight,
        defensive_compensation_instruments=(
            defensive_compensation_instruments
        ),
        market_index_code_mapping=MARKET_INDEX_CODE_MAPPING,
    )

    if show_progress:
        _render_progress(
            1,
            8,
            "读取交易日历并生成调仓计划",
            started_at,
            current=f"{start_date:%Y-%m-%d} 至 {end_date:%Y-%m-%d}",
        )

    trading_calendar = _query_trading_calendar(
        end_date,
        show_progress=show_progress,
        started_at=started_at,
    )
    schedule = _build_schedule(
        trading_calendar=trading_calendar,
        start_date=start_date,
        end_date=end_date,
        rebalance_interval=rebalance_interval,
    )
    signal_dates = pd.DatetimeIndex(schedule["signal_date"])
    execution_dates = pd.DatetimeIndex(schedule["execution_date"])

    defensive_signal_by_date = None
    if defensive_config is not None:
        calendar_positions = {
            date: position
            for position, date in enumerate(trading_calendar)
        }
        first_signal_date = signal_dates.min()
        first_signal_position = calendar_positions.get(first_signal_date)
        if first_signal_position is None:
            raise ValueError("交易日历中缺少首个防御信号日。")
        required_history = defensive_config["ma_window"] - 1
        if first_signal_position < required_history:
            raise ValueError(
                "首个信号日前的交易日不足以计算防御均线："
                f"需要 MA{defensive_config['ma_window']} 的完整历史窗口。"
            )

        defensive_data_start = trading_calendar[
            first_signal_position - required_history
        ]
        if show_progress:
            _render_progress(
                1,
                8,
                "读取防御基准并计算信号日均线",
                started_at,
                completed=0,
                total=1,
                current=defensive_config["market_index_code"],
                detail=(
                    f"MA{defensive_config['ma_window']}，"
                    f"{defensive_data_start:%Y-%m-%d} 至 "
                    f"{signal_dates.max():%Y-%m-%d}"
                ),
            )
        defensive_market_panel = load_market_daily_raw_data(
            standard_fields=["market_close"],
            market_index=defensive_config["market_index"],
            start_date=defensive_data_start,
            end_date=signal_dates.max(),
            show_progress=show_progress,
        )
        defensive_market_panel = _validate_panel(
            defensive_market_panel,
            "defensive_market_panel",
            ["date", "market_index", "market_close"],
        )
        defensive_market_panel = defensive_market_panel.loc[
            defensive_market_panel["market_index"]
            == defensive_config["market_index"],
            ["date", "market_close"],
        ].copy()
        defensive_market_panel["market_close"] = pd.to_numeric(
            defensive_market_panel["market_close"],
            errors="coerce",
        )
        if defensive_market_panel["market_close"].isna().any():
            raise ValueError("防御基准收盘价存在缺失或非数值数据。")
        defensive_market_panel = defensive_market_panel.drop_duplicates(
            subset=["date"],
            keep=False,
        ).set_index("date")
        defensive_dates = trading_calendar[
            (trading_calendar >= defensive_data_start)
            & (trading_calendar <= signal_dates.max())
        ]
        defensive_close = defensive_market_panel["market_close"].reindex(
            defensive_dates
        )
        if defensive_close.isna().any():
            missing_dates = defensive_close.index[defensive_close.isna()]
            raise ValueError(
                "防御基准缺少交易日收盘价："
                f"{[date.strftime('%Y-%m-%d') for date in missing_dates[:5]]}"
            )
        defensive_ma = defensive_close.rolling(
            window=defensive_config["ma_window"],
            min_periods=defensive_config["ma_window"],
        ).mean()
        defensive_frame = pd.DataFrame(
            {
                "market_close": defensive_close.reindex(signal_dates),
                "market_ma": defensive_ma.reindex(signal_dates),
            },
            index=signal_dates,
        )
        if defensive_frame.isna().any(axis=None):
            raise ValueError("防御信号日未能计算完整的基准均线。")
        defensive_signal_by_date = {
            row.Index: {
                "is_defensive": bool(
                    row.market_close < row.market_ma
                ),
                "market_close": float(row.market_close),
                "market_ma": float(row.market_ma),
            }
            for row in defensive_frame.itertuples()
        }
        if show_progress:
            _render_progress(
                1,
                8,
                "防御信号准备完成",
                started_at,
                completed=1,
                total=1,
                detail=(
                    f"触发{sum(item['is_defensive'] for item in defensive_signal_by_date.values())}"
                    f"/{len(defensive_signal_by_date)}个信号日"
                ),
            )
    if show_progress:
        _render_progress(
            1,
            8,
            "调仓计划生成完成",
            started_at,
            completed=1,
            total=1,
            detail=f"{len(schedule)}个信号/执行日组合",
        )

    metadata = get_factor_metadata(factor_name)
    factor_column = _resolve_factor_column(metadata, factor_name)

    universe_panel, load_instruments = _build_universe_panel(
        universe_config,
        signal_dates,
        show_progress=show_progress,
        started_at=started_at,
    )
    if defensive_config is not None and load_instruments is not None:
        # 自定义/指数股票池需要额外读取补偿股票的信号日、执行日状态；
        # 不把它们并入 universe_panel，因此因子选股范围完全不变。
        load_instruments = sorted(
            set(load_instruments).union(
                defensive_config["compensation_instruments"]
            )
        )

    if factor_panel_provider is None:
        requirements = get_factor_data_requirements(
            factor_name,
            factor_params,
        )
        # loader 已经把 FACTOR 默认值与用户 factor_params 合并完成。
        # 数据窗口解析和最终因子计算必须共用这一份参数，避免预热口径与
        # 实际计算口径不一致。target_dates、as_of_date 和进度参数仍由
        # 策略统一控制，因此不从 resolved_factor_params 中重复传入。
        resolved_factor_params = {
            name: value
            for name, value in requirements[
                "resolved_factor_params"
            ].items()
            if name not in _RESERVED_FACTOR_PARAMS
        }
        history_days = _resolve_data_window(requirements)
        factor_date_windows, factor_dates = _build_factor_date_windows(
            schedule=schedule,
            trading_calendar=trading_calendar,
            history_days=history_days,
        )
        if show_progress:
            _render_progress(
                2,
                8,
                "预存因子原始数据",
                started_at,
                completed=0,
                total=1,
                current=factor_name,
                detail=f"{len(factor_dates)}个所需日期",
            )

        factor_raw_bundle = load_factor_raw_data(
            factor_name=factor_name,
            dates=factor_dates,
            factor_params=resolved_factor_params,
            instruments=load_instruments,
            show_progress=show_progress,
        )
        factor_raw_data = _validate_panel(
            factor_raw_bundle.get_security_daily(),
            "factor_raw_data",
            ["date", "instrument"],
        )
        factor_raw_bundle = factor_raw_bundle.with_domain(
            "security_daily",
            factor_raw_data,
            key_columns=("date", "instrument"),
        )
        provided_factor_panel = None
        factor_loading_mode = "raw_dependency_preload"
        if show_progress:
            _render_progress(
                2,
                8,
                "因子原始数据预存完成",
                started_at,
                completed=1,
                total=1,
                current=factor_name,
                detail=f"股票域{len(factor_raw_data):,}行",
            )
    else:
        # 流式提供器自行负责因子的特征读取、逐批计算与内存回收；策略层
        # 只持有最终 date/instrument/factor-value 面板。
        requirements = None
        resolved_factor_params = {}
        history_days = None
        factor_date_windows = None
        factor_raw_bundle = None
        factor_raw_data = pd.DataFrame(columns=["date", "instrument"])
        factor_loading_mode = "factor_panel_provider"
        if show_progress:
            _render_progress(
                2,
                8,
                "调用流式因子面板提供器",
                started_at,
                completed=0,
                total=1,
                current=factor_name,
                detail=f"{len(signal_dates)}个信号日",
            )
        provided_factor_panel = _run_with_stage_heartbeat(
            lambda: factor_panel_provider(signal_dates.copy()),
            2,
            8,
            "流式因子面板生成",
            started_at,
            show_progress,
            current=factor_name,
            detail=f"{len(signal_dates)}个信号日",
        )
        provided_factor_panel = _validate_panel(
            provided_factor_panel,
            "factor_panel_provider 输出",
            ["date", "instrument", factor_column],
        )
        unexpected_dates = pd.DatetimeIndex(
            provided_factor_panel["date"].unique()
        ).difference(signal_dates)
        if len(unexpected_dates) > 0:
            raise ValueError(
                "factor_panel_provider 返回了非信号日数据："
                f"{[date.strftime('%Y-%m-%d') for date in unexpected_dates[:5]]}"
            )
        missing_dates = signal_dates.difference(
            pd.DatetimeIndex(provided_factor_panel["date"].unique())
        )
        if len(missing_dates) > 0:
            raise ValueError(
                "factor_panel_provider 缺少信号日因子截面："
                f"{[date.strftime('%Y-%m-%d') for date in missing_dates[:5]]}"
            )
        if show_progress:
            _render_progress(
                2,
                8,
                "流式因子面板生成完成",
                started_at,
                completed=1,
                total=1,
                current=factor_name,
                detail=f"{len(provided_factor_panel):,}行",
            )

    signal_fields = [
        "total_market_cap",
        "is_risk_warning",
        "suspended",
        "volume",
    ]
    if show_progress:
        _render_progress(
            3,
            8,
            "预存信号日选股状态",
            started_at,
            completed=0,
            total=1,
            detail=f"{len(signal_dates)}个信号日",
        )
    signal_panel = load_daily_raw_data(
        standard_fields=signal_fields,
        dates=signal_dates,
        instruments=load_instruments,
        show_progress=show_progress,
    )
    signal_panel = _validate_panel(
        signal_panel,
        "signal_panel",
        ["date", "instrument", *signal_fields],
    )
    signal_panel = _prepare_signal_state(
        signal_panel,
        universe_panel,
    )
    if show_progress:
        _render_progress(
            3,
            8,
            "信号日选股状态准备完成",
            started_at,
            completed=1,
            total=1,
            detail=f"{len(signal_panel):,}行",
        )

    execution_fields = [
        "volume",
        "upper_limit",
        "lower_limit",
        "is_risk_warning",
        "suspended",
    ]
    for field in (buy_price_field, sell_price_field):
        if field == "vwap":
            execution_fields.extend(["amount", "volume"])
        else:
            execution_fields.append(field)
    execution_fields = list(dict.fromkeys(execution_fields))

    if show_progress:
        _render_progress(
            4,
            8,
            "预存执行日交易约束",
            started_at,
            completed=0,
            total=1,
            detail=f"{len(execution_dates)}个执行日",
        )
    execution_panel = load_daily_raw_data(
        standard_fields=execution_fields,
        dates=execution_dates,
        instruments=load_instruments,
        show_progress=show_progress,
    )
    execution_panel = _validate_panel(
        execution_panel,
        "execution_panel",
        ["date", "instrument", *execution_fields],
    )
    execution_panel = _prepare_execution_state(
        execution_panel,
        buy_price_field=buy_price_field,
        sell_price_field=sell_price_field,
    )
    if show_progress:
        _render_progress(
            4,
            8,
            "执行日交易约束准备完成",
            started_at,
            completed=1,
            total=1,
            detail=f"{len(execution_panel):,}行",
        )

    signal_state_by_date = {
        date: group.copy()
        for date, group in signal_panel.groupby("date", sort=False)
    }
    execution_state_map = {}
    execution_state_columns = [
        "date",
        "instrument",
        "can_buy",
        "can_sell",
        "buy_blocked_reason",
        "sell_blocked_reason",
        "_buy_price",
        "_sell_price",
    ]
    execution_state_rows = execution_panel[execution_state_columns]
    total_execution_rows = len(execution_state_rows)
    for position, row in enumerate(
        execution_state_rows.itertuples(index=False, name=None),
        start=1,
    ):
        (
            row_date,
            instrument,
            can_buy,
            can_sell,
            buy_blocked_reason,
            sell_blocked_reason,
            buy_price,
            sell_price,
        ) = row
        execution_state_map[(row_date, instrument)] = {
            "can_buy": bool(can_buy),
            "can_sell": bool(can_sell),
            "buy_blocked_reason": buy_blocked_reason,
            "sell_blocked_reason": sell_blocked_reason,
            "buy_price": buy_price,
            "sell_price": sell_price,
        }
        if show_progress and (
            position == 1
            or position % 5000 == 0
            or position == total_execution_rows
        ):
            _render_progress(
                5,
                8,
                "建立执行约束快速索引",
                started_at,
                completed=position,
                total=total_execution_rows,
                current=f"{row_date:%Y-%m-%d} {instrument}",
            )

    schedule_by_signal = {
        row.signal_date: {
            "rebalance_number": int(row.rebalance_number),
            "execution_date": row.execution_date,
        }
        for row in schedule.itertuples(index=False)
    }
    provided_factor_by_date = (
        {
            date: panel.copy()
            for date, panel in provided_factor_panel.groupby("date", sort=False)
        }
        if provided_factor_panel is not None
        else None
    )

    engine_instruments = sorted(signal_panel["instrument"].unique())
    if not engine_instruments:
        raise ValueError("没有可供 BigTrader 订阅的股票代码。")

    signal_records = []
    rebalance_records = []
    execution_records = []
    order_records = []
    trade_records = []
    completed_signal_count = 0
    successful_signal_count = 0
    defensive_rebalance_count = 0
    if show_progress:
        _render_progress(
            6,
            8,
            "启动 BigTrader 原生回测",
            started_at,
            completed=0,
            total=len(schedule),
            detail=(
                f"{len(schedule)}个信号，"
                f"{len(engine_instruments):,}只候选股票"
            ),
        )

    from bigmodule import M

    def initialize(context):
        from bigtrader.finance.commission import PerOrder

        context.set_commission(
            PerOrder(
                buy_cost=trading_costs["buy_cost"],
                sell_cost=trading_costs["sell_cost"],
                min_cost=trading_costs["min_cost"],
                tax_ratio=trading_costs["tax_ratio"],
            )
        )
        if slippage_value is not None:
            context.set_slippage_value(
                slippage_type=2,
                slippage_value=slippage_value,
            )

    def handle_data(context, data):
        nonlocal completed_signal_count
        nonlocal successful_signal_count
        nonlocal defensive_rebalance_count

        signal_date = pd.Timestamp(data.current_dt).normalize()
        schedule_item = schedule_by_signal.get(signal_date)
        if schedule_item is None:
            return

        completed_signal_count += 1
        execution_date = schedule_item["execution_date"]
        rebalance_number = schedule_item["rebalance_number"]

        if show_progress:
            _render_progress(
                6,
                8,
                (
                    "回测中：读取流式因子信号"
                    if provided_factor_by_date is not None
                    else "回测中：计算单日因子信号"
                ),
                started_at,
                completed=completed_signal_count - 1,
                total=len(schedule),
                current=f"{signal_date:%Y-%m-%d}",
                detail=f"第{rebalance_number}次调仓",
            )

        defensive_state = (
            defensive_signal_by_date.get(signal_date)
            if defensive_signal_by_date is not None
            else None
        )
        defensive_active = bool(
            defensive_state is not None
            and defensive_state["is_defensive"]
        )
        # 将“本次确已触发防御”收窄为非空配置，既表达运行时前置条件，
        # 也避免静态检查器把 defensive_config 误判为 Optional。
        active_defensive_config = (
            defensive_config if defensive_active else None
        )
        try:
            if provided_factor_by_date is None:
                # 该分支仅由原始批量加载路径进入；断言同时为静态检查器
                # 明确收窄 Optional 类型，不影响任何运行时计算。
                assert factor_date_windows is not None
                assert factor_raw_bundle is not None
                required_dates = factor_date_windows[signal_date]
                for domain_name in factor_raw_bundle.domain_names:
                    missing_dates = factor_raw_bundle.missing_dates(
                        domain_name,
                        required_dates,
                    )
                    if len(missing_dates) > 0:
                        missing_text = [
                            date.strftime("%Y-%m-%d")
                            for date in missing_dates[:5]
                        ]
                        raise ValueError(
                            f"数据域 {domain_name!r} 的因子预热窗口"
                            f"缺少日期：{missing_text}"
                        )

                factor_input = factor_raw_bundle.select_dates(required_dates)
                factor_cross_section = _run_with_stage_heartbeat(
                    lambda: get_factor(
                        factor_name,
                        factor_input,
                        target_dates=[signal_date],
                        as_of_date=signal_date,
                        show_progress=False,
                        progress_every=progress_every,
                        **resolved_factor_params,
                    ),
                    6,
                    8,
                    "回测中：计算信号日因子",
                    started_at,
                    show_progress,
                    current=f"{signal_date:%Y-%m-%d}，{factor_name}",
                )
            else:
                factor_cross_section = provided_factor_by_date.get(signal_date)
                if factor_cross_section is None or factor_cross_section.empty:
                    raise ValueError("流式因子面板缺少当前信号日的有效截面。")
            if show_progress:
                _render_progress(
                    6,
                    8,
                    "回测中：市值分组和因子分位选股",
                    started_at,
                    completed=completed_signal_count - 1,
                    total=len(schedule),
                    current=f"{signal_date:%Y-%m-%d}",
                    detail=f"因子结果{len(factor_cross_section):,}行",
                )
            factor_cross_section = _validate_panel(
                factor_cross_section,
                f"{factor_name}@{signal_date:%Y-%m-%d}",
                ["date", "instrument", factor_column],
            )

            signal_state = signal_state_by_date.get(signal_date)
            if signal_state is None or signal_state.empty:
                raise ValueError("信号日缺少选股状态数据。")

            selected, statistics = _select_cross_section(
                factor_cross_section=factor_cross_section,
                signal_state=signal_state,
                factor_column=factor_column,
                group_count=market_cap_group_count,
                selected_groups=selected_groups,
                quantile_lower=quantile_lower,
                quantile_upper=quantile_upper,
            )
            if active_defensive_config is not None and not selected.empty:
                # 原选股总权重由 100% 缩放至 defensive_strategy_weight；
                # 未分配部分随后才等权转入调用方指定的补偿股票。
                selected = selected.copy()
                selected["target_weight"] *= active_defensive_config[
                    "strategy_weight"
                ]
            successful_signal_count += 1
            status = "ok" if not selected.empty else "empty_target"
            error_message = ""

        except Exception as exc:
            selected = pd.DataFrame()
            statistics = {
                "candidate_count": 0,
                "eligible_count": 0,
                "actual_group_count": 0,
                "group_populations": {},
                "selected_counts": {},
            }
            status = "skipped_error"
            error_message = f"{type(exc).__name__}: {exc}"
            defensive_active = False
            active_defensive_config = None

        rebalance_records.append(
            {
                "rebalance_number": rebalance_number,
                "signal_date": signal_date,
                "execution_date": execution_date,
                "status": status,
                "error_message": error_message,
                **statistics,
                "target_count": len(selected),
            }
        )

        if status == "skipped_error":
            if show_progress:
                _render_progress(
                    6,
                    8,
                    "回测中：当前信号因异常跳过",
                    started_at,
                    completed=completed_signal_count,
                    total=len(schedule),
                    current=f"{signal_date:%Y-%m-%d}",
                    detail=error_message,
                )
            return

        target_weights = {}
        if not selected.empty:
            selected = selected.copy()
            selected["signal_date"] = signal_date
            selected["execution_date"] = execution_date
            selected["factor_name"] = factor_name
            selected["factor_value"] = selected[factor_column]
            selected["rebalance_number"] = rebalance_number
            signal_records.extend(
                selected[
                    [
                        "rebalance_number",
                        "signal_date",
                        "execution_date",
                        "instrument",
                        "total_market_cap",
                        "market_cap_group",
                        "factor_name",
                        "factor_value",
                        "factor_quantile",
                        "target_weight",
                    ]
                ].to_dict("records")
            )
            target_weights = dict(
                zip(
                    selected["instrument"],
                    selected["target_weight"],
                )
            )

        if active_defensive_config is not None:
            defensive_rebalance_count += 1
            compensation_weight = (
                1.0 - active_defensive_config["strategy_weight"]
            ) / len(active_defensive_config["compensation_instruments"])
            for instrument in active_defensive_config["compensation_instruments"]:
                target_weights[instrument] = (
                    float(target_weights.get(instrument, 0.0))
                    + compensation_weight
                )

        positions, current_weights = _current_position_weights(context)
        holding_instruments = {
            instrument
            for instrument, position in positions.items()
            if _get_position_quantity(position) > 0
        }
        all_instruments = holding_instruments.union(target_weights)

        sell_intents = []
        buy_intents = []
        for instrument in sorted(all_instruments):
            current_weight = current_weights.get(instrument, 0.0)
            target_weight = float(target_weights.get(instrument, 0.0))

            if not np.isfinite(current_weight):
                execution_records.append(
                    {
                        "rebalance_number": rebalance_number,
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "instrument": instrument,
                        "current_weight": np.nan,
                        "target_weight": target_weight,
                        "order_direction": "unknown",
                        "tradable": False,
                        "blocked_reason": "unable_to_estimate_current_weight",
                        "order_attempted": False,
                        "order_submitted": False,
                        "submit_result": None,
                    }
                )
                continue

            delta = target_weight - current_weight
            if abs(delta) <= weight_tolerance:
                continue
            intent = (
                instrument,
                current_weight,
                target_weight,
            )
            if delta < 0:
                sell_intents.append(intent)
            else:
                buy_intents.append(intent)

        # 先提交卖出/减仓，再提交买入/加仓。
        total_intents = len(sell_intents) + len(buy_intents)
        completed_intents = 0
        for direction, intents in (
            ("sell", sell_intents),
            ("buy", buy_intents),
        ):
            for instrument, current_weight, target_weight in intents:
                state = execution_state_map.get(
                    (execution_date, instrument)
                )
                if state is None:
                    tradable = False
                    blocked_reason = "missing_execution_data"
                elif direction == "buy":
                    tradable = state["can_buy"]
                    blocked_reason = (
                        "" if tradable else state["buy_blocked_reason"]
                    )
                else:
                    tradable = state["can_sell"]
                    blocked_reason = (
                        "" if tradable else state["sell_blocked_reason"]
                    )

                submit_result = None
                if tradable:
                    submit_result = context.order_target_percent(
                        instrument,
                        target_weight,
                    )
                    try:
                        submit_succeeded = int(submit_result) >= 0
                    except (TypeError, ValueError):
                        submit_succeeded = submit_result is not None
                    if not submit_succeeded:
                        blocked_reason = (
                            f"order_submit_failed:{submit_result}"
                        )
                else:
                    submit_succeeded = False

                execution_records.append(
                    {
                        "rebalance_number": rebalance_number,
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "instrument": instrument,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "order_direction": direction,
                        "tradable": bool(tradable),
                        "blocked_reason": blocked_reason,
                        "order_attempted": bool(tradable),
                        "order_submitted": bool(submit_succeeded),
                        "submit_result": submit_result,
                    }
                )
                completed_intents += 1
                if show_progress and (
                    completed_intents == 1
                    or completed_intents % 50 == 0
                    or completed_intents == total_intents
                ):
                    _render_progress(
                        6,
                        8,
                        "回测中：提交调仓订单",
                        started_at,
                        completed=completed_signal_count - 1,
                        total=len(schedule),
                        current=f"{signal_date:%Y-%m-%d}",
                        detail=(
                            f"订单{completed_intents}/{total_intents}，"
                            f"{direction}:{instrument}"
                        ),
                    )

        if show_progress and (
            completed_signal_count == 1
            or completed_signal_count % progress_every == 0
            or completed_signal_count == len(schedule)
        ):
            _render_progress(
                6,
                8,
                "回测中：调仓信号处理完成",
                started_at,
                completed=completed_signal_count,
                total=len(schedule),
                current=f"{signal_date:%Y-%m-%d}",
                detail=f"选中{len(selected)}只，订单意图{total_intents}个",
            )

    def handle_order(context, order):
        order_records.append(
            {
                "trading_day": _get_first_attribute(
                    order,
                    ["trading_day", "insert_date"],
                ),
                "instrument": _get_first_attribute(
                    order,
                    ["instrument", "symbol"],
                ),
                "direction": str(
                    _get_first_attribute(order, ["direction"], "")
                ),
                "order_qty": _get_first_attribute(
                    order,
                    ["order_qty", "quantity"],
                    np.nan,
                ),
                "filled_qty": _get_first_attribute(
                    order,
                    ["filled_qty", "trade_qty"],
                    np.nan,
                ),
                "order_price": _get_first_attribute(
                    order,
                    ["order_price", "price"],
                    np.nan,
                ),
                "order_status": str(
                    _get_first_attribute(
                        order,
                        ["order_status", "status"],
                        "",
                    )
                ),
                "status_msg": _get_first_attribute(
                    order,
                    ["status_msg", "message"],
                    "",
                ),
                "order_key": _get_first_attribute(
                    order,
                    ["order_key", "order_id"],
                ),
            }
        )

    def handle_trade(context, trade):
        trade_records.append(
            {
                "trading_day": _get_first_attribute(
                    trade,
                    ["trading_day", "trade_date"],
                ),
                "trade_time": _get_first_attribute(
                    trade,
                    ["trade_time", "datetime"],
                ),
                "instrument": _get_first_attribute(
                    trade,
                    ["instrument", "symbol"],
                ),
                "direction": str(
                    _get_first_attribute(trade, ["direction"], "")
                ),
                "filled_qty": _get_first_attribute(
                    trade,
                    ["filled_qty", "trade_qty", "quantity"],
                    np.nan,
                ),
                "filled_price": _get_first_attribute(
                    trade,
                    ["filled_price", "trade_price", "price"],
                    np.nan,
                ),
                "filled_money": _get_first_attribute(
                    trade,
                    ["filled_money", "trade_amount", "amount"],
                    np.nan,
                ),
                "commission": _get_first_attribute(
                    trade,
                    ["commission", "fee"],
                    np.nan,
                ),
                "order_key": _get_first_attribute(
                    trade,
                    ["order_key", "order_id"],
                ),
            }
        )

    bigtrader_kwargs = {
        "data": {"instruments": engine_instruments},
        # 为了让首次执行日能够使用前一交易日信号，内部回测起点前移至首个信号日。
        "start_date": schedule["signal_date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initialize": initialize,
        "handle_data": handle_data,
        "handle_order": handle_order,
        "handle_trade": handle_trade,
        "capital_base": initial_cash,
        "frequency": "daily",
        "volume_limit": volume_limit,
        "order_price_field_buy": buy_price_field,
        "order_price_field_sell": sell_price_field,
        # 回调函数闭包中保存了本次调仓计划、因子数据和审计容器。
        # 必须关闭模块结果缓存，避免复用上一次运行的回测结果，导致
        # 当前回调未执行或审计记录仍为零。
        "m_cached": False,
    }
    if benchmark is not None:
        bigtrader_kwargs["benchmark"] = benchmark

    if show_progress:
        # 避免 BigTrader 原生日志接在单行进度文字之后。
        print()
    performance = M.bigtrader.v35(**bigtrader_kwargs)

    if show_progress:
        _render_progress(
            7,
            8,
            "BigTrader运行完成，整理审计结果",
            started_at,
            completed=completed_signal_count,
            total=len(schedule),
            detail=f"成功信号{successful_signal_count}个",
        )

    signals = pd.DataFrame(signal_records)
    rebalance_audit = pd.DataFrame(rebalance_records)
    execution_audit = pd.DataFrame(execution_records)
    order_audit = pd.DataFrame(order_records)
    trade_audit = pd.DataFrame(trade_records)

    if show_progress:
        _render_progress(
            8,
            8,
            "回测与审计结果整理完成",
            started_at,
            completed=1,
            total=1,
            detail=(
                f"信号{len(signals):,}条，订单{len(order_audit):,}条，"
                f"成交{len(trade_audit):,}条"
            ),
        )
        print()

    data_diagnostics = {
        "requested_start_date": start_date,
        "actual_first_execution_date": schedule["execution_date"].iloc[0],
        "engine_start_date": schedule["signal_date"].iloc[0],
        "end_date": end_date,
        "factor_history_days": history_days,
        "factor_loading_mode": factor_loading_mode,
        "resolved_factor_params": dict(
            resolved_factor_params
        ),
        "factor_raw_rows": len(factor_raw_data),
        "factor_domain_rows": (
            factor_raw_bundle.row_counts()
            if factor_raw_bundle is not None
            else {}
        ),
        "provided_factor_rows": (
            len(provided_factor_panel)
            if provided_factor_panel is not None
            else 0
        ),
        "signal_state_rows": len(signal_panel),
        "execution_state_rows": len(execution_panel),
        "engine_instrument_count": len(engine_instruments),
        "defensive_config": (
            {
                "benchmark_index": defensive_config["market_index_code"],
                "ma_window": defensive_config["ma_window"],
                "strategy_weight": defensive_config["strategy_weight"],
                "compensation_instruments": list(
                    defensive_config["compensation_instruments"]
                ),
            }
            if defensive_config is not None
            else None
        ),
        "defensive_rebalance_count": defensive_rebalance_count,
        "scheduled_rebalance_count": len(schedule),
        "processed_signal_count": completed_signal_count,
        "successful_signal_count": successful_signal_count,
        "total_runtime_seconds": time.perf_counter() - started_at,
    }

    # 不 print/display 以下对象；只有调用方主动索引返回值时才显示。
    return {
        "performance": performance,
        "schedule": schedule.copy(),
        "signals": signals,
        "rebalance_audit": rebalance_audit,
        "execution_audit": execution_audit,
        "order_audit": order_audit,
        "trade_audit": trade_audit,
        "factor_metadata": metadata,
        "factor_data_requirements": requirements,
        "data_diagnostics": data_diagnostics,
    }
