# -*- coding: utf-8 -*-
"""BigQuant N 日频市值分组因子分位等权回测。

公开接口只有 ``run_market_cap_group_backtest``。策略负责：

1. 根据交易日历构造信号日、执行日和因子预热日期；
2. 通过 BigQuant 数据适配器一次性预存原始数据；
3. 在每个信号日动态调用因子函数计算单日截面；
4. 在各市值组内按原始因子值分位区间独立选股；
5. 使用 BigTrader 在下一交易日按指定价格等权调仓；
6. 保留信号、调仓、订单和成交审计对象，但默认不打印它们。

默认显式输出仅为 ``M.bigtrader.v35`` 生成的 BigQuant 回测图表。
"""

# 对外仅使用文件底部的 run_market_cap_group_backtest()。

from __future__ import annotations

import math
import time
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


def _query_trading_calendar(end_date):
    """读取截至回测结束日的完整 A 股交易日历。"""
    import dai

    sql = f"""
    SELECT DISTINCT date
    FROM cn_stock_bar1d
    WHERE date <= '{end_date:%Y-%m-%d}'
    ORDER BY date
    """
    calendar = dai.query(sql).df()
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


def _query_index_universe(index_codes, signal_dates):
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
    panel = dai.query(sql, filters=partition_filters).df()
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


def _build_universe_panel(universe_config, signal_dates):
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


def _progress_line(
    completed,
    total,
    current_date,
    started_at,
):
    elapsed = time.perf_counter() - started_at
    remaining = (
        elapsed / completed * (total - completed)
        if completed > 0
        else np.nan
    )
    print(
        "\r"
        f"[市值分组回测] 信号 {completed}/{total} "
        f"| {completed / total:.1%} "
        f"| 当前：{current_date:%Y-%m-%d} "
        f"| 已耗时：{elapsed:.1f}s "
        f"| 预计剩余：{remaining:.1f}s",
        end="",
        flush=True,
    )


def run_market_cap_group_backtest(
    start_date,
    end_date,
    rebalance_interval,
    universe,
    factor_name,
    market_cap_group_count=15,
    selected_market_cap_groups=None,
    factor_quantile_range=(0.0, 0.1),
    factor_params=None,
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
    from factor_lib.common.data_adapters.bigquant_adapters.loader import (
        get_factor_data_requirements,
        get_factor_metadata,
        load_factor_raw_data,
    )
    from factor_lib.factor_hub.get_factor import get_factor

    if show_progress:
        print("[市值分组回测] [1/5] 读取交易日历并生成调仓计划...")

    trading_calendar = _query_trading_calendar(end_date)
    schedule = _build_schedule(
        trading_calendar=trading_calendar,
        start_date=start_date,
        end_date=end_date,
        rebalance_interval=rebalance_interval,
    )
    signal_dates = pd.DatetimeIndex(schedule["signal_date"])
    execution_dates = pd.DatetimeIndex(schedule["execution_date"])

    metadata = get_factor_metadata(factor_name)
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
    factor_column = _resolve_factor_column(metadata, factor_name)
    history_days = _resolve_data_window(requirements)
    factor_date_windows, factor_dates = _build_factor_date_windows(
        schedule=schedule,
        trading_calendar=trading_calendar,
        history_days=history_days,
    )

    universe_panel, load_instruments = _build_universe_panel(
        universe_config,
        signal_dates,
    )

    if show_progress:
        print("[市值分组回测] [2/5] 预存因子、选股和执行约束数据...")

    factor_raw_data = load_factor_raw_data(
        factor_name=factor_name,
        dates=factor_dates,
        factor_params=resolved_factor_params,
        instruments=load_instruments,
        show_progress=False,
    )
    factor_raw_data = _validate_panel(
        factor_raw_data,
        "factor_raw_data",
        ["date", "instrument"],
    )

    signal_fields = [
        "total_market_cap",
        "is_risk_warning",
        "suspended",
        "volume",
    ]
    signal_panel = load_daily_raw_data(
        standard_fields=signal_fields,
        dates=signal_dates,
        instruments=load_instruments,
        show_progress=False,
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

    execution_panel = load_daily_raw_data(
        standard_fields=execution_fields,
        dates=execution_dates,
        instruments=load_instruments,
        show_progress=False,
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

    factor_data_by_date = {
        date: group.copy()
        for date, group in factor_raw_data.groupby("date", sort=False)
    }
    signal_state_by_date = {
        date: group.copy()
        for date, group in signal_panel.groupby("date", sort=False)
    }
    execution_state_map = {}
    for _, row in execution_panel[
        [
            "date",
            "instrument",
            "can_buy",
            "can_sell",
            "buy_blocked_reason",
            "sell_blocked_reason",
            "_buy_price",
            "_sell_price",
        ]
    ].iterrows():
        execution_state_map[(row["date"], row["instrument"])] = {
            "can_buy": bool(row["can_buy"]),
            "can_sell": bool(row["can_sell"]),
            "buy_blocked_reason": row["buy_blocked_reason"],
            "sell_blocked_reason": row["sell_blocked_reason"],
            "buy_price": row["_buy_price"],
            "sell_price": row["_sell_price"],
        }

    schedule_by_signal = {
        row.signal_date: {
            "rebalance_number": int(row.rebalance_number),
            "execution_date": row.execution_date,
        }
        for row in schedule.itertuples(index=False)
    }

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
    progress_started_at = time.perf_counter()

    if show_progress:
        print(
            "[市值分组回测] [3/5] 数据准备完成："
            f"{len(schedule)} 个信号，"
            f"{len(engine_instruments):,} 只候选股票。"
        )
        print("[市值分组回测] [4/5] 启动 BigTrader 原生回测...")

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
        nonlocal completed_signal_count, successful_signal_count

        signal_date = pd.Timestamp(data.current_dt).normalize()
        schedule_item = schedule_by_signal.get(signal_date)
        if schedule_item is None:
            return

        completed_signal_count += 1
        execution_date = schedule_item["execution_date"]
        rebalance_number = schedule_item["rebalance_number"]

        try:
            required_dates = factor_date_windows[signal_date]
            raw_parts = [
                factor_data_by_date[date]
                for date in required_dates
                if date in factor_data_by_date
            ]
            if len(raw_parts) != len(required_dates):
                available = {part["date"].iloc[0] for part in raw_parts}
                missing = [
                    date.strftime("%Y-%m-%d")
                    for date in required_dates
                    if date not in available
                ]
                raise ValueError(
                    f"因子预热窗口缺少日期：{missing[:5]}"
                )

            factor_input = pd.concat(raw_parts, ignore_index=True)
            factor_cross_section = get_factor(
                factor_name,
                factor_input,
                target_dates=[signal_date],
                as_of_date=signal_date,
                show_progress=False,
                progress_every=progress_every,
                **resolved_factor_params,
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
            if show_progress and (
                completed_signal_count == 1
                or completed_signal_count % progress_every == 0
                or completed_signal_count == len(schedule)
            ):
                _progress_line(
                    completed_signal_count,
                    len(schedule),
                    signal_date,
                    progress_started_at,
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

        if show_progress and (
            completed_signal_count == 1
            or completed_signal_count % progress_every == 0
            or completed_signal_count == len(schedule)
        ):
            _progress_line(
                completed_signal_count,
                len(schedule),
                signal_date,
                progress_started_at,
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
    }
    if benchmark is not None:
        bigtrader_kwargs["benchmark"] = benchmark

    performance = M.bigtrader.v35(**bigtrader_kwargs)

    if show_progress:
        if completed_signal_count:
            print()
        elapsed = time.perf_counter() - started_at
        print(
            "[市值分组回测] [5/5] 完成："
            f"{completed_signal_count}/{len(schedule)} 个信号已处理，"
            f"{successful_signal_count} 个信号成功，"
            f"耗时 {elapsed:.1f}s。"
        )

    signals = pd.DataFrame(signal_records)
    rebalance_audit = pd.DataFrame(rebalance_records)
    execution_audit = pd.DataFrame(execution_records)
    order_audit = pd.DataFrame(order_records)
    trade_audit = pd.DataFrame(trade_records)

    data_diagnostics = {
        "requested_start_date": start_date,
        "actual_first_execution_date": schedule["execution_date"].iloc[0],
        "engine_start_date": schedule["signal_date"].iloc[0],
        "end_date": end_date,
        "factor_history_days": history_days,
        "resolved_factor_params": dict(
            resolved_factor_params
        ),
        "factor_raw_rows": len(factor_raw_data),
        "signal_state_rows": len(signal_panel),
        "execution_state_rows": len(execution_panel),
        "engine_instrument_count": len(engine_instruments),
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
