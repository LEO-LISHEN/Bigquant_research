# -*- coding: utf-8 -*-
"""BigQuant 市值分组因子回测策略。"""

import time

import numpy as np
import pandas as pd


def _to_bool(series, default=False):
    """将布尔、数值和常见文本标识统一转为 bool。"""
    text = series.astype("string").str.strip().str.lower()
    result = text.isin({"1", "true", "t", "yes", "y", "是"})

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].ne(0)

    missing_mask = (
        series.isna()
        | text.isin({"", "nan", "none", "<na>"})
    )
    result.loc[missing_mask] = bool(default)
    return result.astype(bool)


def _query_trading_dates(start_date, end_date):
    """读取 BigQuant 全市场交易日历。"""
    import dai

    calendar = dai.query(
        f"""
        SELECT DISTINCT date
        FROM cn_stock_bar1d
        WHERE date BETWEEN '{start_date:%Y-%m-%d}'
          AND '{end_date:%Y-%m-%d}'
        ORDER BY date
        """
    ).df()

    if calendar.empty:
        raise ValueError("未查询到回测区间的交易日历。")

    return (
        pd.DatetimeIndex(pd.to_datetime(calendar["date"]))
        .normalize()
        .unique()
        .sort_values()
    )


def _resolve_factor_metadata(
    factor_name,
    factor_column,
    factor_direction,
):
    """从因子中心读取方向；临时因子才允许手工给方向。"""
    if factor_direction is not None and factor_direction not in {1, -1}:
        raise ValueError(
            "factor_direction 仅支持 1（正向）或 -1（反向）。"
        )

    lookup_name = factor_name or factor_column

    try:
        from factor_lib.factor_hub.describe_factor import describe_factor

        metadata = describe_factor(lookup_name)
        metadata_direction = metadata.get("direction")

        if metadata_direction not in {1, -1}:
            raise ValueError(
                f"因子 {lookup_name} 的 FACTOR['direction'] 必须为 1 或 -1。"
            )

        if (
            factor_direction is not None
            and factor_direction != metadata_direction
        ):
            raise ValueError(
                "手工传入的 factor_direction 与因子元数据不一致："
                f"{factor_direction} != {metadata_direction}。"
            )

        return lookup_name, metadata_direction, metadata

    except ValueError as error:
        if factor_name is not None:
            raise error

        if factor_direction is None:
            raise ValueError(
                f"未在因子中心找到因子 {lookup_name}。"
                "请传入已登记的 factor_name，或为临时因子显式传入 "
                "factor_direction。"
            ) from error

        return lookup_name, factor_direction, {}


def _build_universe_panel(
    universe,
    factor_panel,
    rebalance_dates,
    active_date_sql,
):
    """构造调仓日动态股票池。"""
    if isinstance(universe, str):
        if universe != "all_a":
            raise ValueError("字符串 universe 目前仅支持 'all_a'。")
        panel = factor_panel[["date", "instrument"]].drop_duplicates()
        panel["in_universe"] = True

    elif isinstance(universe, pd.DataFrame):
        panel = universe.copy()
        required = {"date", "instrument"}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"动态股票池缺少字段：{sorted(missing)}")
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        if "in_universe" not in panel.columns:
            panel["in_universe"] = True
        panel = panel[["date", "instrument", "in_universe"]].copy()

    elif isinstance(universe, (list, tuple, set)):
        panel = factor_panel[["date", "instrument"]].drop_duplicates()
        panel["in_universe"] = panel["instrument"].isin(set(universe))

    elif isinstance(universe, dict):
        universe_type = universe.get("type")

        if universe_type == "custom_list":
            panel = factor_panel[["date", "instrument"]].drop_duplicates()
            panel["in_universe"] = panel["instrument"].isin(
                set(universe.get("instruments", []))
            )

        elif universe_type == "custom_panel":
            panel = universe.get("data")
            if not isinstance(panel, pd.DataFrame):
                raise ValueError("custom_panel 的 data 必须是 pandas.DataFrame。")
            panel = panel.copy()
            required = {"date", "instrument"}
            missing = required - set(panel.columns)
            if missing:
                raise ValueError(f"动态股票池缺少字段：{sorted(missing)}")
            panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
            if "in_universe" not in panel.columns:
                panel["in_universe"] = True
            panel = panel[["date", "instrument", "in_universe"]].copy()

        elif universe_type == "index":
            import dai

            index_code = universe.get("code")
            if not index_code:
                raise ValueError("指数股票池需要提供 universe['code']。")
            panel = dai.query(
                f"""
                SELECT date, member_code AS instrument
                FROM cn_stock_index_component
                WHERE instrument = '{index_code}'
                  AND date IN ({active_date_sql})
                """
            ).df()
            if panel.empty:
                raise ValueError(f"未读取到指数 {index_code} 的历史成分股。")
            panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
            panel["in_universe"] = True

        elif universe_type == "industry":
            import dai

            industry = universe.get("industry", "sw2021")
            level1_code = universe.get("level1_code")
            if not level1_code:
                raise ValueError("行业股票池需要提供 universe['level1_code']。")
            panel = dai.query(
                f"""
                SELECT date, instrument
                FROM cn_stock_industry_component
                WHERE industry = '{industry}'
                  AND industry_level1_code = '{level1_code}'
                  AND date IN ({active_date_sql})
                """
            ).df()
            if panel.empty:
                raise ValueError("未读取到指定行业的历史成分股。")
            panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
            panel["in_universe"] = True

        else:
            raise ValueError(
                "universe dict 的 type 仅支持："
                "custom_list、custom_panel、index、industry。"
            )

    else:
        raise TypeError(
            "universe 仅支持 str、list、dict 或 pandas.DataFrame。"
        )

    panel = panel.loc[panel["date"].isin(rebalance_dates)].copy()
    if panel.duplicated(["date", "instrument"]).any():
        raise ValueError("universe 中存在重复的 date + instrument。")
    panel["in_universe"] = _to_bool(panel["in_universe"], default=False)
    return panel


def run_market_cap_group_backtest(
    factor_data,
    factor_column,
    factor_name=None,
    universe="all_a",
    start_date=None,
    end_date=None,
    market_cap_group_count=15,
    selected_market_cap_groups=None,
    factor_direction=None,
    factor_quantile_range=None,
    rebalance_interval=5,
    initial_cash=1_000_000,
    benchmark="000300.SH",
    trading_costs=None,
    slippage_value=None,
    volume_limit=0.025,
    show_progress=False,
    progress_every=10,
):
    """运行 BigQuant A 股市值分组回测。

    已登记因子会优先从 FACTOR['direction'] 自动读取方向：
    ``factor_name`` 不传时，默认用 ``factor_column`` 查询。
    只有未登记的临时因子才需手动传入 ``factor_direction``。

    函数内部读取 BigQuant 市值、风险警示、停牌、开盘价、涨跌停价
    与成交量；外部仅需提供因子面板。
    """
    if not isinstance(factor_column, str) or not factor_column:
        raise ValueError("factor_column 必须是非空字符串。")
    if market_cap_group_count < 1:
        raise ValueError("market_cap_group_count 必须至少为 1。")
    if rebalance_interval < 1 or progress_every < 1:
        raise ValueError("rebalance_interval 和 progress_every 必须为正整数。")
    if not 0 < volume_limit <= 1:
        raise ValueError("volume_limit 必须满足 0 < volume_limit <= 1。")

    resolved_name, resolved_direction, factor_metadata = (
        _resolve_factor_metadata(
            factor_name=factor_name,
            factor_column=factor_column,
            factor_direction=factor_direction,
        )
    )

    if factor_quantile_range is None:
        factor_quantile_range = (
            (0.9, 1.0)
            if resolved_direction == 1
            else (0.0, 0.1)
        )

    if (
        not isinstance(factor_quantile_range, (list, tuple))
        or len(factor_quantile_range) != 2
    ):
        raise ValueError("factor_quantile_range 必须是 (lower, upper) 形式。")

    quantile_lower = float(factor_quantile_range[0])
    quantile_upper = float(factor_quantile_range[1])
    if not 0 <= quantile_lower < quantile_upper <= 1:
        raise ValueError("factor_quantile_range 必须满足 0 <= lower < upper <= 1。")

    if trading_costs is None:
        trading_costs = {
            "buy_cost": 0.0003,
            "sell_cost": 0.0003,
            "min_cost": 5.0,
            "tax_ratio": 0.0005,
        }
    else:
        trading_costs = dict(trading_costs)

    required_costs = {"buy_cost", "sell_cost", "min_cost", "tax_ratio"}
    missing_costs = required_costs - set(trading_costs)
    if missing_costs:
        raise ValueError(f"trading_costs 缺少字段：{sorted(missing_costs)}")

    required_factor = {"date", "instrument", factor_column}
    missing_factor = required_factor - set(factor_data.columns)
    if missing_factor:
        raise ValueError(f"factor_data 缺少字段：{sorted(missing_factor)}")

    factor_panel = factor_data[["date", "instrument", factor_column]].copy()
    factor_panel["date"] = pd.to_datetime(factor_panel["date"]).dt.normalize()
    if factor_panel.duplicated(["date", "instrument"]).any():
        raise ValueError("factor_data 中存在重复的 date + instrument。")
    factor_panel[factor_column] = pd.to_numeric(
        factor_panel[factor_column], errors="coerce"
    )

    start_date = pd.Timestamp(
        factor_panel["date"].min() if start_date is None else start_date
    ).normalize()
    end_date = pd.Timestamp(
        factor_panel["date"].max() if end_date is None else end_date
    ).normalize()
    if start_date > end_date:
        raise ValueError("start_date 不能晚于 end_date。")

    factor_panel = factor_panel.loc[
        factor_panel["date"].between(start_date, end_date)
    ].copy()
    if factor_panel.empty:
        raise ValueError("回测区间内没有因子数据。")

    if show_progress:
        print("[市值分组回测] [1/4] 读取交易日历...")

    trading_dates = _query_trading_dates(start_date, end_date)
    if len(trading_dates) < 2:
        raise ValueError("回测区间至少需要两个交易日。")

    rebalance_dates = trading_dates[:-1][::rebalance_interval]
    factor_dates = pd.DatetimeIndex(factor_panel["date"].unique())
    missing_dates = rebalance_dates.difference(factor_dates)
    if not missing_dates.empty:
        preview = [date.strftime("%Y-%m-%d") for date in missing_dates[:5]]
        raise ValueError(
            "factor_data 未覆盖部分计划调仓日："
            f"{preview}。"
        )

    execution_date_map = {
        date: trading_dates[position + 1]
        for position, date in enumerate(trading_dates[:-1])
    }
    execution_dates = pd.DatetimeIndex(
        [execution_date_map[date] for date in rebalance_dates]
    )
    active_dates = pd.DatetimeIndex(rebalance_dates).union(
        execution_dates
    ).sort_values()
    active_date_sql = ", ".join(
        f"'{date:%Y-%m-%d}'" for date in active_dates
    )

    if show_progress:
        print("[市值分组回测] [2/4] 读取市值、状态与执行限制数据...")

    import dai

    market_panel = dai.query(
        f"""
        SELECT
            b.date,
            b.instrument,
            v.total_market_cap,
            p.is_risk_warning,
            p.suspended,
            b.open,
            b.volume,
            b.upper_limit,
            b.lower_limit
        FROM cn_stock_bar1d AS b
        JOIN cn_stock_valuation AS v
            ON b.date = v.date
            AND b.instrument = v.instrument
        LEFT JOIN cn_stock_prefactors AS p
            ON b.date = p.date
            AND b.instrument = p.instrument
        WHERE b.date IN ({active_date_sql})
        """
    ).df()

    if market_panel.empty:
        raise ValueError("未读取到市值和交易状态数据。")

    market_panel["date"] = pd.to_datetime(market_panel["date"]).dt.normalize()
    if market_panel.duplicated(["date", "instrument"]).any():
        raise ValueError("读取到重复的市场状态数据。")

    numeric_columns = [
        "total_market_cap", "is_risk_warning", "suspended", "open",
        "volume", "upper_limit", "lower_limit",
    ]
    for column in numeric_columns:
        market_panel[column] = pd.to_numeric(
            market_panel[column], errors="coerce"
        )

    market_panel["_is_st"] = market_panel["is_risk_warning"].fillna(1).ne(0)
    market_panel["_is_suspended"] = (
        market_panel["suspended"].fillna(1).ne(0)
        | market_panel["volume"].fillna(0).le(0)
    )

    valid_open = market_panel["open"].notna() & market_panel["open"].gt(0)
    valid_upper = (
        market_panel["upper_limit"].notna()
        & market_panel["upper_limit"].gt(0)
    )
    valid_lower = (
        market_panel["lower_limit"].notna()
        & market_panel["lower_limit"].gt(0)
    )
    open_at_upper = pd.Series(
        np.isclose(
            market_panel["open"], market_panel["upper_limit"],
            rtol=0.0, atol=1e-8, equal_nan=False,
        ),
        index=market_panel.index,
    )
    open_at_lower = pd.Series(
        np.isclose(
            market_panel["open"], market_panel["lower_limit"],
            rtol=0.0, atol=1e-8, equal_nan=False,
        ),
        index=market_panel.index,
    )
    market_panel["_can_buy"] = (
        ~market_panel["_is_suspended"]
        & valid_open & valid_upper & ~open_at_upper
    )
    market_panel["_can_sell"] = (
        ~market_panel["_is_suspended"]
        & valid_open & valid_lower & ~open_at_lower
    )

    universe_panel = _build_universe_panel(
        universe=universe,
        factor_panel=factor_panel,
        rebalance_dates=rebalance_dates,
        active_date_sql=active_date_sql,
    )

    signal_panel = factor_panel.merge(
        market_panel[
            ["date", "instrument", "total_market_cap", "_is_st", "_is_suspended"]
        ],
        on=["date", "instrument"],
        how="inner",
    ).merge(
        universe_panel,
        on=["date", "instrument"],
        how="left",
    )
    signal_panel["in_universe"] = _to_bool(
        signal_panel["in_universe"], default=False
    )
    signal_panel["_eligible"] = (
        signal_panel["in_universe"]
        & signal_panel[factor_column].notna()
        & signal_panel["total_market_cap"].gt(0)
        & ~signal_panel["_is_st"]
        & ~signal_panel["_is_suspended"]
    )

    if selected_market_cap_groups is None:
        selected_market_cap_groups = list(range(1, market_cap_group_count + 1))
    else:
        selected_market_cap_groups = sorted({int(item) for item in selected_market_cap_groups})
    if (
        not selected_market_cap_groups
        or min(selected_market_cap_groups) < 1
        or max(selected_market_cap_groups) > market_cap_group_count
    ):
        raise ValueError("selected_market_cap_groups 不在有效市值组范围内。")

    if show_progress:
        print("[市值分组回测] [3/4] 生成市值分组与理论目标权重...")

    target_parts = []
    rebalance_records = []
    progress_start = time.perf_counter()
    total_dates = len(rebalance_dates)

    for position, signal_date in enumerate(rebalance_dates, start=1):
        cross_section = signal_panel.loc[
            signal_panel["date"] == signal_date
        ].copy()
        eligible = cross_section.loc[cross_section["_eligible"]].copy()
        selected_parts = []

        if eligible.empty:
            rebalance_records.append({
                "signal_date": signal_date,
                "execution_date": execution_date_map[signal_date],
                "candidate_count": len(cross_section),
                "eligible_count": 0,
                "market_cap_group": pd.NA,
                "group_population": 0,
                "selected_count": 0,
            })
            continue

        actual_group_count = min(market_cap_group_count, len(eligible))
        eligible = eligible.sort_values(
            ["total_market_cap", "instrument"],
            kind="mergesort",
        ).copy()
        eligible["_market_cap_rank"] = np.arange(1, len(eligible) + 1)
        eligible["_market_cap_group"] = (
            pd.qcut(
                eligible["_market_cap_rank"],
                q=actual_group_count,
                labels=False,
            ).astype(int) + 1
        )

        for group_number in range(1, actual_group_count + 1):
            group = eligible.loc[
                eligible["_market_cap_group"] == group_number
            ].sort_values([factor_column, "instrument"], kind="mergesort")
            selected = group.iloc[0:0].copy()

            if group_number in selected_market_cap_groups:
                start_index = int(np.floor(len(group) * quantile_lower))
                end_index = int(np.ceil(len(group) * quantile_upper))
                end_index = max(end_index, start_index + 1)
                selected = group.iloc[start_index:end_index].copy()
                if not selected.empty:
                    selected_parts.append(selected)

            rebalance_records.append({
                "signal_date": signal_date,
                "execution_date": execution_date_map[signal_date],
                "candidate_count": len(cross_section),
                "eligible_count": len(eligible),
                "market_cap_group": group_number,
                "group_population": len(group),
                "selected_count": len(selected),
            })

        if selected_parts:
            selected = pd.concat(selected_parts, ignore_index=True)
            selected = selected.drop_duplicates("instrument").copy()
            selected["target_weight"] = 1.0 / len(selected)
            selected["execution_date"] = execution_date_map[signal_date]
            target_parts.append(
                selected[
                    [
                        "date", "execution_date", "instrument", factor_column,
                        "total_market_cap", "_market_cap_group", "target_weight",
                    ]
                ].rename(columns={
                    "date": "signal_date",
                    "_market_cap_group": "market_cap_group",
                })
            )

        if show_progress and (
            position == 1
            or position % progress_every == 0
            or position == total_dates
        ):
            elapsed = time.perf_counter() - progress_start
            remaining = elapsed / position * (total_dates - position)
            print(
                f"\r[市值分组回测] {position}/{total_dates} "
                f"| {position / total_dates:.1%} "
                f"| 当前：{signal_date:%Y-%m-%d} "
                f"| 已耗时：{elapsed:.1f}s "
                f"| 预计剩余：{remaining:.1f}s",
                end="",
                flush=True,
            )

    if show_progress:
        print()

    if not target_parts:
        raise ValueError("所有调仓日均未选出股票。")

    target_weights = pd.concat(target_parts, ignore_index=True)
    rebalance_audit = pd.DataFrame(rebalance_records)

    execution_state = market_panel[
        ["date", "instrument", "_can_buy", "_can_sell"]
    ].rename(columns={
        "date": "execution_date",
        "_can_buy": "can_buy_at_execution",
        "_can_sell": "can_sell_at_execution",
    })
    execution_audit = target_weights.merge(
        execution_state,
        on=["execution_date", "instrument"],
        how="left",
        validate="many_to_one",
    )
    execution_audit["execution_data_available"] = (
        execution_audit["can_buy_at_execution"].notna()
        & execution_audit["can_sell_at_execution"].notna()
    )
    execution_audit["can_buy_at_execution"] = execution_audit[
        "can_buy_at_execution"
    ].fillna(False)
    execution_audit["can_sell_at_execution"] = execution_audit[
        "can_sell_at_execution"
    ].fillna(False)

    execution_state_map = {
        (row.execution_date, row.instrument): {
            "can_buy": bool(row.can_buy_at_execution),
            "can_sell": bool(row.can_sell_at_execution),
        }
        for row in execution_state.itertuples(index=False)
    }
    target_map = {
        pd.Timestamp(signal_date).strftime("%Y-%m-%d"): group[
            ["instrument", "target_weight"]
        ].to_dict("records")
        for signal_date, group in target_weights.groupby("signal_date", sort=True)
    }
    engine_instruments = sorted(target_weights["instrument"].unique())

    order_records = []
    trade_records = []
    position_records = []
    account_records = []

    if show_progress:
        print("[市值分组回测] [4/4] 启动 BigTrader 原生回测...")

    from bigmodule import M

    def _execution_state(execution_date, instrument):
        return execution_state_map.get(
            (pd.Timestamp(execution_date), instrument),
            {"can_buy": False, "can_sell": False},
        )

    def initialize(context):
        from bigtrader.finance.commission import PerOrder

        context.set_commission(
            PerOrder(
                buy_cost=float(trading_costs["buy_cost"]),
                sell_cost=float(trading_costs["sell_cost"]),
                min_cost=float(trading_costs["min_cost"]),
                tax_ratio=float(trading_costs["tax_ratio"]),
            )
        )
        if slippage_value is not None:
            context.set_slippage_value(
                slippage_type=2,
                slippage_value=float(slippage_value),
            )

    def handle_data(context, data):
        signal_date = pd.Timestamp(data.current_dt).normalize()
        targets = target_map.get(signal_date.strftime("%Y-%m-%d"))
        if targets is None:
            return

        execution_date = execution_date_map[signal_date]
        target_instruments = {item["instrument"] for item in targets}
        positions = context.get_account_positions()
        holding_instruments = {
            instrument
            for instrument, position in positions.items()
            if getattr(position, "current_qty", 0) > 0
        }

        for instrument in sorted(holding_instruments - target_instruments):
            state = _execution_state(execution_date, instrument)
            if state["can_sell"]:
                result = context.order_target(instrument, 0)
                action, reason = "sell_submitted", ""
            else:
                result = None
                action, reason = "sell_blocked", "can_sell=False"
            order_records.append({
                "signal_date": signal_date,
                "execution_date": execution_date,
                "instrument": instrument,
                "target_weight": 0.0,
                "action": action,
                "reason": reason,
                "submit_result": result,
            })

        for item in targets:
            instrument = item["instrument"]
            target_weight = float(item["target_weight"])
            state = _execution_state(execution_date, instrument)

            if instrument not in holding_instruments:
                if state["can_buy"]:
                    result = context.order_target_percent(instrument, target_weight)
                    action, reason = "buy_submitted", ""
                else:
                    result = None
                    action, reason = "buy_blocked", "can_buy=False"
            elif state["can_buy"] and state["can_sell"]:
                result = context.order_target_percent(instrument, target_weight)
                action, reason = "rebalance_submitted", ""
            else:
                result = None
                action, reason = "rebalance_frozen", "can_buy=False or can_sell=False"

            order_records.append({
                "signal_date": signal_date,
                "execution_date": execution_date,
                "instrument": instrument,
                "target_weight": target_weight,
                "action": action,
                "reason": reason,
                "submit_result": result,
            })

    def handle_trade(context, trade):
        trade_records.append({
            "trade_date": pd.to_datetime(
                str(getattr(trade, "trading_day", "")),
                errors="coerce",
            ),
            "instrument": getattr(trade, "instrument", None),
            "direction": str(getattr(trade, "direction", None)),
            "filled_qty": getattr(trade, "filled_qty", np.nan),
            "filled_price": getattr(trade, "filled_price", np.nan),
            "filled_money": getattr(trade, "filled_money", np.nan),
            "order_key": getattr(trade, "order_key", None),
        })

    def after_trading(context, data):
        current_date = pd.Timestamp(data.current_dt).normalize()
        portfolio = context.portfolio
        account_records.append({
            "date": current_date,
            "cash": getattr(portfolio, "cash", np.nan),
            "portfolio_value": getattr(portfolio, "portfolio_value", np.nan),
        })
        for instrument, position in context.get_account_positions().items():
            if getattr(position, "current_qty", 0) > 0:
                position_records.append({
                    "date": current_date,
                    "instrument": instrument,
                    "current_qty": getattr(position, "current_qty", np.nan),
                    "avail_qty": getattr(position, "avail_qty", np.nan),
                    "cost_price": getattr(position, "cost_price", np.nan),
                    "last_price": getattr(position, "last_price", np.nan),
                })

    backtest = M.bigtrader.v35(
        data={"instruments": engine_instruments},
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initialize=initialize,
        handle_data=handle_data,
        handle_trade=handle_trade,
        after_trading=after_trading,
        capital_base=float(initial_cash),
        benchmark=benchmark,
        frequency="daily",
        volume_limit=float(volume_limit),
        order_price_field_buy="open",
        order_price_field_sell="open",
    )

    if show_progress:
        print("[市值分组回测] 已完成。")

    return {
        "backtest": backtest,
        "factor_metadata": factor_metadata,
        "factor_name": resolved_name,
        "factor_direction": resolved_direction,
        "factor_quantile_range": tuple(factor_quantile_range),
        "target_weights": target_weights,
        "execution_constraint_audit": execution_audit,
        "rebalance_audit": rebalance_audit,
        "order_audit": pd.DataFrame(order_records),
        "trade_audit": pd.DataFrame(trade_records),
        "position_audit": pd.DataFrame(position_records),
        "account_audit": pd.DataFrame(account_records),
        "rebalance_dates": list(rebalance_dates),
    }
