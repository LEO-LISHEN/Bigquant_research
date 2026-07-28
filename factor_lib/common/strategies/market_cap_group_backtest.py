# -*- coding: utf-8 -*-
"""市值分组因子回测策略。"""

import time

import numpy as np
import pandas as pd


def run_market_cap_group_backtest(
    factor_data,
    factor_column,
    market_data,
    universe="all_a",
    start_date=None,
    end_date=None,
    market_cap_column="total_market_cap",
    st_column="is_st",
    suspended_column="is_suspended",
    market_cap_group_count=15,
    selected_market_cap_groups=None,
    factor_quantile_range=(0.0, 0.1),
    rebalance_interval=5,
    initial_cash=1_000_000,
    benchmark="000300.SH",
    trading_costs=None,
    slippage_value=None,
    show_progress=False,
    progress_every=10,
):
    """
    基于因子分位数和市值分组的 A 股回测。

    因子在信号日收盘后生成，订单交由 BigTrader 在下一交易日开盘撮合。

    参数
    ----
    factor_data : pandas.DataFrame
        因子值面板，必须包含 date、instrument 和 factor_column。
        必须覆盖每一个计划调仓日。
    factor_column : str
        本次回测使用的因子列名。
    market_data : pandas.DataFrame
        市场状态面板，必须包含：
        date、instrument、market_cap_column、st_column、suspended_column。

        推荐将 BigQuant 原始字段预先映射为标准字段：
        total_market_cap、is_st、is_suspended。
    universe : str、list、dict 或 pandas.DataFrame，默认 "all_a"
        选股范围：

        - "all_a"：
          使用 factor_data 与 market_data 同时存在的股票作为候选范围。
          因此 factor_data 必须确实按全 A 股计算。
        - list / tuple / set：
          静态自定义股票列表。历史回测可能有幸存者偏差。
        - pandas.DataFrame：
          动态股票池，必须包含 date、instrument；
          可选 in_universe 列，True 表示该日属于股票池。
        - {"type": "custom_list", "instruments": [...]}：
          静态自定义股票列表。
        - {"type": "custom_panel", "data": universe_data}：
          动态股票池面板。
        - {"type": "index", "code": "000300.SH"}：
          通过 BigQuant 的历史指数成分表生成动态股票池。
        - {"type": "industry", "industry": "sw2021",
           "level1_code": "650000"}：
          通过 BigQuant 的历史行业成分表生成动态股票池。
    start_date, end_date : str 或 datetime，可选
        回测区间。不传时使用 factor_data 的最早、最晚日期。
    market_cap_column : str，默认 "total_market_cap"
        市值字段名。
    st_column : str，默认 "is_st"
        ST 标识字段。0 / False 表示正常，非 0 / True 表示 ST 或 *ST。
    suspended_column : str，默认 "is_suspended"
        停牌标识字段。0 / False 表示未停牌。
    market_cap_group_count : int，默认 15
        每个信号日按市值划分的近似等数量组数。
        第 1 组为最小市值组。
    selected_market_cap_groups : list[int]，可选
        参与选股的市值组，例如 [1, 2, 3]。
        默认使用全部市值组。
    factor_quantile_range : tuple[float, float]，默认 (0.0, 0.1)
        每个市值组内按原始因子值升序排序后的分位区间。

        (0.0, 0.01) 表示最低 1%；
        (0.0, 0.10) 表示最低 10%；
        (0.9, 1.0) 表示最高 10%。

        使用精确的排序切片，不会因边界因子值相同而重复选股。
    rebalance_interval : int，默认 5
        调仓间隔，单位为交易日。
        因子数据必须覆盖回测期内每一个计划调仓日。
    initial_cash : float，默认 1_000_000
        初始资金。
    benchmark : str，默认 "000300.SH"
        回测基准指数。
    trading_costs : dict，可选
        交易成本参数。默认值：

        {
            "buy_cost": 0.0003,
            "sell_cost": 0.0003,
            "min_cost": 5.0,
            "tax_ratio": 0.0005,
        }

        可按你的实际费率覆盖。
    slippage_value : float，可选
        BigTrader 滑点参数。None 表示不额外设置滑点。
    show_progress : bool，默认 False
        是否显示单行刷新进度。
    progress_every : int，默认 10
        每处理多少个调仓日刷新一次进度。

    返回
    ----
    dict
        {
            "backtest": BigTrader 回测模块结果,
            "target_weights": 每期目标权重表,
            "rebalance_audit": 市值组与选股审计表,
            "rebalance_dates": 实际信号日列表,
        }

    注意
    ----
    - 目标权重是理想目标；涨跌停、停牌、资金不足等导致的未成交，
      由 BigTrader 原生撮合逻辑处理。
    - 未成交买入不会使用未来信息重新分配现金。
    - 静态股票列表仅适合特定研究用途；指数和行业回测应优先使用
      历史动态股票池。
    """
    if not isinstance(factor_column, str) or not factor_column:
        raise ValueError("factor_column 必须是非空字符串")

    if market_cap_group_count < 1:
        raise ValueError("market_cap_group_count 必须至少为 1")

    if rebalance_interval < 1:
        raise ValueError("rebalance_interval 必须至少为 1")

    if progress_every < 1:
        raise ValueError("progress_every 必须至少为 1")

    if (
        not isinstance(factor_quantile_range, (list, tuple))
        or len(factor_quantile_range) != 2
    ):
        raise ValueError(
            "factor_quantile_range 必须是 (lower, upper) 形式。"
        )

    quantile_lower = float(factor_quantile_range[0])
    quantile_upper = float(factor_quantile_range[1])

    if not (
        0.0 <= quantile_lower < quantile_upper <= 1.0
    ):
        raise ValueError(
            "factor_quantile_range 必须满足 "
            "0 <= lower < upper <= 1。"
        )

    if trading_costs is None:
        trading_costs = {
            "buy_cost": 0.0003,
            "sell_cost": 0.0003,
            "min_cost": 5.0,
            "tax_ratio": 0.0005,
        }

    required_cost_keys = {
        "buy_cost",
        "sell_cost",
        "min_cost",
        "tax_ratio",
    }
    missing_cost_keys = required_cost_keys - set(trading_costs)

    if missing_cost_keys:
        raise ValueError(
            f"trading_costs 缺少字段：{sorted(missing_cost_keys)}"
        )

    factor_required = {
        "date",
        "instrument",
        factor_column,
    }
    market_required = {
        "date",
        "instrument",
        market_cap_column,
        st_column,
        suspended_column,
    }

    missing_factor_columns = factor_required - set(factor_data.columns)
    missing_market_columns = market_required - set(market_data.columns)

    if missing_factor_columns:
        raise ValueError(
            "factor_data 缺少字段："
            f"{sorted(missing_factor_columns)}"
        )

    if missing_market_columns:
        raise ValueError(
            "market_data 缺少字段："
            f"{sorted(missing_market_columns)}"
        )

    factor_panel = factor_data[
        ["date", "instrument", factor_column]
    ].copy()

    market_panel = market_data[
        [
            "date",
            "instrument",
            market_cap_column,
            st_column,
            suspended_column,
        ]
    ].copy()

    factor_panel["date"] = pd.to_datetime(factor_panel["date"])
    market_panel["date"] = pd.to_datetime(market_panel["date"])

    if factor_panel.duplicated(["date", "instrument"]).any():
        raise ValueError(
            "factor_data 中存在重复的 date + instrument。"
        )

    if market_panel.duplicated(["date", "instrument"]).any():
        raise ValueError(
            "market_data 中存在重复的 date + instrument。"
        )

    factor_panel[factor_column] = pd.to_numeric(
        factor_panel[factor_column],
        errors="coerce",
    )

    market_panel[market_cap_column] = pd.to_numeric(
        market_panel[market_cap_column],
        errors="coerce",
    )

    if start_date is None:
        start_date = factor_panel["date"].min()

    if end_date is None:
        end_date = factor_panel["date"].max()

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    if start_date > end_date:
        raise ValueError("start_date 不能晚于 end_date。")

    factor_panel = factor_panel.loc[
        factor_panel["date"].between(start_date, end_date)
    ].copy()

    market_panel = market_panel.loc[
        market_panel["date"].between(start_date, end_date)
    ].copy()

    if factor_panel.empty:
        raise ValueError("回测区间内没有因子数据。")

    # 使用 BigQuant 实际交易日历，确保调仓间隔是真实交易日间隔。
    import dai

    calendar_sql = f"""
    SELECT DISTINCT date
    FROM cn_stock_bar1d
    WHERE date BETWEEN '{start_date:%Y-%m-%d}'
      AND '{end_date:%Y-%m-%d}'
    ORDER BY date
    """

    trading_calendar = dai.query(calendar_sql).df()

    if trading_calendar.empty:
        raise ValueError("未查询到回测区间的交易日历。")

    trading_dates = pd.DatetimeIndex(
        pd.to_datetime(trading_calendar["date"])
    ).sort_values().unique()

    if len(trading_dates) < 2:
        raise ValueError(
            "回测区间至少需要两个交易日，才能形成信号日与执行日。"
        )

    # 最后一个交易日没有下一交易日可执行，因此不作为信号日。
    candidate_signal_dates = trading_dates[:-1]
    rebalance_dates = candidate_signal_dates[
        ::rebalance_interval
    ]

    factor_dates = pd.DatetimeIndex(
        factor_panel["date"].unique()
    )

    missing_factor_dates = rebalance_dates.difference(factor_dates)

    if not missing_factor_dates.empty:
        preview = [
            date.strftime("%Y-%m-%d")
            for date in missing_factor_dates[:5]
        ]

        raise ValueError(
            "factor_data 未覆盖部分计划调仓日："
            f"{preview}。"
            "请先按回测交易日计算因子，或调整回测区间与调仓频率。"
        )

    execution_date_map = {
        signal_date: trading_dates[position + 1]
        for position, signal_date in enumerate(trading_dates[:-1])
    }

    # ===== 生成动态日期股票池 =====
    if isinstance(universe, str):
        if universe != "all_a":
            raise ValueError(
                "字符串 universe 目前仅支持 'all_a'。"
                "指数或行业股票池请传入 dict。"
            )

        universe_panel = factor_panel[
            ["date", "instrument"]
        ].drop_duplicates()

        universe_panel["in_universe"] = True

    elif isinstance(universe, pd.DataFrame):
        universe_panel = universe.copy()

        required_universe = {"date", "instrument"}
        missing_universe = required_universe - set(
            universe_panel.columns
        )

        if missing_universe:
            raise ValueError(
                "动态股票池缺少字段："
                f"{sorted(missing_universe)}"
            )

        universe_panel["date"] = pd.to_datetime(
            universe_panel["date"]
        )

        if "in_universe" not in universe_panel.columns:
            universe_panel["in_universe"] = True

        universe_panel = universe_panel[
            ["date", "instrument", "in_universe"]
        ].copy()

    elif isinstance(universe, (list, tuple, set)):
        static_instruments = set(universe)

        universe_panel = factor_panel[
            ["date", "instrument"]
        ].drop_duplicates()

        universe_panel["in_universe"] = (
            universe_panel["instrument"].isin(static_instruments)
        )

    elif isinstance(universe, dict):
        universe_type = universe.get("type")

        if universe_type == "custom_list":
            static_instruments = set(
                universe.get("instruments", [])
            )

            universe_panel = factor_panel[
                ["date", "instrument"]
            ].drop_duplicates()

            universe_panel["in_universe"] = (
                universe_panel["instrument"].isin(
                    static_instruments
                )
            )

        elif universe_type == "custom_panel":
            universe_panel = universe.get("data")

            if not isinstance(universe_panel, pd.DataFrame):
                raise ValueError(
                    "custom_panel 的 data 必须是 pandas.DataFrame。"
                )

            universe_panel = universe_panel.copy()

            required_universe = {"date", "instrument"}
            missing_universe = required_universe - set(
                universe_panel.columns
            )

            if missing_universe:
                raise ValueError(
                    "动态股票池缺少字段："
                    f"{sorted(missing_universe)}"
                )

            universe_panel["date"] = pd.to_datetime(
                universe_panel["date"]
            )

            if "in_universe" not in universe_panel.columns:
                universe_panel["in_universe"] = True

            universe_panel = universe_panel[
                ["date", "instrument", "in_universe"]
            ].copy()

        elif universe_type == "index":
            index_code = universe.get("code")

            if not index_code:
                raise ValueError(
                    "指数股票池需要提供 universe['code']。"
                )

            index_sql = f"""
            SELECT
                date,
                member_code AS instrument
            FROM cn_stock_index_component
            WHERE instrument = '{index_code}'
              AND date BETWEEN '{start_date:%Y-%m-%d}'
              AND '{end_date:%Y-%m-%d}'
            """

            universe_panel = dai.query(index_sql).df()

            if universe_panel.empty:
                raise ValueError(
                    f"未读取到指数 {index_code} 的历史成分股数据。"
                )

            universe_panel["date"] = pd.to_datetime(
                universe_panel["date"]
            )
            universe_panel["in_universe"] = True

        elif universe_type == "industry":
            industry = universe.get("industry", "sw2021")
            level1_code = universe.get("level1_code")

            if not level1_code:
                raise ValueError(
                    "行业股票池需要提供 universe['level1_code']。"
                )

            industry_sql = f"""
            SELECT
                date,
                instrument
            FROM cn_stock_industry_component
            WHERE industry = '{industry}'
              AND industry_level1_code = '{level1_code}'
              AND date BETWEEN '{start_date:%Y-%m-%d}'
              AND '{end_date:%Y-%m-%d}'
            """

            universe_panel = dai.query(industry_sql).df()

            if universe_panel.empty:
                raise ValueError(
                    "未读取到指定行业的历史成分股数据。"
                )

            universe_panel["date"] = pd.to_datetime(
                universe_panel["date"]
            )
            universe_panel["in_universe"] = True

        else:
            raise ValueError(
                "universe dict 的 type 仅支持："
                "custom_list、custom_panel、index、industry。"
            )

    else:
        raise TypeError(
            "universe 仅支持 str、list、dict 或 pandas.DataFrame。"
        )

    universe_panel = universe_panel.loc[
        universe_panel["date"].isin(rebalance_dates)
    ].copy()

    if universe_panel.duplicated(["date", "instrument"]).any():
        raise ValueError(
            "universe 中存在重复的 date + instrument。"
        )

    universe_panel["in_universe"] = (
        universe_panel["in_universe"]
        .fillna(False)
        .astype(bool)
    )

    # ===== 构造信号日候选样本 =====
    signal_panel = factor_panel.merge(
        market_panel,
        on=["date", "instrument"],
        how="inner",
    ).merge(
        universe_panel,
        on=["date", "instrument"],
        how="left",
    )

    signal_panel["in_universe"] = (
        signal_panel["in_universe"]
        .fillna(False)
        .astype(bool)
    )

    signal_panel["_is_st"] = pd.to_numeric(
        signal_panel[st_column],
        errors="coerce",
    ).fillna(1)

    signal_panel["_is_suspended"] = pd.to_numeric(
        signal_panel[suspended_column],
        errors="coerce",
    ).fillna(1)

    signal_panel["_is_eligible"] = (
        signal_panel["in_universe"]
        & signal_panel[factor_column].notna()
        & signal_panel[market_cap_column].notna()
        & (signal_panel[market_cap_column] > 0)
        & (signal_panel["_is_st"] == 0)
        & (signal_panel["_is_suspended"] == 0)
    )

    if selected_market_cap_groups is None:
        selected_market_cap_groups = list(
            range(1, market_cap_group_count + 1)
        )
    else:
        selected_market_cap_groups = sorted(
            {int(group) for group in selected_market_cap_groups}
        )

    if not selected_market_cap_groups:
        raise ValueError(
            "selected_market_cap_groups 不能为空。"
        )

    if min(selected_market_cap_groups) < 1:
        raise ValueError(
            "市值组编号从 1 开始。"
        )

    # ===== 逐信号日生成目标权重与审计表 =====
    audit_records = []
    target_weight_parts = []
    start_time = time.perf_counter()
    total_rebalances = len(rebalance_dates)

    if show_progress:
        print(
            f"\r[市值分组回测] 0/{total_rebalances} 个调仓日 | 0.0%",
            end="",
            flush=True,
        )

    try:
        for position, signal_date in enumerate(
            rebalance_dates,
            start=1,
        ):
            cross_section = signal_panel.loc[
                signal_panel["date"] == signal_date
            ].copy()

            eligible = cross_section.loc[
                cross_section["_is_eligible"]
            ].copy()

            candidate_count = len(cross_section)
            eligible_count = len(eligible)
            selected_parts = []

            if eligible_count > 0:
                actual_group_count = min(
                    market_cap_group_count,
                    eligible_count,
                )

                eligible["_market_cap_rank"] = eligible[
                    market_cap_column
                ].rank(
                    method="first",
                    ascending=True,
                )

                eligible["_market_cap_group"] = (
                    pd.qcut(
                        eligible["_market_cap_rank"],
                        q=actual_group_count,
                        labels=False,
                    ).astype(int)
                    + 1
                )

                for group_number in range(
                    1,
                    actual_group_count + 1,
                ):
                    group_data = eligible.loc[
                        eligible["_market_cap_group"] == group_number
                    ].sort_values(
                        factor_column,
                        ascending=True,
                        kind="mergesort",
                    )

                    group_population = len(group_data)
                    selected_data = group_data.iloc[0:0].copy()

                    if (
                        group_number in selected_market_cap_groups
                        and group_population > 0
                    ):
                        start_index = int(
                            np.floor(
                                group_population * quantile_lower
                            )
                        )
                        end_index = int(
                            np.ceil(
                                group_population * quantile_upper
                            )
                        )

                        if (
                            end_index <= start_index
                            and quantile_upper > quantile_lower
                        ):
                            end_index = min(
                                group_population,
                                start_index + 1,
                            )

                        selected_data = group_data.iloc[
                            start_index:end_index
                        ].copy()

                        if not selected_data.empty:
                            selected_parts.append(selected_data)

                    audit_records.append(
                        {
                            "signal_date": signal_date,
                            "execution_date": execution_date_map[
                                signal_date
                            ],
                            "market_cap_group": group_number,
                            "candidate_count": candidate_count,
                            "eligible_count": eligible_count,
                            "group_population": group_population,
                            "valid_factor_count": group_population,
                            "selected_count": len(selected_data),
                            "factor_quantile_lower": quantile_lower,
                            "factor_quantile_upper": quantile_upper,
                        }
                    )

            if selected_parts:
                selected = pd.concat(
                    selected_parts,
                    ignore_index=True,
                )

                selected = selected.drop_duplicates(
                    subset=["instrument"]
                ).copy()

                selected["target_weight"] = 1.0 / len(selected)

                target_weight_parts.append(
                    selected[
                        [
                            "date",
                            "instrument",
                            factor_column,
                            market_cap_column,
                            "_market_cap_group",
                            "target_weight",
                        ]
                    ].rename(
                        columns={
                            "date": "signal_date",
                            "_market_cap_group": "market_cap_group",
                        }
                    )
                )

            should_refresh = (
                position == 1
                or position % progress_every == 0
                or position == total_rebalances
            )

            if show_progress and should_refresh:
                elapsed = time.perf_counter() - start_time
                estimated_remaining = (
                    elapsed
                    / position
                    * (total_rebalances - position)
                )

                print(
                    "\r"
                    f"[市值分组回测] "
                    f"{position}/{total_rebalances} 个调仓日 "
                    f"| {position / total_rebalances:.1%} "
                    f"| 当前：{signal_date:%Y-%m-%d} "
                    f"| 已耗时：{elapsed:.1f}s "
                    f"| 预计剩余：{estimated_remaining:.1f}s",
                    end="",
                    flush=True,
                )
    finally:
        if show_progress:
            print()

    if not target_weight_parts:
        raise ValueError(
            "所有调仓日均未选出股票。请检查动态股票池、"
            "市值字段、ST/停牌字段和因子分位区间。"
        )

    target_weights = pd.concat(
        target_weight_parts,
        ignore_index=True,
    )

    rebalance_audit = pd.DataFrame(audit_records)

    target_map = {
        pd.Timestamp(signal_date).strftime("%Y-%m-%d"): group[
            ["instrument", "target_weight"]
        ].to_dict("records")
        for signal_date, group in target_weights.groupby(
            "signal_date",
            sort=True,
        )
    }

    engine_instruments = sorted(
        target_weights["instrument"].unique().tolist()
    )

    # ===== 使用 BigTrader 原生引擎回测 =====
    from bigmodule import M

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
        signal_date = pd.Timestamp(
            data.current_dt
        ).strftime("%Y-%m-%d")

        targets = target_map.get(signal_date)

        if targets is None:
            return

        target_instruments = {
            item["instrument"] for item in targets
        }

        # 先生成卖出目标。停牌、跌停等无法卖出的持仓由引擎保留。
        for instrument in context.get_account_positions():
            if instrument not in target_instruments:
                context.order_target(instrument, 0)

        # 所有入选股票全局等权；无法买入时，引擎保留现金。
        for item in targets:
            context.order_target_percent(
                item["instrument"],
                float(item["target_weight"]),
            )

    if show_progress:
        print("[市值分组回测] 正在启动 BigTrader 原生回测...")

    backtest = M.bigtrader.v35(
        data={"instruments": engine_instruments},
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initialize=initialize,
        handle_data=handle_data,
        capital_base=float(initial_cash),
        benchmark=benchmark,
        order_price_field_buy="open",
        order_price_field_sell="open",
    )

    if show_progress:
        print("[市值分组回测] BigTrader 回测已提交完成。")

    return {
        "backtest": backtest,
        "target_weights": target_weights,
        "rebalance_audit": rebalance_audit,
        "rebalance_dates": list(rebalance_dates),
    }
