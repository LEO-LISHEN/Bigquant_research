# -*- coding: utf-8 -*-
"""市场、规模与估值三因子模型特质波动率。"""

import time

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_size_industry import (
    neutralize_size_industry,
)


OUTPUT_COLUMNS = ["date", "instrument", "id2_std_nm"]


def _resolve_id2_std_nm_data_window(resolved_params):
    """根据月份参数解析回归和因子收益构造窗口。"""
    n_months = resolved_params.get("n_months", 3)
    trading_days_per_month = resolved_params.get(
        "trading_days_per_month",
        21,
    )
    for name, value in {
        "n_months": n_months,
        "trading_days_per_month": trading_days_per_month,
    }.items():
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} 必须是正整数。")

    window = int(n_months) * int(trading_days_per_month)
    return {
        "lookback_trading_days": window,
        "requires_target_date_data": True,
        "minimum_history_observations": window,
        "preheating_required": True,
        "insufficient_window_behavior": (
            "三因子回归窗口内完整观测不足min_ts_observations时，"
            "该股票目标日因子值输出NaN。"
        ),
    }


def _normalize_target_dates(data_dates, target_dates):
    available = pd.DatetimeIndex(
        data_dates.dropna().unique()
    ).normalize().sort_values()
    if target_dates is None:
        return available
    if isinstance(target_dates, (str, pd.Timestamp)):
        target_dates = [target_dates]
    else:
        try:
            target_dates = list(target_dates)
        except TypeError:
            target_dates = [target_dates]
    normalized = pd.DatetimeIndex(
        pd.to_datetime(target_dates, errors="raise")
    ).normalize().unique().sort_values()
    missing = normalized.difference(available)
    if not missing.empty:
        preview = [d.strftime("%Y-%m-%d") for d in missing[:5]]
        raise ValueError(
            f"id2_std_nm 缺少目标日期原始数据：{preview}。"
        )
    return normalized


def _get_market_close(domain_data, market_index, as_of_date=None):
    """从分域容器读取指定市场指数的原始收盘点位。"""
    if domain_data is None or not hasattr(domain_data, "get_domain"):
        raise TypeError(
            "id2_std_nm 需要包含 market_daily 的 FactorDataBundle；"
            "请通过 loader 和 get_factor 调用。"
        )
    if not isinstance(market_index, str) or not market_index.strip():
        raise ValueError("market_index 必须是非空统一指数名称。")

    market_data = domain_data.get_domain("market_daily").copy()
    required = {"date", "market_index", "market_close"}
    missing = required - set(market_data.columns)
    if missing:
        raise ValueError(
            f"market_daily 缺少字段：{sorted(missing)}"
        )

    market_data["date"] = pd.to_datetime(
        market_data["date"], errors="coerce"
    ).dt.normalize()
    if market_data["date"].isna().any():
        raise ValueError("market_daily.date 包含无效日期。")
    market_data["market_index"] = market_data[
        "market_index"
    ].astype(str)
    market_data = market_data.loc[
        market_data["market_index"] == market_index.strip()
    ].copy()
    if as_of_date is not None:
        market_data = market_data.loc[
            market_data["date"] <= pd.Timestamp(as_of_date).normalize()
        ].copy()
    if market_data.empty:
        raise ValueError(
            f"market_daily 中没有指数 {market_index!r} 的数据。"
        )
    if market_data.duplicated(["date", "market_index"]).any():
        raise ValueError(
            "market_daily 存在重复的 date + market_index。"
        )

    market_close = pd.to_numeric(
        market_data.set_index("date")["market_close"],
        errors="coerce",
    ).sort_index()
    return market_close.where(
        np.isfinite(market_close) & (market_close > 0)
    )


def _build_style_factor_returns(
    stock_return,
    market_cap,
    pb,
    lower_quantile,
    upper_quantile,
    min_universe,
    show_progress,
    progress_every,
    started_at,
):
    """按Notebook口径构造SMB近似和BP因子日收益。"""
    dates = stock_return.index
    size_values = np.full(len(dates), np.nan, dtype=float)
    bp_values = np.full(len(dates), np.nan, dtype=float)

    for position in range(1, len(dates)):
        daily_return = stock_return.iloc[position]
        lagged_cap = market_cap.iloc[position - 1]
        lagged_pb = pb.iloc[position - 1]

        size_valid = (
            daily_return.notna()
            & lagged_cap.notna()
            & (lagged_cap > 0)
        )
        if int(size_valid.sum()) >= int(min_universe):
            cap = lagged_cap[size_valid]
            returns = daily_return[size_valid]
            q_low = cap.quantile(lower_quantile)
            q_high = cap.quantile(upper_quantile)
            small_return = returns[cap <= q_low].mean()
            big_return = returns[cap >= q_high].mean()
            size_values[position] = small_return - big_return

        bp_valid = (
            daily_return.notna()
            & lagged_pb.notna()
            & (lagged_pb > 0)
        )
        if int(bp_valid.sum()) >= int(min_universe):
            pb_value = lagged_pb[bp_valid]
            returns = daily_return[bp_valid]
            q_low = pb_value.quantile(lower_quantile)
            q_high = pb_value.quantile(upper_quantile)
            low_pb_return = returns[pb_value <= q_low].mean()
            high_pb_return = returns[pb_value >= q_high].mean()
            bp_values[position] = low_pb_return - high_pb_return

        completed = position + 1
        refresh = (
            completed == 2
            or completed % int(progress_every) == 0
            or completed == len(dates)
        )
        if show_progress and refresh:
            elapsed = time.perf_counter() - started_at
            remaining = elapsed / completed * (len(dates) - completed)
            print(
                "\r[id2_std_nm] [1/2] 构造风格因子收益 "
                f"{completed}/{len(dates)} "
                f"| {completed / len(dates):.1%} "
                f"| 当前：{dates[position]:%Y-%m-%d} "
                f"| 已耗时：{elapsed:.1f}s "
                f"| 预计剩余：{remaining:.1f}s",
                end="",
                flush=True,
            )

    return pd.DataFrame(
        {
            "size_return": size_values,
            "bp_return": bp_values,
        },
        index=dates,
    )


def _three_factor_residual_std(y_matrix, x_matrix, min_obs):
    """逐股票执行三因子OLS，并按Notebook使用残差样本标准差。"""
    y_matrix = np.asarray(y_matrix, dtype=float)
    x_matrix = np.asarray(x_matrix, dtype=float)
    result = np.full(y_matrix.shape[1], np.nan, dtype=float)

    for column in range(y_matrix.shape[1]):
        y = y_matrix[:, column]
        valid = np.isfinite(y) & np.isfinite(x_matrix).all(axis=1)
        if int(valid.sum()) < int(min_obs):
            continue
        y_valid = y[valid]
        x_valid = x_matrix[valid]
        design = np.column_stack(
            [np.ones(len(x_valid), dtype=float), x_valid]
        )
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        try:
            beta = np.linalg.lstsq(design, y_valid, rcond=None)[0]
            residual = y_valid - design @ beta
            if len(residual) > 1:
                value = np.std(residual, ddof=1)
                if np.isfinite(value):
                    result[column] = value
        except np.linalg.LinAlgError:
            continue
    return result


def _collapse_small_industries(industry, minimum_count):
    values = industry.copy()
    valid = values.dropna().astype(str)
    counts = valid.value_counts()
    small = counts[counts < int(minimum_count)].index
    values.loc[valid.index] = valid.where(
        ~valid.isin(small),
        "Other",
    )
    return values


def calc_id2_std_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=3,
    trading_days_per_month=21,
    market_index="csi_all_share",
    min_ts_observations=40,
    style_lower_quantile=0.30,
    style_upper_quantile=0.70,
    min_style_universe=200,
    neutralize_industry=True,
    min_industry_count=5,
    min_cs_count=200,
    standardize_residual=False,
    show_progress=False,
    progress_every=20,
    domain_data=None,
):
    """计算市值行业中性化后的 id2_std_nm。

    时间序列回归为：

    ``stock_return = alpha + beta_mkt * market_return``
    ``+ beta_size * size_return + beta_bp * bp_return + epsilon``

    `size_return` 使用上一交易日总市值构造“小市值组合收益减大市值
    组合收益”；`bp_return` 使用上一交易日PB构造“低PB组合收益减高PB
    组合收益”。股票面板由 ``data`` 提供，市场指数面板由
    ``domain_data`` 提供；市场收益率在因子内部计算。原始因子为
    回归残差的样本标准差，最后进行市值行业中性化。
    """
    started_at = time.perf_counter()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    for name, value in {
        "show_progress": show_progress,
        "neutralize_industry": neutralize_industry,
        "standardize_residual": standardize_residual,
    }.items():
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} 必须是 bool。")
    for name, value in {
        "progress_every": progress_every,
        "min_style_universe": min_style_universe,
        "min_industry_count": min_industry_count,
        "min_cs_count": min_cs_count,
    }.items():
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} 必须是正整数。")
    if not (
        np.isfinite(style_lower_quantile)
        and np.isfinite(style_upper_quantile)
        and 0 < float(style_lower_quantile)
        < float(style_upper_quantile) < 1
    ):
        raise ValueError(
            "风格分组分位点必须满足0 < lower < upper < 1。"
        )

    window_info = _resolve_id2_std_nm_data_window(
        {
            "n_months": n_months,
            "trading_days_per_month": trading_days_per_month,
        }
    )
    window = window_info["lookback_trading_days"]
    if (
        not isinstance(min_ts_observations, (int, np.integer))
        or isinstance(min_ts_observations, (bool, np.bool_))
        or not 5 <= int(min_ts_observations) <= window
    ):
        raise ValueError(
            "min_ts_observations 必须是5至回归窗口长度之间的整数。"
        )

    required = {
        "date",
        "instrument",
        "close",
        "total_market_cap",
        "pb",
    }
    if neutralize_industry:
        required.add("industry")
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"id2_std_nm 缺少字段：{sorted(missing)}")

    df = data.loc[:, list(required)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("date 存在无法解析的日期或缺失值。")
    if df["instrument"].isna().any():
        raise ValueError("instrument 不允许缺失。")
    df["instrument"] = df["instrument"].astype(str)
    if df.duplicated(["date", "instrument"]).any():
        raise ValueError("data 存在重复的 date + instrument。")
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date).normalize()
        df = df[df["date"] <= cutoff].copy()
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in [
        "close",
        "total_market_cap",
        "pb",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    target_index = _normalize_target_dates(df["date"], target_dates)
    if target_index.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    all_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    all_instruments = pd.Index(df["instrument"].unique()).sort_values()
    close_wide = (
        df.pivot(index="date", columns="instrument", values="close")
        .reindex(index=all_dates, columns=all_instruments)
    )
    market_cap_wide = (
        df.pivot(
            index="date",
            columns="instrument",
            values="total_market_cap",
        )
        .reindex(index=all_dates, columns=all_instruments)
    )
    pb_wide = (
        df.pivot(index="date", columns="instrument", values="pb")
        .reindex(index=all_dates, columns=all_instruments)
    )
    return_wide = close_wide.pct_change(fill_method=None)
    market_close = _get_market_close(
        domain_data,
        market_index,
        as_of_date=as_of_date,
    ).reindex(all_dates)
    missing_market_dates = market_close.index[market_close.isna()]
    if len(missing_market_dates) > 0:
        preview = [
            date.strftime("%Y-%m-%d")
            for date in missing_market_dates[:5]
        ]
        raise ValueError(
            f"指数 {market_index!r} 缺少或存在无效收盘价的日期："
            f"{preview}。"
        )
    market_return = (
        market_close.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .rename("market_return")
    )
    style_returns = _build_style_factor_returns(
        stock_return=return_wide,
        market_cap=market_cap_wide,
        pb=pb_wide,
        lower_quantile=float(style_lower_quantile),
        upper_quantile=float(style_upper_quantile),
        min_universe=int(min_style_universe),
        show_progress=show_progress,
        progress_every=int(progress_every),
        started_at=started_at,
    )
    factor_returns = pd.concat(
        [market_return.rename("market_return"), style_returns],
        axis=1,
    ).reindex(all_dates)

    target_state = df[df["date"].isin(target_index)].copy()
    result_parts = []
    total_dates = len(target_index)

    try:
        for position, date in enumerate(target_index, start=1):
            date_position = all_dates.get_indexer([date])[0]
            first_position = date_position - window + 1
            raw_values = np.full(len(all_instruments), np.nan, dtype=float)
            if first_position > 0:
                window_dates = all_dates[first_position:date_position + 1]
                y_matrix = return_wide.loc[window_dates].to_numpy(dtype=float)
                x_matrix = factor_returns.loc[
                    window_dates,
                    ["market_return", "size_return", "bp_return"],
                ].to_numpy(dtype=float)
                raw_values = _three_factor_residual_std(
                    y_matrix,
                    x_matrix,
                    int(min_ts_observations),
                )

            raw_factor = pd.Series(
                raw_values,
                index=all_instruments,
                name="factor_raw",
            )
            cross_section = (
                target_state[target_state["date"] == date]
                .drop_duplicates("instrument", keep="last")
                .set_index("instrument")
                .reindex(all_instruments)
            )
            industry = None
            if neutralize_industry:
                industry = _collapse_small_industries(
                    cross_section["industry"],
                    int(min_industry_count),
                )
            factor = neutralize_size_industry(
                target=raw_factor,
                market_cap=cross_section["total_market_cap"],
                industry=industry,
                min_obs=int(min_cs_count),
                standardize_residual=standardize_residual,
                zscore_ddof=1,
                show_progress=False,
            ).replace([np.inf, -np.inf], np.nan)
            result_parts.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "instrument": all_instruments.to_numpy(),
                        "id2_std_nm": factor.to_numpy(),
                    }
                )
            )

            refresh = (
                position == 1
                or position % int(progress_every) == 0
                or position == total_dates
            )
            if show_progress and refresh:
                elapsed = time.perf_counter() - started_at
                remaining = elapsed / position * (total_dates - position)
                print(
                    "\r[id2_std_nm] [2/2] 回归和中性化 "
                    f"{position}/{total_dates} "
                    f"| {position / total_dates:.1%} "
                    f"| 当前：{date:%Y-%m-%d} "
                    f"| 已耗时：{elapsed:.1f}s "
                    f"| 预计剩余：{remaining:.1f}s",
                    end="",
                    flush=True,
                )

        return (
            pd.concat(result_parts, ignore_index=True)
            .sort_values(["date", "instrument"], kind="mergesort")
            .reset_index(drop=True)
        )
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": "id2_std_nm",
    "func": calc_id2_std_nm,
    "category": "volatility",
    "direction": -1,
    "description": (
        "个股N个月日收益对市场、规模和BP三个日收益因子回归后的"
        "残差样本标准差，再进行市值行业中性化。"
    ),
    "formula": (
        "size_return=small_cap_return-big_cap_return；"
        "bp_return=low_pb_return-high_pb_return；"
        "r_i=alpha+beta_mkt*r_m+beta_size*size_return+beta_bp*bp_return+epsilon；"
        "raw=StdSamp(epsilon)；最终对log(total_market_cap)和行业哑变量"
        "做截面回归。分组使用上一交易日市值和PB。"
    ),
    "input_schema": {
        "required": {
            "date": {
                "dtype": "datetime64[ns] 或可解析日期",
                "meaning": "日频观测日期及目标因子截面日期。",
            },
            "instrument": {
                "dtype": "string",
                "meaning": "证券唯一标识。",
            },
            "close": {
                "dtype": "float",
                "meaning": "复权口径一致的股票日收盘价。",
            },
            "total_market_cap": {
                "dtype": "float",
                "meaning": (
                    "日总市值；滞后一期构造规模因子，目标日用于中性化。"
                ),
            },
            "pb": {
                "dtype": "float",
                "meaning": "日PB；滞后一期构造高BP减低BP近似因子。",
            },
            "market_close": {
                "dtype": "float",
                "meaning": (
                    "market_index 参数指定的市场代理指数原始日收盘点位；"
                    "因子内部计算日收益率。"
                ),
            },
        },
        "conditional": {
            "industry": {
                "dtype": "string",
                "meaning": "目标日点时可得的一级行业分类。",
                "required_when": {"neutralize_industry": True},
            },
        },
    },
    "parameters": {
        "target_dates": {
            "default": None,
            "accepted_values": "单个日期、日期序列或None。",
            "effect": "指定实际输出因子截面。",
            "changes_data_requirements": True,
        },
        "as_of_date": {
            "default": None,
            "accepted_values": "可解析日期或None。",
            "effect": "截断晚于信息截止日的数据。",
            "changes_data_requirements": False,
        },
        "n_months": {
            "default": 3,
            "accepted_values": "正整数。",
            "effect": "决定三因子时间序列回归月份窗口。",
            "changes_data_requirements": True,
        },
        "trading_days_per_month": {
            "default": 21,
            "accepted_values": "正整数。",
            "effect": "将月份换算为交易日。",
            "changes_data_requirements": True,
        },
        "market_index": {
            "default": "csi_all_share",
            "accepted_values": "数据适配层支持的统一市场指数名称。",
            "effect": "指定三因子回归使用的市场代理指数。",
            "changes_data_requirements": True,
        },
        "min_ts_observations": {
            "default": 40,
            "accepted_values": "5至回归窗口长度之间的整数。",
            "effect": "单只股票三因子回归的最少完整观测数。",
            "changes_data_requirements": False,
        },
        "style_lower_quantile": {
            "default": 0.30,
            "accepted_values": "0至1之间且小于upper。",
            "effect": "规模和PB低组边界。",
            "changes_data_requirements": False,
        },
        "style_upper_quantile": {
            "default": 0.70,
            "accepted_values": "0至1之间且大于lower。",
            "effect": "规模和PB高组边界。",
            "changes_data_requirements": False,
        },
        "min_style_universe": {
            "default": 200,
            "accepted_values": "正整数。",
            "effect": "每日构造规模和BP因子收益的最少股票数。",
            "changes_data_requirements": False,
        },
        "neutralize_industry": {
            "default": True,
            "accepted_values": [True, False],
            "effect": "True为市值+行业中性化；False仅做市值中性化。",
            "changes_data_requirements": True,
        },
        "min_industry_count": {
            "default": 5,
            "accepted_values": "正整数。",
            "effect": "目标日样本过少的行业合并为Other。",
            "changes_data_requirements": False,
        },
        "min_cs_count": {
            "default": 200,
            "accepted_values": "正整数。",
            "effect": "目标日市值行业中性化的最少有效股票数。",
            "changes_data_requirements": False,
        },
        "standardize_residual": {
            "default": False,
            "accepted_values": [True, False],
            "effect": "是否把中性化残差继续做样本Z-score。",
            "changes_data_requirements": False,
        },
        "show_progress": {
            "default": False,
            "accepted_values": [True, False],
            "effect": "只控制进度显示，不改变计算结果。",
            "changes_data_requirements": False,
        },
        "progress_every": {
            "default": 20,
            "accepted_values": "正整数。",
            "effect": "风格收益构造和目标截面的进度刷新间隔。",
            "changes_data_requirements": False,
        },
    },
    "data_window": {
        "resolver": _resolve_id2_std_nm_data_window,
        "default": {
            "lookback_trading_days": 63,
            "requires_target_date_data": True,
            "minimum_history_observations": 63,
            "preheating_required": True,
            "insufficient_window_behavior": (
                "默认63日窗口，完整三因子回归观测不足40日时输出NaN。"
            ),
        },
        "resolver_notes": (
            "L个日收益以及首日风格分组所用滞后市值/PB均要求目标日前"
            "至少L个交易日原始数据。"
        ),
    },
    "output_schema": {
        "date": {"dtype": "datetime64[ns]", "meaning": "目标截面日期。"},
        "instrument": {"dtype": "string", "meaning": "证券唯一标识。"},
        "id2_std_nm": {
            "dtype": "float64",
            "meaning": "市值行业中性化后的三因子特质波动率。",
        },
    },
    "usage_notes": [
        "n_months=3、trading_days_per_month=21对应原id2_std_3m。",
        "默认market_index='csi_all_share'，对应原Notebook的中证全指。",
        "market_close由适配器独立加载，不会广播到每只股票行。",
        "市场收益率只在因子内部由原始指数收盘点位计算。",
        "规模和BP因子收益必须在完整的全A股形成股票池上构造，不能只用策略候选子集。",
        "中性化结果依赖目标日截面股票池；预热不足产生的NaN不得填0。",
    ],
    "best_practice": {
        "instance_name": "id2_std_3m",
        "parameters": {
            "n_months": 3,
            "trading_days_per_month": 21,
            "market_index": "csi_all_share",
            "min_ts_observations": 40,
            "style_lower_quantile": 0.30,
            "style_upper_quantile": 0.70,
            "min_style_universe": 200,
            "neutralize_industry": True,
            "min_industry_count": 5,
            "min_cs_count": 200,
            "standardize_residual": False,
        },
        "description": "原Notebook的3个月市值行业中性化三因子特质波动率。",
    },
    "pit_notes": [
        "股票收益和由market_close计算的市场收益只使用目标日及以前数据。",
        "规模和BP因子收益使用前一交易日市值与PB进行分组。",
        "市场指数、行业、总市值和PB必须按历史日期提供。",
        "目标日收盘价参与计算，因此信号最早在目标日收盘后形成。",
    ],
    "references": ["Bigquant_research：id2_std_3m因子.ipynb。"],
    "tags": [
        "volatility",
        "idiosyncratic_volatility",
        "three_factor_model",
        "size_neutralized",
        "industry_neutralized",
        "parameterized_window",
    ],
    "status": "research",
    "version": "2.0.0",
}
