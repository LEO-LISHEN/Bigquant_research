# -*- coding: utf-8 -*-
"""单市场因子模型特质波动率。"""

import time

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_size_industry import (
    neutralize_size_industry,
)


OUTPUT_COLUMNS = ["date", "instrument", "id1_std_nm"]


def _resolve_id1_std_nm_data_window(resolved_params):
    """根据月份参数解析价格和回归窗口。"""
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
            "回归窗口内有效股票收益与市场收益配对数不足"
            "min_ts_observations时，该股票目标日因子值输出NaN。"
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
            f"id1_std_nm 缺少目标日期原始数据：{preview}。"
        )
    return normalized


def _get_market_close(domain_data, market_index, as_of_date=None):
    """从分域容器读取指定市场指数的原始收盘点位。"""
    if domain_data is None or not hasattr(domain_data, "get_domain"):
        raise TypeError(
            "id1_std_nm 需要包含 market_daily 的 FactorDataBundle；"
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


def _capm_residual_standard_error(y_matrix, market_return, min_obs):
    """按Notebook的SSE/(n-2)公式批量计算CAPM残差标准误。"""
    y = np.asarray(y_matrix, dtype=float)
    x = np.asarray(market_return, dtype=float).reshape(-1, 1)
    valid = np.isfinite(y) & np.isfinite(x)

    n = valid.sum(axis=0).astype(float)
    x_full = np.broadcast_to(x, y.shape)
    safe_x = np.where(valid, x_full, 0.0)
    safe_y = np.where(valid, y, 0.0)

    sum_x = safe_x.sum(axis=0)
    sum_y = safe_y.sum(axis=0)
    sum_xx = (safe_x * safe_x).sum(axis=0)
    sum_xy = (safe_x * safe_y).sum(axis=0)
    sum_yy = (safe_y * safe_y).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = sum_xx - sum_x * sum_x / n
        beta = (sum_xy - sum_x * sum_y / n) / denom
        alpha = sum_y / n - beta * sum_x / n
        sse = (
            sum_yy
            + n * alpha * alpha
            + beta * beta * sum_xx
            - 2.0 * alpha * sum_y
            - 2.0 * beta * sum_xy
            + 2.0 * alpha * beta * sum_x
        )
        residual_variance = sse / (n - 2.0)

    tolerance = 1e-14
    residual_variance = np.where(
        (residual_variance < 0)
        & (residual_variance > -tolerance),
        0.0,
        residual_variance,
    )
    usable = (
        (n >= int(min_obs))
        & (n > 2)
        & np.isfinite(denom)
        & (denom > 0)
        & np.isfinite(residual_variance)
        & (residual_variance >= 0)
    )
    result = np.full(y.shape[1], np.nan, dtype=float)
    result[usable] = np.sqrt(residual_variance[usable])
    return result


def calc_id1_std_nm(
    data,
    target_dates=None,
    as_of_date=None,
    n_months=3,
    trading_days_per_month=21,
    market_index="csi_all_share",
    min_ts_observations=50,
    neutralize_industry=True,
    min_cs_count=80,
    standardize_residual=True,
    show_progress=False,
    progress_every=20,
    domain_data=None,
):
    """计算市值行业中性化后的 id1_std_nm。

    对每只股票最近 ``L`` 个交易日的日收益进行滚动CAPM回归：

    ``stock_return = alpha + beta * market_return + epsilon``

    原始因子为 ``sqrt(SSE / (n - 2))``。随后在每个目标日截面
    对 log(总市值) 和行业哑变量回归，输出中性化残差；默认继续
    对截面残差进行Z-score标准化。

    股票面板由 ``data`` 提供；市场指数面板由 ``domain_data`` 提供。
    函数内部根据 ``market_index`` 选取市场原始收盘点位并计算日收益率。
    精确复现原 Notebook 时使用中证全指。
    """
    started_at = time.perf_counter()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if not isinstance(neutralize_industry, (bool, np.bool_)):
        raise TypeError("neutralize_industry 必须是 bool。")
    if not isinstance(standardize_residual, (bool, np.bool_)):
        raise TypeError("standardize_residual 必须是 bool。")
    for name, value in {
        "progress_every": progress_every,
        "min_cs_count": min_cs_count,
    }.items():
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} 必须是正整数。")

    window_info = _resolve_id1_std_nm_data_window(
        {
            "n_months": n_months,
            "trading_days_per_month": trading_days_per_month,
        }
    )
    window = window_info["lookback_trading_days"]
    if (
        not isinstance(min_ts_observations, (int, np.integer))
        or isinstance(min_ts_observations, (bool, np.bool_))
        or not 3 <= int(min_ts_observations) <= window
    ):
        raise ValueError(
            "min_ts_observations 必须是3至回归窗口长度之间的整数。"
        )

    required = {
        "date",
        "instrument",
        "close",
        "volume",
        "amount",
        "suspended",
        "total_market_cap",
    }
    if neutralize_industry:
        required.add("industry")
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"id1_std_nm 缺少字段：{sorted(missing)}")

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
        "volume",
        "amount",
        "suspended",
        "total_market_cap",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    target_index = _normalize_target_dates(df["date"], target_dates)
    if target_index.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.sort_values(
        ["instrument", "date"],
        kind="mergesort",
    ).reset_index(drop=True)
    df["stock_return"] = (
        df.groupby("instrument", sort=False)["close"]
        .pct_change(fill_method=None)
    )
    valid_trade = (
        (df["suspended"] == 0)
        & (df["volume"] > 0)
        & (df["amount"] > 0)
        & np.isfinite(df["stock_return"])
    )
    df.loc[~valid_trade, "stock_return"] = np.nan

    all_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    all_instruments = pd.Index(
        df["instrument"].astype(str).unique()
    ).sort_values()
    return_wide = (
        df.pivot(index="date", columns="instrument", values="stock_return")
        .reindex(index=all_dates, columns=all_instruments)
    )
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

    target_state = df[df["date"].isin(target_index)].copy()
    target_state["instrument"] = target_state["instrument"].astype(str)
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
                x_vector = market_return.loc[window_dates].to_numpy(dtype=float)
                raw_values = _capm_residual_standard_error(
                    y_matrix,
                    x_vector,
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
            industry = (
                cross_section["industry"]
                if neutralize_industry
                else None
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
                        "id1_std_nm": factor.to_numpy(),
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
                    "\r[id1_std_nm] "
                    f"{position}/{total_dates} 个截面 "
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
    "name": "id1_std_nm",
    "func": calc_id1_std_nm,
    "category": "volatility",
    "direction": -1,
    "description": (
        "个股N个月日收益对市场日收益做滚动CAPM回归后的残差标准误，"
        "再进行市值行业中性化；数值越低通常越优。"
    ),
    "formula": (
        "L=n_months*trading_days_per_month；r_i=alpha+beta*r_m+epsilon；"
        "raw=sqrt(SSE/(n-2))；最终对log(total_market_cap)和行业哑变量"
        "做截面回归，默认输出残差样本Z-score。"
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
            "volume": {
                "dtype": "float",
                "meaning": "成交量；无成交日不参与时间序列回归。",
            },
            "amount": {
                "dtype": "float",
                "meaning": "成交额；无成交日不参与时间序列回归。",
            },
            "suspended": {
                "dtype": "int/bool",
                "meaning": "历史停牌状态；停牌日不参与回归。",
            },
            "total_market_cap": {
                "dtype": "float",
                "meaning": "目标日总市值，中性化时取自然对数。",
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
            "effect": "决定CAPM滚动回归的月份窗口。",
            "changes_data_requirements": True,
        },
        "trading_days_per_month": {
            "default": 21,
            "accepted_values": "正整数。",
            "effect": "将月份换算成交易日。",
            "changes_data_requirements": True,
        },
        "market_index": {
            "default": "csi_all_share",
            "accepted_values": "数据适配层支持的统一市场指数名称。",
            "effect": "指定CAPM回归使用的市场代理指数。",
            "changes_data_requirements": True,
        },
        "min_ts_observations": {
            "default": 50,
            "accepted_values": "3至回归窗口长度之间的整数。",
            "effect": "单只股票滚动回归的最少有效配对观测数。",
            "changes_data_requirements": False,
        },
        "neutralize_industry": {
            "default": True,
            "accepted_values": [True, False],
            "effect": "True为市值+行业中性化；False仅做市值中性化。",
            "changes_data_requirements": True,
        },
        "min_cs_count": {
            "default": 80,
            "accepted_values": "正整数。",
            "effect": "目标日中性化的最少有效股票数。",
            "changes_data_requirements": False,
        },
        "standardize_residual": {
            "default": True,
            "accepted_values": [True, False],
            "effect": "是否把市值行业中性化残差继续做样本Z-score。",
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
            "effect": "进度刷新间隔，单位为目标截面数。",
            "changes_data_requirements": False,
        },
    },
    "data_window": {
        "resolver": _resolve_id1_std_nm_data_window,
        "default": {
            "lookback_trading_days": 63,
            "requires_target_date_data": True,
            "minimum_history_observations": 63,
            "preheating_required": True,
            "insufficient_window_behavior": (
                "默认63日回归窗口，有效配对观测不足50日时输出NaN。"
            ),
        },
        "resolver_notes": (
            "计算L个日收益需要目标日前至少L个交易日价格，因此预热为L日。"
        ),
    },
    "output_schema": {
        "date": {"dtype": "datetime64[ns]", "meaning": "目标截面日期。"},
        "instrument": {"dtype": "string", "meaning": "证券唯一标识。"},
        "id1_std_nm": {
            "dtype": "float64",
            "meaning": "市值行业中性化后的单市场因子特质波动率。",
        },
    },
    "usage_notes": [
        "n_months=3、trading_days_per_month=21对应原id1_std_3m。",
        "默认market_index='csi_all_share'，对应原Notebook的中证全指。",
        "market_close由适配器独立加载，不会广播到每只股票行。",
        "市场收益率只在因子内部由原始指数收盘点位计算。",
        "因子层只把停牌和无成交日排除出波动率估计，不负责策略股票池过滤。",
        "中性化股票池会影响结果，跨研究比较时必须保持一致。",
    ],
    "best_practice": {
        "instance_name": "id1_std_3m",
        "parameters": {
            "n_months": 3,
            "trading_days_per_month": 21,
            "market_index": "csi_all_share",
            "min_ts_observations": 50,
            "neutralize_industry": True,
            "min_cs_count": 80,
            "standardize_residual": True,
        },
        "description": (
            "3个月CAPM特质波动率，并按本次迁移要求加入市值行业中性化。"
        ),
    },
    "pit_notes": [
        "股票收益和由market_close计算的市场收益只使用目标日及以前数据。",
        "市场指数、行业、停牌状态和总市值必须按历史日期提供。",
        "目标日收盘价参与计算，因此信号最早在目标日收盘后形成。",
    ],
    "references": ["Bigquant_research：id1_std_3m因子.ipynb。"],
    "tags": [
        "volatility",
        "idiosyncratic_volatility",
        "capm",
        "size_neutralized",
        "industry_neutralized",
        "parameterized_window",
    ],
    "status": "research",
    "version": "2.0.0",
}
