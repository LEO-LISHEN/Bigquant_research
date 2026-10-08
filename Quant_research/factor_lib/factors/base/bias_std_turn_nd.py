# -*- coding: utf-8 -*-
"""华泰换手率波动偏离因子。"""

import time

import numpy as np
import pandas as pd

from factor_lib.common.preprocess.neutralize_size_industry import (
    neutralize_size_industry,
)
from factor_lib.common.preprocess.winsorize_mad import winsorize_mad
from factor_lib.common.preprocess.zscore import zscore


OUTPUT_COLUMNS = [
    "date",
    "instrument",
    "bias_std_turn_nd",
]


def _resolve_bias_std_turn_nd_data_window(resolved_params):
    """根据短、长窗口参数解析目标日所需历史数据。"""
    short_window = resolved_params.get("short_window", 5)
    long_window = resolved_params.get("long_window", 504)

    for name, value in {
        "short_window": short_window,
        "long_window": long_window,
    }.items():
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) < 2
        ):
            raise ValueError(f"{name} 必须是至少为2的整数。")

    short_window = int(short_window)
    long_window = int(long_window)
    if long_window <= short_window:
        raise ValueError(
            "long_window 必须大于 short_window。"
        )

    # 滚动窗口包含目标日，因此L日窗口最多需要目标日前L-1个交易日。
    lookback_days = long_window - 1
    return {
        "lookback_trading_days": lookback_days,
        "requires_target_date_data": True,
        "minimum_history_observations": lookback_days,
        "preheating_required": True,
        "insufficient_window_behavior": (
            "目标日前可用交易日不足长窗口，或窗口内正换手率观测数"
            "低于 min_long_observations 时，该股票因子值输出 NaN。"
        ),
    }


def _normalize_target_dates(data_dates, target_dates):
    """规范目标日期，并检查目标截面是否存在于准备数据中。"""
    available_dates = pd.DatetimeIndex(
        data_dates.dropna().unique()
    ).sort_values()

    if target_dates is None:
        return available_dates

    if isinstance(target_dates, (str, pd.Timestamp)):
        target_dates = [target_dates]
    else:
        try:
            target_dates = list(target_dates)
        except TypeError:
            target_dates = [target_dates]

    normalized = pd.DatetimeIndex(
        pd.to_datetime(
            target_dates,
            errors="raise",
        )
    ).normalize().unique().sort_values()
    if normalized.empty:
        return normalized

    missing = normalized.difference(available_dates)
    if not missing.empty:
        preview = [
            date.strftime("%Y-%m-%d")
            for date in missing[:5]
        ]
        raise ValueError(
            "bias_std_turn_nd 因子缺少目标日期原始数据："
            f"{preview}。请检查预存日期和 as_of_date。"
        )
    return normalized


def calc_bias_std_turn_nd(
    data,
    target_dates=None,
    as_of_date=None,
    short_window=5,
    long_window=504,
    min_short_observations=2,
    min_long_observations=2,
    neutralize_industry=True,
    winsor_k=5.0,
    min_cs_count=80,
    show_progress=False,
    progress_every=20,
):
    """计算市值、行业中性化后的换手率波动偏离因子。

    原始定义：

    ``raw = std(turn, short_window)
            / std(turn, long_window) - 1``

    其中 ``turn <= 0`` 的观测不参与滚动样本标准差。随后逐目标日：

    ``MAD去极值 → Z-score → log(总市值)+行业OLS中性化
    → 中性化残差Z-score``

    中性化残差不会再次去极值，避免破坏其与控制变量的截面正交性。
    因子保持华泰研究原始方向，不乘负号；数值越低越优。

    参数
    ----
    data : pandas.DataFrame
        必须包含 date、instrument、turn、total_market_cap；当
        neutralize_industry=True 时还必须包含 industry。data 应覆盖
        目标日和 long_window 所需的历史预热区间。
    target_dates : 日期或日期序列，可选
        实际输出截面；None 表示输出 data 中的全部日期。
    as_of_date : 日期，可选
        全局信息截止日，晚于该日的数据不会参与计算。
    short_window : int，默认 5
        短期换手率样本标准差窗口。
    long_window : int，默认 504
        长期换手率样本标准差窗口，必须大于 short_window。
    min_short_observations : int，默认 2
        短窗口内最少正换手率观测数。默认2与样本标准差的最低要求
        一致，并复现旧notebook的 m_stddev 非空观测口径。
    min_long_observations : int，默认 2
        长窗口内最少正换手率观测数。若希望严格要求完整504日，
        可传入504。
    neutralize_industry : bool，默认 True
        True为市值+行业中性化；False为仅市值中性化。
    winsor_k : float，默认 5.0
        原始因子截面MAD去极值倍数。
    min_cs_count : int，默认 80
        单日中性化所需最少有效股票数。
    show_progress : bool，默认 False
        是否使用终端单行刷新显示计算进度。
    progress_every : int，默认 20
        每处理多少个目标截面刷新一次进度。

    返回
    ----
    pandas.DataFrame
        date、instrument、bias_std_turn_nd三列。预热不足、分母无效、
        市值/行业缺失或中性化样本不足时保留NaN，不填0。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")

    resolved_window = _resolve_bias_std_turn_nd_data_window(
        {
            "short_window": short_window,
            "long_window": long_window,
        }
    )
    short_window = int(short_window)
    long_window = int(long_window)

    for name, value, upper_bound in [
        (
            "min_short_observations",
            min_short_observations,
            short_window,
        ),
        (
            "min_long_observations",
            min_long_observations,
            long_window,
        ),
    ]:
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or not 2 <= int(value) <= upper_bound
        ):
            raise ValueError(
                f"{name} 必须是[2, {upper_bound}]内的整数。"
            )

    if not isinstance(
        neutralize_industry,
        (bool, np.bool_),
    ):
        raise TypeError("neutralize_industry 必须是 bool。")
    if (
        not isinstance(winsor_k, (int, float, np.number))
        or isinstance(winsor_k, (bool, np.bool_))
        or not np.isfinite(winsor_k)
        or float(winsor_k) <= 0
    ):
        raise ValueError("winsor_k 必须是正数。")
    if (
        not isinstance(min_cs_count, (int, np.integer))
        or isinstance(min_cs_count, (bool, np.bool_))
        or int(min_cs_count) <= 0
    ):
        raise ValueError("min_cs_count 必须是正整数。")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress 必须是 bool。")
    if (
        not isinstance(progress_every, (int, np.integer))
        or isinstance(progress_every, (bool, np.bool_))
        or int(progress_every) <= 0
    ):
        raise ValueError("progress_every 必须是正整数。")

    required_columns = {
        "date",
        "instrument",
        "turn",
        "total_market_cap",
    }
    if neutralize_industry:
        required_columns.add("industry")

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "bias_std_turn_nd 因子缺少字段："
            f"{sorted(missing_columns)}"
        )

    selected_columns = [
        "date",
        "instrument",
        "turn",
        "total_market_cap",
    ]
    if neutralize_industry:
        selected_columns.append("industry")

    df = data.loc[:, selected_columns].copy()
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()
    if df["date"].isna().any():
        raise ValueError(
            "bias_std_turn_nd 的 date 存在无效值。"
        )
    if df["instrument"].isna().any():
        raise ValueError(
            "bias_std_turn_nd 的 instrument 不允许缺失。"
        )

    duplicated = df.duplicated(
        ["date", "instrument"],
        keep=False,
    )
    if duplicated.any():
        examples = (
            df.loc[
                duplicated,
                ["date", "instrument"],
            ]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            "bias_std_turn_nd 输入存在重复的 "
            f"date + instrument：{examples}"
        )

    df["turn"] = pd.to_numeric(
        df["turn"],
        errors="coerce",
    )
    df["total_market_cap"] = pd.to_numeric(
        df["total_market_cap"],
        errors="coerce",
    )

    if as_of_date is not None:
        as_of_timestamp = pd.Timestamp(as_of_date)
        if pd.isna(as_of_timestamp):
            raise ValueError(
                "as_of_date 必须是可解析日期。"
            )
        df = df.loc[
            df["date"] <= as_of_timestamp.normalize()
        ].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    target_date_index = _normalize_target_dates(
        df["date"],
        target_dates,
    )
    if target_date_index.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.sort_values(
        ["instrument", "date"],
        kind="mergesort",
    ).reset_index(drop=True)
    df["_valid_turn"] = df["turn"].where(
        np.isfinite(df["turn"])
        & (df["turn"] > 0)
    )

    started_at = time.perf_counter()
    if show_progress:
        print(
            "\r[bias_std_turn_nd] [1/2] "
            f"计算{short_window}/{long_window}日滚动换手率标准差...",
            end="",
            flush=True,
        )

    try:
        grouped_turn = df.groupby(
            "instrument",
            sort=False,
        )["_valid_turn"]
        df["_turn_std_short"] = (
            grouped_turn
            .rolling(
                window=short_window,
                min_periods=int(min_short_observations),
            )
            .std(ddof=1)
            .reset_index(level=0, drop=True)
        )
        df["_turn_std_long"] = (
            grouped_turn
            .rolling(
                window=long_window,
                min_periods=int(min_long_observations),
            )
            .std(ddof=1)
            .reset_index(level=0, drop=True)
        )

        valid_denominator = (
            np.isfinite(df["_turn_std_long"])
            & (df["_turn_std_long"] > 0)
        )
        df["_factor_raw"] = np.nan
        df.loc[
            valid_denominator,
            "_factor_raw",
        ] = (
            df.loc[
                valid_denominator,
                "_turn_std_short",
            ]
            / df.loc[
                valid_denominator,
                "_turn_std_long",
            ]
            - 1.0
        )
        df["_factor_raw"] = df["_factor_raw"].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        target_panel = df.loc[
            df["date"].isin(target_date_index)
        ].copy()
        grouped_dates = list(
            target_panel.groupby("date", sort=True)
        )
        total_dates = len(grouped_dates)
        result_parts = []

        for position, (date, section) in enumerate(
            grouped_dates,
            start=1,
        ):
            section = section.copy()

            raw_z = zscore(
                winsorize_mad(
                    section["_factor_raw"],
                    k=float(winsor_k),
                ),
                ddof=1,
                show_progress=False,
            )

            neutralized = neutralize_size_industry(
                target=raw_z,
                market_cap=section["total_market_cap"],
                industry=(
                    section["industry"]
                    if neutralize_industry
                    else None
                ),
                min_obs=int(min_cs_count),
                standardize_residual=True,
                zscore_ddof=1,
                show_progress=False,
            )

            result_parts.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "instrument": section[
                            "instrument"
                        ].to_numpy(),
                        "bias_std_turn_nd": neutralized.to_numpy(),
                    }
                )
            )

            should_refresh = (
                position == 1
                or position % int(progress_every) == 0
                or position == total_dates
            )
            if show_progress and should_refresh:
                elapsed = time.perf_counter() - started_at
                remaining = (
                    elapsed / position
                    * (total_dates - position)
                )
                print(
                    "\r[bias_std_turn_nd] [2/2] "
                    f"{position}/{total_dates} 个截面 "
                    f"| {position / total_dates:.1%} "
                    f"| 当前：{date:%Y-%m-%d} "
                    f"| 已耗时：{elapsed:.1f}s "
                    f"| 预计剩余：{remaining:.1f}s",
                    end="",
                    flush=True,
                )

        if not result_parts:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        result = pd.concat(
            result_parts,
            ignore_index=True,
        )
        return result.sort_values(
            ["date", "instrument"],
            kind="mergesort",
        ).reset_index(drop=True)
    finally:
        if show_progress:
            print()


FACTOR = {
    "name": 'bias_std_turn_nd',
    "func": calc_bias_std_turn_nd,
    "factor_type": "base",
    "candidate_instances": {"5d": {"short_window": 5}},
    "input_schema": {
        "required": {
            'date': {},
            'instrument': {},
            'turn': {},
            'total_market_cap': {},
        },
        "conditional": {
            'industry': {"required_when": {'neutralize_industry': True}},
        },
    },
    "parameters": {
        'target_dates': {"default": None},
        'as_of_date': {"default": None},
        'short_window': {"default": 5},
        'long_window': {"default": 504},
        'min_short_observations': {"default": 2},
        'min_long_observations': {"default": 2},
        'neutralize_industry': {"default": True},
        'winsor_k': {"default": 5.0},
        'min_cs_count': {"default": 80},
        'show_progress': {"default": False},
        'progress_every': {"default": 20},
    },
    "data_window": {
        "resolver": _resolve_bias_std_turn_nd_data_window,
        "default": {
            "lookback_trading_days": 503,
            "requires_target_date_data": True,
            "minimum_history_observations": 503,
            "preheating_required": True,
        },
    },
    "output_schema": {
        'date': {},
        'instrument': {},
        'bias_std_turn_nd': {},
    },
}


FACTOR_INFO = """
# 换手率偏离长期波动（N 日）

衡量短期换手率相对长期换手率波动基准的偏离程度，并进行市值与可选行业中性化。数值越高表示近期交易活跃度偏离更明显。

- **计算**：短窗与长窗由参数决定，因而预热长度随参数变化。
- **时点**：仅使用目标日及此前的换手率、市值和行业数据。
- **研究提示**：极端换手率可能同时反映事件冲击与流动性风险，应结合交易成本检验。
"""
