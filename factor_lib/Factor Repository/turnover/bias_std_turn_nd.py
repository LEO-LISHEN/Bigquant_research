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
    "name": "bias_std_turn_nd",
    "func": calc_bias_std_turn_nd,
    "category": "turnover",
    "direction": -1,
    "description": (
        "短期换手率波动相对长期换手率波动的偏离程度，"
        "经MAD去极值、标准化及市值行业中性化；"
        "华泰研究中因子值越低越优。"
    ),
    "formula": (
        "raw_t=StdSamp(turn_{t-short_window+1:t}, turn>0)"
        "/StdSamp(turn_{t-long_window+1:t}, turn>0)-1；"
        "raw截面MAD去极值并Z-score后，对log(total_market_cap)"
        "和可选行业哑变量回归，最终因子为残差Z-score。"
    ),
    "input_schema": {
        "required": {
            "date": {
                "dtype": "datetime64[ns] 或可解析日期",
                "meaning": "日频观测日期和目标因子截面日期。",
            },
            "instrument": {
                "dtype": "string",
                "meaning": (
                    "证券唯一标识；同一date+instrument不允许重复。"
                ),
            },
            "turn": {
                "dtype": "float",
                "meaning": (
                    "日换手率；小于等于0的记录不参与滚动标准差。"
                ),
            },
            "total_market_cap": {
                "dtype": "float",
                "meaning": (
                    "目标日总市值；取自然对数后作为截面中性化控制变量。"
                ),
            },
        },
        "conditional": {
            "industry": {
                "dtype": "string",
                "meaning": "目标日一级行业分类，用于构造行业哑变量。",
                "required_when": {
                    "neutralize_industry": True,
                },
            },
        },
    },
    "parameters": {
        "target_dates": {
            "default": None,
            "accepted_values": "单个日期、日期序列或None。",
            "effect": (
                "指定实际输出截面；None输出data中全部日期。"
            ),
            "changes_data_requirements": True,
        },
        "as_of_date": {
            "default": None,
            "accepted_values": "可解析日期或None。",
            "effect": (
                "全局信息截止日，晚于该日的数据不参与计算。"
            ),
            "changes_data_requirements": False,
        },
        "short_window": {
            "default": 5,
            "accepted_values": (
                "至少为2且小于long_window的整数。"
            ),
            "effect": "改变短期换手率波动窗口。",
            "changes_data_requirements": False,
        },
        "long_window": {
            "default": 504,
            "accepted_values": (
                "大于short_window的整数。"
            ),
            "effect": (
                "改变长期基准波动窗口和所需历史预热长度。"
            ),
            "changes_data_requirements": True,
        },
        "min_short_observations": {
            "default": 2,
            "accepted_values": (
                "2至short_window之间的整数。"
            ),
            "effect": (
                "短窗口内正换手率有效观测不足时输出NaN。"
            ),
            "changes_data_requirements": False,
        },
        "min_long_observations": {
            "default": 2,
            "accepted_values": (
                "2至long_window之间的整数。"
            ),
            "effect": (
                "长窗口内正换手率有效观测不足时输出NaN。"
            ),
            "changes_data_requirements": False,
        },
        "neutralize_industry": {
            "default": True,
            "accepted_values": [True, False],
            "effect": (
                "True为市值+行业中性化；False为仅市值中性化。"
            ),
            "changes_data_requirements": True,
        },
        "winsor_k": {
            "default": 5.0,
            "accepted_values": "正数。",
            "effect": "改变原始因子截面MAD去极值边界。",
            "changes_data_requirements": False,
        },
        "min_cs_count": {
            "default": 80,
            "accepted_values": "正整数。",
            "effect": "单日中性化的最小有效股票数。",
            "changes_data_requirements": False,
        },
        "show_progress": {
            "default": False,
            "accepted_values": [True, False],
            "effect": "仅控制进度显示，不改变计算结果。",
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
        "resolver": _resolve_bias_std_turn_nd_data_window,
        "default": {
            "lookback_trading_days": 503,
            "requires_target_date_data": True,
            "minimum_history_observations": 503,
            "preheating_required": True,
            "insufficient_window_behavior": (
                "默认504日长窗口需要目标日前503个交易日；"
                "历史或正换手率观测不足时输出NaN。"
            ),
        },
        "resolver_notes": (
            "实际预热为long_window-1个目标日前交易日；"
            "策略和评价函数必须使用本次resolved_factor_params解析。"
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
        "bias_std_turn_nd": {
            "dtype": "float64",
            "meaning": (
                "市值行业中性化后的换手率波动偏离因子；"
                "数值越低，按原华泰研究定义越优。"
            ),
        },
    },
    "usage_notes": [
        (
            "short_window=5、long_window=504对应旧notebook的"
            "bias_std_turn_5d；short_window=11可复现敏感性测试中的"
            "bias_std_turn_11d，无需新增因子脚本。"
        ),
        (
            "因子层不执行ST、停牌、上市天数、板块或策略股票池过滤；"
            "这些应由研究或策略层按信号日点时状态处理。"
        ),
        (
            "若要与旧notebook研究样本严格对照，研究层还应使用"
            "list_days>=730、非ST、非停牌及行业非空的信号日股票池。"
        ),
        (
            "中性化结果依赖传入截面的股票范围；比较不同研究时必须"
            "固定同一目标日期和同一中性化股票池。"
        ),
        "NaN结果不应填0，应由研究或策略层剔除。",
    ],
    "best_practice": {
        "instance_name": "bias_std_turn_5d",
        "parameters": {
            "short_window": 5,
            "long_window": 504,
            "min_short_observations": 2,
            "min_long_observations": 2,
            "neutralize_industry": True,
            "winsor_k": 5.0,
            "min_cs_count": 80,
        },
        "description": (
            "当前最佳实践为5日/504日换手率样本标准差比值偏离，"
            "并使用市值行业中性化后的残差。"
        ),
    },
    "pit_notes": [
        (
            "滚动标准差只使用目标日及以前的turn，不读取未来数据。"
        ),
        (
            "行业和总市值必须是目标日点时可得数据；"
            "不得使用当前行业分类回填历史。"
        ),
        (
            "目标日收盘换手率参与计算，因此信号在目标日收盘后形成，"
            "最早于下一可交易时点执行。"
        ),
    ],
    "references": [
        (
            "LEO-LISHEN/Bigquant_research：研究稿/华泰因子复现/"
            "华泰换手率类因子/bias_std_turn_5d因子/"
            "bias_std_turn_5d.ipynb。"
        ),
        (
            "BigQuant m_stddev为滚动样本标准差；迁移版本使用"
            "pandas rolling.std(ddof=1)保持同一统计口径。"
        ),
    ],
    "tags": [
        "turnover",
        "volatility_bias",
        "size_neutralized",
        "industry_neutralized",
        "parameterized_window",
    ],
    "status": "research",
    "version": "1.1.0",
}
