# -*- coding: utf-8 -*-
"""按因子名称自动完成基础指标计算和可选 IC/RankIC 绘图。"""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np
import pandas as pd

from factor_lib.common.data_adapters.bigquant_adapters.daily import (
    load_daily_raw_data,
)
from factor_lib.common.data_adapters.bigquant_adapters.loader import (
    get_factor_data_requirements,
    get_factor_metadata,
    load_factor_raw_data,
)
from factor_lib.common.preprocess.zscore import zscore
from factor_lib.factor_hub.get_factor import get_factor


_RESERVED_FACTOR_PARAMS = {
    "data",
    "target_dates",
    "as_of_date",
    "show_progress",
    "progress_every",
}


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
    conflicts = sorted(
        _RESERVED_FACTOR_PARAMS.intersection(result)
    )
    if conflicts:
        raise ValueError(
            "factor_params 包含由评价函数统一控制的保留参数："
            f"{conflicts}。"
        )
    return result


def _normalize_instruments(instruments):
    if instruments is None:
        return None
    if isinstance(instruments, str):
        instruments = [instruments]

    try:
        values = list(instruments)
    except TypeError as exc:
        raise TypeError(
            "instruments 必须是股票代码、代码序列或 None。"
        ) from exc

    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"无效股票代码：{value!r}")
        value = value.strip()
        if value not in normalized:
            normalized.append(value)

    if not normalized:
        raise ValueError(
            "instruments 不能为空；全部A股请使用 None。"
        )
    return normalized


def _normalize_universe(universe):
    """标准化本评价函数支持的股票池配置。"""
    if universe == "all_a":
        return {"type": "all_a"}
    if isinstance(universe, (list, tuple, set, frozenset)):
        return {
            "type": "custom",
            "instruments": _normalize_instruments(universe),
        }
    if not isinstance(universe, Mapping):
        raise TypeError(
            "universe 必须是 'all_a'、股票代码序列或配置字典。"
        )

    universe_type = str(universe.get("type", "")).strip().lower()
    if universe_type == "all_a":
        return {"type": "all_a"}
    if universe_type in {"custom", "custom_list"}:
        instruments = _normalize_instruments(universe.get("instruments"))
        return {"type": "custom", "instruments": instruments}
    if universe_type == "index":
        index_codes = _normalize_instruments(
            universe.get("index_codes", universe.get("code"))
        )
        return {"type": "index", "index_codes": index_codes}
    raise ValueError("universe['type'] 仅支持 all_a、index、custom。")


def _quote_sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _query_index_universe(index_codes, target_dates):
    """读取每个目标日的历史指数成分股。"""
    try:
        import dai
    except ImportError as exc:
        raise ImportError("未能导入 dai；请在 BigQuant 环境运行。") from exc

    target_dates = (
        pd.DatetimeIndex(pd.to_datetime(target_dates))
        .normalize()
        .unique()
        .sort_values()
    )
    if target_dates.empty:
        raise ValueError("target_dates 不能为空。")

    index_sql = ", ".join(_quote_sql_literal(code) for code in index_codes)
    date_sql = ", ".join(
        _quote_sql_literal(date.strftime("%Y-%m-%d"))
        for date in target_dates
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
    panel = dai.query(
        sql,
        filters={
            "date": [
                target_dates.min().strftime("%Y-%m-%d"),
                target_dates.max().strftime("%Y-%m-%d"),
            ]
        },
    ).df()
    required = {"date", "index_code", "instrument"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            f"指数成分股查询结果缺少字段：{sorted(missing)}"
        )
    if panel.empty:
        raise ValueError("未读取到目标日的指数历史成分股。")

    panel = panel.loc[:, ["date", "instrument"]].copy()
    panel["date"] = pd.to_datetime(
        panel["date"], errors="coerce"
    ).dt.normalize()
    panel["instrument"] = panel["instrument"].astype("string").str.strip()
    if panel["date"].isna().any() or panel["instrument"].isna().any():
        raise ValueError("指数成分股数据包含无效 date 或 instrument。")
    if (panel["instrument"] == "").any():
        raise ValueError("指数成分股数据包含空 instrument。")

    panel = panel.drop_duplicates().sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
    covered_dates = pd.DatetimeIndex(panel["date"].unique())
    missing_dates = target_dates.difference(covered_dates)
    if not missing_dates.empty:
        preview = [date.strftime("%Y-%m-%d") for date in missing_dates[:5]]
        raise ValueError(
            "部分目标日未读取到指数历史成分股："
            f"{preview}。请检查指数代码或日期覆盖。"
        )
    return panel


def _resolve_universe_panel(universe, target_dates):
    """解析为点时股票池面板及仅用于查询优化的代码并集。"""
    target_dates = (
        pd.DatetimeIndex(pd.to_datetime(target_dates))
        .normalize()
        .unique()
        .sort_values()
    )
    if target_dates.empty:
        raise ValueError("target_dates 不能为空。")

    config = _normalize_universe(universe)
    if config["type"] == "all_a":
        return config, None, None
    if config["type"] == "custom":
        panel = pd.MultiIndex.from_product(
            [target_dates, config["instruments"]],
            names=["date", "instrument"],
        ).to_frame(index=False)
        return config, panel, config["instruments"]
    if config["type"] == "index":
        panel = _query_index_universe(config["index_codes"], target_dates)
        return config, panel, sorted(panel["instrument"].unique().tolist())
    raise RuntimeError(f"未处理的股票池类型：{config['type']}")


def _filter_panel_to_universe(panel, universe_panel, panel_name):
    """按目标日 date + instrument 股票池过滤评价面板。"""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{panel_name} 必须是 pandas.DataFrame。")
    required = {"date", "instrument"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"{panel_name} 缺少字段：{sorted(missing)}")
    if universe_panel is None:
        return panel.copy()

    result = panel.copy()
    result["date"] = pd.to_datetime(
        result["date"], errors="coerce"
    ).dt.normalize()
    if result["date"].isna().any() or result["instrument"].isna().any():
        raise ValueError(f"{panel_name} 含有无效 date 或 instrument。")
    if result.duplicated(["date", "instrument"]).any():
        raise ValueError(f"{panel_name} 存在重复 date + instrument。")
    return result.merge(
        universe_panel,
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )


def _render_progress(
    stage_number,
    stage_total,
    message,
    started_at,
):
    elapsed = time.perf_counter() - started_at
    print(
        "\r"
        f"[因子基础指标] [{stage_number}/{stage_total}] "
        f"{message} | 已耗时 {elapsed:.1f}s",
        end="",
        flush=True,
    )


def _query_trading_calendar(end_date):
    """读取截至截止日的完整A股交易日历，供预热和标签定位使用。"""
    try:
        import dai
    except ImportError as exc:
        raise ImportError(
            "未能导入 dai；请在 BigQuant 环境运行。"
        ) from exc

    sql = f"""
    SELECT DISTINCT date
    FROM cn_stock_bar1d
    WHERE date <= '{end_date:%Y-%m-%d}'
    ORDER BY date
    """
    calendar = dai.query(sql).df()
    if calendar.empty or "date" not in calendar.columns:
        raise ValueError("未读取到有效的A股交易日历。")

    dates = pd.DatetimeIndex(
        pd.to_datetime(calendar["date"], errors="coerce")
    )
    if dates.isna().any():
        raise ValueError("交易日历中存在无效日期。")

    dates = dates.normalize().unique().sort_values()
    if dates.empty:
        raise ValueError("A股交易日历为空。")
    return dates


def _resolve_history_days(requirements):
    data_window = requirements.get("data_window", {})
    if not isinstance(data_window, Mapping):
        raise ValueError("FACTOR['data_window'] 必须是字典。")

    lookback = data_window.get("lookback_trading_days", 0)
    minimum_history = data_window.get(
        "minimum_history_observations",
        0,
    )
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


def _build_evaluation_schedule(
    trading_calendar,
    start_date,
    end_date,
    frequency,
):
    """每N个交易日取一次截面，并以未来N日收盘收益作为标签。"""
    candidates = trading_calendar[
        (trading_calendar >= start_date)
        & (trading_calendar <= end_date)
    ]
    if candidates.empty:
        raise ValueError("指定区间内没有交易日。")

    sampled_dates = candidates[::frequency]
    calendar_positions = {
        date: position
        for position, date in enumerate(trading_calendar)
    }

    records = []
    for signal_date in sampled_dates:
        start_position = calendar_positions[signal_date]
        end_position = start_position + frequency
        if end_position >= len(trading_calendar):
            continue

        return_end_date = trading_calendar[end_position]
        # end_date 是本次研究的信息截止日，最终标签必须在此前完整结束。
        if return_end_date > end_date:
            continue

        records.append(
            {
                "date": signal_date,
                "return_end_date": return_end_date,
            }
        )

    schedule = pd.DataFrame(
        records,
        columns=["date", "return_end_date"],
    )
    if schedule.empty:
        raise ValueError(
            "没有完整结束的未来收益标签；请扩大日期区间或降低 frequency。"
        )
    return schedule


def _build_factor_dates(
    target_dates,
    trading_calendar,
    history_days,
):
    calendar_positions = {
        date: position
        for position, date in enumerate(trading_calendar)
    }
    required_dates = set()

    for target_date in target_dates:
        position = calendar_positions.get(target_date)
        if position is None:
            raise ValueError(
                f"目标日 {target_date:%Y-%m-%d} 不在交易日历中。"
            )

        start_position = position - history_days
        if start_position < 0:
            raise ValueError(
                f"目标日 {target_date:%Y-%m-%d} 前历史数据不足 "
                f"{history_days} 个交易日。"
            )

        required_dates.update(
            trading_calendar[
                start_position : position + 1
            ].tolist()
        )

    return pd.DatetimeIndex(sorted(required_dates))


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
                f"因子 {factor_name!r} 声明多个数值输出："
                f"{factor_columns}，无法自动确定评价列。"
            )
    return factor_name


def _prepare_forward_return_labels(
    schedule,
    instruments,
):
    price_dates = sorted(
        set(schedule["date"])
        | set(schedule["return_end_date"])
    )

    price_data = load_daily_raw_data(
        standard_fields=["close"],
        dates=price_dates,
        instruments=instruments,
        show_progress=False,
    )
    required = {"date", "instrument", "close"}
    missing = required - set(price_data.columns)
    if missing:
        raise ValueError(
            f"价格适配器输出缺少字段：{sorted(missing)}"
        )

    price_data = price_data.loc[
        :,
        ["date", "instrument", "close"],
    ].copy()
    price_data["date"] = pd.to_datetime(
        price_data["date"],
        errors="coerce",
    ).dt.normalize()
    price_data["close"] = pd.to_numeric(
        price_data["close"],
        errors="coerce",
    )
    price_data = price_data.loc[
        price_data["date"].notna()
        & price_data["instrument"].notna()
        & price_data["close"].gt(0)
    ].copy()

    duplicated = price_data.duplicated(
        ["date", "instrument"],
        keep=False,
    )
    if duplicated.any():
        raise ValueError(
            "标签价格数据存在重复的 date + instrument。"
        )

    start_prices = price_data.rename(
        columns={"close": "start_close"}
    )
    end_prices = price_data.rename(
        columns={
            "date": "return_end_date",
            "close": "end_close",
        }
    )

    label_data = (
        schedule
        .merge(
            start_prices,
            on="date",
            how="inner",
            validate="one_to_many",
        )
        .merge(
            end_prices,
            on=["return_end_date", "instrument"],
            how="inner",
            validate="one_to_one",
        )
    )
    label_data["forward_return"] = (
        label_data["end_close"]
        / label_data["start_close"]
        - 1.0
    )
    label_data = label_data[
        [
            "date",
            "instrument",
            "forward_return",
            "return_end_date",
        ]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    if label_data.empty:
        raise ValueError("未来收益标签为空。")
    return label_data


def _calculate_metrics_from_panels(
    factor_data,
    label_data,
    factor_column,
    min_obs,
    show_progress,
    progress_every,
    started_at,
    stage_number,
    stage_total,
):
    factor_required = {"date", "instrument", factor_column}
    label_required = {
        "date",
        "instrument",
        "forward_return",
        "return_end_date",
    }
    missing_factor = factor_required - set(factor_data.columns)
    missing_label = label_required - set(label_data.columns)
    if missing_factor:
        raise ValueError(
            f"因子计算结果缺少字段：{sorted(missing_factor)}"
        )
    if missing_label:
        raise ValueError(
            f"收益标签缺少字段：{sorted(missing_label)}"
        )

    factor_panel = factor_data[
        ["date", "instrument", factor_column]
    ].copy()
    label_panel = label_data[
        [
            "date",
            "instrument",
            "forward_return",
            "return_end_date",
        ]
    ].copy()

    if factor_panel.duplicated(["date", "instrument"]).any():
        raise ValueError(
            "因子计算结果存在重复的 date + instrument。"
        )
    if label_panel.duplicated(["date", "instrument"]).any():
        raise ValueError(
            "收益标签存在重复的 date + instrument。"
        )

    factor_panel["date"] = pd.to_datetime(factor_panel["date"])
    label_panel["date"] = pd.to_datetime(label_panel["date"])
    label_panel["return_end_date"] = pd.to_datetime(
        label_panel["return_end_date"]
    )

    panel = factor_panel.merge(
        label_panel,
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    ).replace([np.inf, -np.inf], np.nan)

    records = []
    grouped = list(panel.groupby("date", sort=True))
    total_periods = len(grouped)

    for position, (signal_date, cross_section) in enumerate(
        grouped,
        start=1,
    ):
        valid = cross_section[
            [
                factor_column,
                "forward_return",
                "return_end_date",
            ]
        ].dropna()

        sample_count = len(valid)
        return_end_date = (
            valid["return_end_date"].max()
            if sample_count
            else pd.NaT
        )
        ic = np.nan
        rank_ic = np.nan
        factor_return = np.nan
        factor_t_value = np.nan

        if sample_count >= min_obs:
            ic = valid[factor_column].corr(
                valid["forward_return"],
                method="pearson",
            )
            rank_ic = valid[factor_column].corr(
                valid["forward_return"],
                method="spearman",
            )

            regression_data = pd.DataFrame(
                {
                    "factor": zscore(valid[factor_column]),
                    "return": valid["forward_return"],
                }
            ).dropna()

            if len(regression_data) >= max(min_obs, 3):
                x = np.column_stack(
                    [
                        np.ones(len(regression_data)),
                        regression_data["factor"].to_numpy(
                            dtype=float
                        ),
                    ]
                )
                y = regression_data["return"].to_numpy(
                    dtype=float
                )

                if np.linalg.matrix_rank(x) == 2:
                    beta = np.linalg.lstsq(
                        x,
                        y,
                        rcond=None,
                    )[0]
                    residual = y - x @ beta
                    degrees_of_freedom = len(y) - 2

                    if degrees_of_freedom > 0:
                        residual_variance = (
                            np.sum(residual ** 2)
                            / degrees_of_freedom
                        )
                        covariance = (
                            residual_variance
                            * np.linalg.inv(x.T @ x)
                        )
                        standard_error = np.sqrt(
                            covariance[1, 1]
                        )
                        factor_return = beta[1]

                        if (
                            standard_error > 0
                            and np.isfinite(standard_error)
                        ):
                            factor_t_value = (
                                beta[1] / standard_error
                            )

        records.append(
            {
                "signal_date": signal_date,
                "return_end_date": return_end_date,
                "sample_count": sample_count,
                "ic": ic,
                "rank_ic": rank_ic,
                "factor_return": factor_return,
                "factor_t_value": factor_t_value,
            }
        )

        should_refresh = (
            position == 1
            or position % progress_every == 0
            or position == total_periods
        )
        if show_progress and should_refresh:
            _render_progress(
                stage_number,
                stage_total,
                (
                    f"计算截面 {position}/{total_periods}"
                    f"（{position / total_periods:.1%}），"
                    f"当前 {signal_date:%Y-%m-%d}"
                ),
                started_at,
            )

    timeseries = pd.DataFrame(
        records,
        columns=[
            "signal_date",
            "return_end_date",
            "sample_count",
            "ic",
            "rank_ic",
            "factor_return",
            "factor_t_value",
        ],
    )

    ic_std = timeseries["ic"].std(ddof=1)
    rank_ic_std = timeseries["rank_ic"].std(ddof=1)
    summary = pd.DataFrame(
        [
            {
                "factor_column": factor_column,
                "ic_mean": timeseries["ic"].mean(),
                "icir": (
                    timeseries["ic"].mean() / ic_std
                    if pd.notna(ic_std) and ic_std != 0
                    else np.nan
                ),
                "rank_ic_mean": timeseries["rank_ic"].mean(),
                "rank_icir": (
                    timeseries["rank_ic"].mean() / rank_ic_std
                    if pd.notna(rank_ic_std)
                    and rank_ic_std != 0
                    else np.nan
                ),
                "factor_return_mean": timeseries[
                    "factor_return"
                ].mean(),
                "mean_t_value": timeseries[
                    "factor_t_value"
                ].mean(),
                "cross_section_count": len(timeseries),
                "valid_ic_count": timeseries[
                    "ic"
                ].notna().sum(),
            }
        ]
    )
    return summary, timeseries


def _configure_chinese_font(plt):
    from matplotlib import font_manager

    available_fonts = {
        font.name for font in font_manager.fontManager.ttflist
    }
    for font_name in [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
    ]:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _plot_ic_rankic_series(
    metrics_timeseries,
    title,
    figsize=(14, 5),
    show=True,
):
    """内置绘图实现；兼容脚本可转发到本函数。"""
    import matplotlib.pyplot as plt

    required = {"signal_date", "ic", "rank_ic"}
    missing = required - set(metrics_timeseries.columns)
    if missing:
        raise ValueError(
            f"指标时序表缺少字段：{sorted(missing)}"
        )

    _configure_chinese_font(plt)
    data = metrics_timeseries.sort_values(
        "signal_date"
    ).copy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        data["signal_date"],
        data["ic"],
        label="IC",
        linewidth=1.5,
    )
    ax.plot(
        data["signal_date"],
        data["rank_ic"],
        label="RankIC",
        linewidth=1.5,
    )
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.set_title(title)
    ax.set_xlabel("信号日")
    ax.set_ylabel("相关系数")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def calculate_factor_basic_metrics(
    start_date,
    end_date,
    frequency,
    factor_name,
    factor_params=None,
    instruments=None,
    universe=None,
    min_obs=30,
    plot=True,
    plot_title=None,
    figsize=(14, 5),
    show_progress=False,
    progress_every=20,
):
    """自动计算指定因子的标准基础指标，并可选绘制IC/RankIC时序图。

    参数
    ----
    start_date, end_date : str 或 datetime
        研究区间。end_date 同时是未来收益标签的信息截止日，未在该日
        前完整结束的最终标签会被剔除。
    frequency : int
        每隔多少个交易日取一个因子截面，同时也是未来收益标签的
        交易日跨度。例如20表示每20个交易日评价一次，并计算未来20日
        收盘到收盘收益。
    factor_name : str
        因子中心登记的 FACTOR 名称。
    factor_params : dict 或 None
        因子内部参数。目标日期、截止日和进度参数由本函数统一控制。
    instruments : sequence[str]、str 或 None
        固定股票范围；None 表示不限定代码，由适配器返回全部A股。
    universe : "all_a"、股票代码集合或 dict，可选
        点时股票池定义。``{"type": "index", "index_codes": ["000300.SH"]}``
        会在每个评价信号日使用该指数的历史成分股。传入 universe 时不得再传
        instruments。固定股票池仍可通过 instruments 保持原调用方式。
    min_obs : int
        单个截面计算指标所需的最少完整股票数。
    plot : bool
        是否绘制并展示 IC/RankIC 时序图。

    返回
    ----
    dict
        包含 summary、timeseries、factor_data、label_data、
        evaluation_schedule、universe_config、universe_panel、figure 和 axis。

    说明
    ----
    未来收益采用信号日收盘价到未来N个交易日收盘价，仅作为因子研究
    标签，不等同于可成交的组合回测收益。
    """
    started_at = time.perf_counter()
    start_date = _normalize_timestamp(
        start_date,
        "start_date",
    )
    end_date = _normalize_timestamp(
        end_date,
        "end_date",
    )
    if start_date > end_date:
        raise ValueError("start_date 不能晚于 end_date。")

    frequency = _normalize_positive_integer(
        frequency,
        "frequency",
    )
    min_obs = _normalize_positive_integer(
        min_obs,
        "min_obs",
    )
    progress_every = _normalize_positive_integer(
        progress_every,
        "progress_every",
    )
    if not isinstance(factor_name, str) or not factor_name.strip():
        raise ValueError("factor_name 必须是非空字符串。")
    factor_name = factor_name.strip()
    factor_params = _normalize_factor_params(factor_params)
    if universe is not None and instruments is not None:
        raise ValueError("universe 与 instruments 互斥；请只指定一种股票池。")
    instruments = _normalize_instruments(instruments)

    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot 必须是 bool。")

    try:
        if show_progress:
            _render_progress(
                1,
                7,
                "解析因子元数据和动态预热窗口",
                started_at,
            )

        metadata = get_factor_metadata(factor_name)
        requirements = get_factor_data_requirements(
            factor_name,
            factor_params,
        )
        factor_column = _resolve_factor_column(
            metadata,
            factor_name,
        )
        history_days = _resolve_history_days(requirements)

        # 使用loader解析后的同一份因子参数，保证预热和计算口径一致。
        resolved_factor_params = {
            name: value
            for name, value in requirements[
                "resolved_factor_params"
            ].items()
            if name not in _RESERVED_FACTOR_PARAMS
        }

        if show_progress:
            _render_progress(
                2,
                7,
                "读取交易日历并生成评价截面",
                started_at,
            )

        trading_calendar = _query_trading_calendar(end_date)
        schedule = _build_evaluation_schedule(
            trading_calendar,
            start_date,
            end_date,
            frequency,
        )
        target_dates = pd.DatetimeIndex(schedule["date"])
        if show_progress:
            _render_progress(
                3,
                7,
                "读取并校验点时动态股票池" if universe is not None else "确认固定股票池范围",
                started_at,
            )
        if universe is None:
            universe_config = (
                {"type": "all_a"}
                if instruments is None
                else {"type": "custom", "instruments": instruments}
            )
            universe_panel = None
            load_instruments = instruments
        else:
            universe_config, universe_panel, load_instruments = _resolve_universe_panel(
                universe,
                target_dates,
            )
        factor_dates = _build_factor_dates(
            target_dates,
            trading_calendar,
            history_days,
        )

        if show_progress:
            _render_progress(
                4,
                7,
                (
                    f"适配器读取因子数据："
                    f"{len(factor_dates)} 个日期"
                ),
                started_at,
            )

        factor_raw_data = load_factor_raw_data(
            factor_name=factor_name,
            dates=factor_dates,
            factor_params=resolved_factor_params,
            instruments=load_instruments,
            show_progress=False,
        )
        factor_data = get_factor(
            factor_name,
            factor_raw_data,
            target_dates=target_dates,
            as_of_date=end_date,
            **resolved_factor_params,
            show_progress=False,
            progress_every=progress_every,
        )

        if show_progress:
            _render_progress(
                5,
                7,
                "适配器读取价格并构造完整未来收益标签",
                started_at,
            )

        label_data = _prepare_forward_return_labels(
            schedule,
            load_instruments,
        )
        factor_data = _filter_panel_to_universe(
            factor_data,
            universe_panel,
            "factor_data",
        )
        label_data = _filter_panel_to_universe(
            label_data,
            universe_panel,
            "label_data",
        )
        summary, timeseries = _calculate_metrics_from_panels(
            factor_data=factor_data,
            label_data=label_data,
            factor_column=factor_column,
            min_obs=min_obs,
            show_progress=show_progress,
            progress_every=progress_every,
            started_at=started_at,
            stage_number=6,
            stage_total=7,
        )

        figure = None
        axis = None
        if plot:
            if show_progress:
                _render_progress(
                    7,
                    7,
                    "绘制并展示IC/RankIC时序图",
                    started_at,
                )
            title = (
                plot_title
                if plot_title is not None
                else (
                    f"{factor_name}未来{frequency}个交易日"
                    "IC与RankIC时序"
                )
            )
            figure, axis = _plot_ic_rankic_series(
                metrics_timeseries=timeseries,
                title=title,
                figsize=figsize,
                show=True,
            )
        elif show_progress:
            _render_progress(
                7,
                7,
                "计算完成，已按参数跳过绘图",
                started_at,
            )

        return {
            "summary": summary,
            "timeseries": timeseries,
            "factor_data": factor_data,
            "label_data": label_data,
            "evaluation_schedule": schedule,
            "universe_config": universe_config,
            "universe_panel": universe_panel,
            "figure": figure,
            "axis": axis,
        }
    finally:
        if show_progress:
            print()
