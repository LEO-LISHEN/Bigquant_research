# -*- coding: utf-8 -*-
"""按因子名称列表自动计算相关矩阵并可选绘制热力图。"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from factor_lib.common.data_adapters.bigquant_adapters.loader import (
    get_factor_data_requirements,
    get_factor_metadata,
    load_factor_raw_data,
)
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


def _normalize_factor_names(factor_names):
    if isinstance(factor_names, str):
        raise TypeError(
            "factor_names 必须是至少包含两个因子名称的列表，"
            "不能是单个字符串。"
        )
    if not isinstance(factor_names, Sequence):
        raise TypeError("factor_names 必须是因子名称序列。")

    normalized = []
    for name in factor_names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"无效因子名称：{name!r}")
        name = name.strip()
        if name in normalized:
            raise ValueError(f"factor_names 中存在重复因子：{name}")
        normalized.append(name)

    if len(normalized) < 2:
        raise ValueError("至少需要两个不同因子。")
    return normalized


def _normalize_factor_params_by_name(
    factor_names,
    factor_params_by_name,
):
    if factor_params_by_name is None:
        return {name: {} for name in factor_names}
    if not isinstance(factor_params_by_name, Mapping):
        raise TypeError(
            "factor_params_by_name 必须是字典或 None。"
        )

    unknown_factors = sorted(
        set(factor_params_by_name) - set(factor_names)
    )
    if unknown_factors:
        raise ValueError(
            "factor_params_by_name 包含未请求的因子："
            f"{unknown_factors}"
        )

    result = {}
    for factor_name in factor_names:
        params = factor_params_by_name.get(factor_name, {})
        if not isinstance(params, Mapping):
            raise TypeError(
                f"因子 {factor_name!r} 的参数必须是字典。"
            )
        params = dict(params)
        conflicts = sorted(
            _RESERVED_FACTOR_PARAMS.intersection(params)
        )
        if conflicts:
            raise ValueError(
                f"因子 {factor_name!r} 的参数包含保留项："
                f"{conflicts}"
            )
        result[factor_name] = params
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
        raise TypeError("universe 必须是 'all_a'、股票代码序列或配置字典。")

    universe_type = str(universe.get("type", "")).strip().lower()
    if universe_type == "all_a":
        return {"type": "all_a"}
    if universe_type in {"custom", "custom_list"}:
        return {
            "type": "custom",
            "instruments": _normalize_instruments(universe.get("instruments")),
        }
    if universe_type == "index":
        return {
            "type": "index",
            "index_codes": _normalize_instruments(
                universe.get("index_codes", universe.get("code"))
            ),
        }
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
    SELECT date, instrument AS index_code, member_code AS instrument
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
        raise ValueError(f"指数成分股查询结果缺少字段：{sorted(missing)}")
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
    missing_dates = target_dates.difference(
        pd.DatetimeIndex(panel["date"].unique())
    )
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
        f"[因子相关性] [{stage_number}/{stage_total}] "
        f"{message} | 已耗时 {elapsed:.1f}s",
        end="",
        flush=True,
    )


def _query_trading_calendar(end_date):
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
    return dates.normalize().unique().sort_values()


def _resolve_history_days(requirements):
    data_window = requirements.get("data_window", {})
    if not isinstance(data_window, Mapping):
        raise ValueError("FACTOR['data_window'] 必须是字典。")

    try:
        lookback = int(
            data_window.get("lookback_trading_days", 0)
        )
        minimum_history = int(
            data_window.get(
                "minimum_history_observations",
                0,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "data_window 的历史窗口必须是非负整数。"
        ) from exc

    if lookback < 0 or minimum_history < 0:
        raise ValueError("因子历史窗口不能为负数。")
    return max(lookback, minimum_history)


def _build_target_dates(
    trading_calendar,
    start_date,
    end_date,
    frequency,
):
    candidates = trading_calendar[
        (trading_calendar >= start_date)
        & (trading_calendar <= end_date)
    ]
    if candidates.empty:
        raise ValueError("指定区间内没有交易日。")
    return candidates[::frequency]


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
                f"{factor_columns}，无法自动确定相关性字段。"
            )
    return factor_name


def _load_one_factor(
    factor_name,
    target_dates,
    trading_calendar,
    factor_params,
    instruments,
    progress_every,
):
    metadata = get_factor_metadata(factor_name)
    requirements = get_factor_data_requirements(
        factor_name,
        factor_params,
    )
    history_days = _resolve_history_days(requirements)
    factor_dates = _build_factor_dates(
        target_dates,
        trading_calendar,
        history_days,
    )

    resolved_factor_params = {
        name: value
        for name, value in requirements[
            "resolved_factor_params"
        ].items()
        if name not in _RESERVED_FACTOR_PARAMS
    }

    raw_data = load_factor_raw_data(
        factor_name=factor_name,
        dates=factor_dates,
        factor_params=resolved_factor_params,
        instruments=instruments,
        show_progress=False,
    )
    factor_data = get_factor(
        factor_name,
        raw_data,
        target_dates=target_dates,
        as_of_date=target_dates.max(),
        **resolved_factor_params,
        show_progress=False,
        progress_every=progress_every,
    )

    factor_column = _resolve_factor_column(
        metadata,
        factor_name,
    )
    required = {"date", "instrument", factor_column}
    missing = required - set(factor_data.columns)
    if missing:
        raise ValueError(
            f"因子 {factor_name!r} 输出缺少字段：{sorted(missing)}"
        )

    panel = factor_data[
        ["date", "instrument", factor_column]
    ].rename(columns={factor_column: factor_name})
    panel["date"] = pd.to_datetime(
        panel["date"],
        errors="coerce",
    ).dt.normalize()
    panel[factor_name] = pd.to_numeric(
        panel[factor_name],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)

    if panel["date"].isna().any():
        raise ValueError(f"因子 {factor_name!r} 输出包含无效日期。")
    if panel["instrument"].isna().any():
        raise ValueError(
            f"因子 {factor_name!r} 输出包含空 instrument。"
        )
    if panel.duplicated(["date", "instrument"]).any():
        raise ValueError(
            f"因子 {factor_name!r} 输出存在重复 date + instrument。"
        )
    return panel


def _merge_factor_panels(factor_panels):
    merged = factor_panels[0]
    for panel in factor_panels[1:]:
        merged = merged.merge(
            panel,
            on=["date", "instrument"],
            how="outer",
            validate="one_to_one",
        )
    return merged.sort_values(
        ["date", "instrument"],
        kind="mergesort",
    ).reset_index(drop=True)


def _calculate_correlation_from_panel(
    factor_data,
    factor_names,
    method,
    min_obs,
    show_progress,
    progress_every,
    started_at,
    stage_number,
    stage_total,
):
    correlation_sum = pd.DataFrame(
        0.0,
        index=factor_names,
        columns=factor_names,
    )
    overlap_days = pd.DataFrame(
        0,
        index=factor_names,
        columns=factor_names,
        dtype=int,
    )

    grouped = list(factor_data.groupby("date", sort=True))
    total_dates = len(grouped)

    for position, (date, cross_section) in enumerate(
        grouped,
        start=1,
    ):
        correlation = cross_section[factor_names].corr(
            method=method,
            min_periods=min_obs,
        )
        valid = correlation.notna()
        correlation_sum = correlation_sum.add(
            correlation.where(valid, 0.0),
            fill_value=0.0,
        )
        overlap_days = overlap_days.add(
            valid.astype(int),
            fill_value=0,
        )

        should_refresh = (
            position == 1
            or position % progress_every == 0
            or position == total_dates
        )
        if show_progress and should_refresh:
            _render_progress(
                stage_number,
                stage_total,
                (
                    f"计算截面 {position}/{total_dates}"
                    f"（{position / total_dates:.1%}），"
                    f"当前 {date:%Y-%m-%d}"
                ),
                started_at,
            )

    correlation_matrix = correlation_sum.divide(
        overlap_days.replace(0, np.nan)
    )
    for factor_name in factor_names:
        if overlap_days.loc[factor_name, factor_name] > 0:
            correlation_matrix.loc[
                factor_name,
                factor_name,
            ] = 1.0

    return correlation_matrix, overlap_days


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


def _plot_factor_correlation_heatmap(
    correlation_matrix,
    title,
    figsize=(10, 8),
    annotate=True,
    show=True,
):
    """内置绘图实现；兼容脚本可转发到本函数。"""
    import matplotlib.pyplot as plt

    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("correlation_matrix 必须是方阵。")
    if list(correlation_matrix.index) != list(
        correlation_matrix.columns
    ):
        raise ValueError("相关矩阵的行名和列名必须一致。")

    _configure_chinese_font(plt)
    labels = list(correlation_matrix.columns)
    values = correlation_matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("平均截面相关系数")

    if annotate and len(labels) <= 20:
        for row in range(len(labels)):
            for col in range(len(labels)):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if abs(value) > 0.5
                            else "black"
                        ),
                        fontsize=9,
                    )

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def calculate_factor_correlation(
    start_date,
    end_date,
    frequency,
    factor_names,
    factor_params_by_name=None,
    instruments=None,
    universe=None,
    method="spearman",
    min_obs=30,
    plot=True,
    plot_title=None,
    figsize=(10, 8),
    annotate=True,
    show_progress=False,
    progress_every=20,
):
    """自动计算多个因子的平均截面相关矩阵，并可选绘制热力图。

    参数
    ----
    start_date, end_date : str 或 datetime
        因子相关性研究区间。
    frequency : int
        每隔多少个交易日取一个因子截面。
    factor_names : sequence[str]
        至少两个因子中心登记名称。
    factor_params_by_name : dict 或 None
        以因子名称为键、因子内部参数字典为值。
    instruments : sequence[str]、str 或 None
        固定股票范围；None 表示全部A股。
    universe : "all_a"、股票代码集合或 dict，可选
        点时股票池定义。指数股票池例如
        ``{"type": "index", "index_codes": ["000300.SH"]}``；每个目标日
        仅以当日历史成分股计算截面相关性。传入时不得同时传 instruments。
    method : {"pearson", "spearman"}
        单日截面相关系数方法。
    min_obs : int
        每对因子单日计算相关系数所需的最少共同有效样本数。
    plot : bool
        是否绘制并展示相关系数热力图。

    返回
    ----
    dict
        包含 correlation_matrix、overlap_days、factor_data、
        target_dates、universe_config、universe_panel、figure 和 axis。
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
    factor_names = _normalize_factor_names(factor_names)
    factor_params_by_name = (
        _normalize_factor_params_by_name(
            factor_names,
            factor_params_by_name,
        )
    )
    if universe is not None and instruments is not None:
        raise ValueError("universe 与 instruments 互斥；请只指定一种股票池。")
    instruments = _normalize_instruments(instruments)

    if method not in {"pearson", "spearman"}:
        raise ValueError(
            "method 仅支持 pearson 或 spearman。"
        )
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot 必须是 bool。")
    if not isinstance(annotate, (bool, np.bool_)):
        raise TypeError("annotate 必须是 bool。")

    try:
        if show_progress:
            _render_progress(
                1,
                6,
                "读取交易日历并生成目标截面",
                started_at,
            )

        trading_calendar = _query_trading_calendar(end_date)
        target_dates = _build_target_dates(
            trading_calendar,
            start_date,
            end_date,
            frequency,
        )
        if show_progress:
            _render_progress(
                2,
                6,
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

        factor_panels = []
        total_factors = len(factor_names)
        for position, factor_name in enumerate(
            factor_names,
            start=1,
        ):
            if show_progress:
                _render_progress(
                    3,
                    6,
                    (
                        f"适配器读取并计算因子 "
                        f"{position}/{total_factors}：{factor_name}"
                    ),
                    started_at,
                )

            factor_panels.append(
                _load_one_factor(
                    factor_name=factor_name,
                    target_dates=target_dates,
                    trading_calendar=trading_calendar,
                    factor_params=factor_params_by_name[
                        factor_name
                    ],
                    instruments=load_instruments,
                    progress_every=progress_every,
                )
            )

        if show_progress:
            _render_progress(
                4,
                6,
                "按date + instrument对齐全部因子",
                started_at,
            )

        factor_data = _merge_factor_panels(factor_panels)
        factor_data = _filter_panel_to_universe(
            factor_data,
            universe_panel,
            "factor_data",
        )
        correlation_matrix, overlap_days = (
            _calculate_correlation_from_panel(
                factor_data=factor_data,
                factor_names=factor_names,
                method=method,
                min_obs=min_obs,
                show_progress=show_progress,
                progress_every=progress_every,
                started_at=started_at,
                stage_number=5,
                stage_total=6,
            )
        )

        figure = None
        axis = None
        if plot:
            if show_progress:
                _render_progress(
                    6,
                    6,
                    "绘制并展示因子相关系数热力图",
                    started_at,
                )
            title = (
                plot_title
                if plot_title is not None
                else f"因子平均截面{method}相关系数"
            )
            figure, axis = _plot_factor_correlation_heatmap(
                correlation_matrix=correlation_matrix,
                title=title,
                figsize=figsize,
                annotate=annotate,
                show=True,
            )
        elif show_progress:
            _render_progress(
                6,
                6,
                "计算完成，已按参数跳过绘图",
                started_at,
            )

        return {
            "correlation_matrix": correlation_matrix,
            "overlap_days": overlap_days,
            "factor_data": factor_data,
            "target_dates": target_dates,
            "universe_config": universe_config,
            "universe_panel": universe_panel,
            "figure": figure,
            "axis": axis,
        }
    finally:
        if show_progress:
            print()
