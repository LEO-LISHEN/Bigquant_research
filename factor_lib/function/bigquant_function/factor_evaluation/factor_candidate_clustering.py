# -*- coding: utf-8 -*-
"""候选因子分组与代表因子筛选。

本模块是 BigQuant 的基础因子研究入口，不属于因子层：它只管理
``FACTOR['factor_type'] == 'base'`` 的候选实例，通过数据适配器准备数据、
构造完成的未来收益标签，并按因子暴露相似度分组。模型因子、线性合成因子
和其他依赖因子输出的复合因子不属于本模块的候选池。

候选字典示例
------------
candidate_spec = {
    "book_to_price": [{}],
    "return_nm": [
        {"n_months": 1, "trading_days_per_month": 21},
        {"n_months": 3, "trading_days_per_month": 21},
    ],
    # None 表示展开 FACTOR["candidate_instances"] 中登记的全部常用实例。
    "exp_wgt_return_nm": None,
}

FACTOR["candidate_instances"] 约定
---------------------------------
可为列表或字典。列表中每项形如：
    {"id": "n_6m", "params": {"n_months": 6}}
字典形式则为：
    {"n_6m": {"n_months": 6}}
其中 id 只用于显示和稳定标识；params 才是实际传入因子函数的参数。
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import date, datetime
from collections.abc import Mapping, Sequence
from pathlib import Path

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
from factor_lib.factor_hub.get_factor import get_factor


_RESERVED_FACTOR_PARAMS = {
    "data",
    "target_dates",
    "as_of_date",
    "show_progress",
    "progress_every",
}


def _normalize_date(value, parameter_name):
    try:
        date = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter_name} 必须是可解析日期：{value!r}") from exc
    if pd.isna(date):
        raise ValueError(f"{parameter_name} 不能为空。")
    return date


def _positive_integer(value, parameter_name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{parameter_name} 必须为正整数。")
    return int(value)


def _normalize_instruments(instruments):
    if instruments is None:
        return None
    if isinstance(instruments, str):
        instruments = [instruments]
    try:
        values = list(instruments)
    except TypeError as exc:
        raise TypeError("instruments 必须为股票代码、代码序列或 None。") from exc

    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"无效股票代码：{value!r}")
        value = value.strip()
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("instruments 不能为空；全 A 股请传入 None。")
    return result


def _normalize_universe(universe):
    """标准化本候选因子管理函数支持的股票池配置。"""
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
    completed=None,
    total=None,
    current=None,
):
    """以单行刷新展示当前阶段；不伪造平台查询的内部百分比。"""
    elapsed = time.perf_counter() - started_at
    parts = [f"[候选因子分组] [{stage_number}/{stage_total}] {message}"]
    if completed is not None and total:
        ratio = completed / total
        parts.append(f"{completed}/{total} ({ratio:.1%})")
        if 0 < completed < total:
            parts.append(f"预计剩余 {elapsed / completed * (total - completed):.1f}s")
    if current:
        parts.append(f"当前 {current}")
    parts.append(f"已耗时 {elapsed:.1f}s")
    print("\r" + " | ".join(parts).ljust(220), end="", flush=True)


def _query_trading_calendar(end_date):
    try:
        import dai
    except ImportError as exc:
        raise ImportError("未能导入 dai；请在 BigQuant 环境运行。") from exc

    sql = f"""
    SELECT DISTINCT date
    FROM cn_stock_bar1d
    WHERE date <= '{end_date:%Y-%m-%d}'
    ORDER BY date
    """
    calendar = dai.query(sql).df()
    if calendar.empty or "date" not in calendar.columns:
        raise ValueError("未读取到有效的 A 股交易日历。")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar["date"], errors="coerce"))
    if dates.isna().any():
        raise ValueError("交易日历中存在无效日期。")
    dates = dates.normalize().unique().sort_values()
    if dates.empty:
        raise ValueError("A 股交易日历为空。")
    return dates


def _build_schedule(trading_calendar, start_date, end_date, frequency, holding_days):
    """生成信号日和完整标签终止日，二者分别由频率和持有期控制。"""
    available = trading_calendar[
        (trading_calendar >= start_date) & (trading_calendar <= end_date)
    ]
    if available.empty:
        raise ValueError("指定区间内没有交易日。")

    positions = {date: position for position, date in enumerate(trading_calendar)}
    records = []
    for signal_date in available[::frequency]:
        end_position = positions[signal_date] + holding_days
        if end_position >= len(trading_calendar):
            continue
        return_end_date = trading_calendar[end_position]
        if return_end_date <= end_date:
            records.append(
                {"date": signal_date, "return_end_date": return_end_date}
            )
    schedule = pd.DataFrame(records, columns=["date", "return_end_date"])
    if schedule.empty:
        raise ValueError(
            "没有完整结束的未来收益标签；请扩大区间或缩短 holding_period_days。"
        )
    return schedule


def _history_days(requirements):
    data_window = requirements.get("data_window", {})
    if not isinstance(data_window, Mapping):
        raise ValueError("FACTOR['data_window'] 必须为字典。")
    try:
        lookback = int(data_window.get("lookback_trading_days", 0))
        minimum = int(data_window.get("minimum_history_observations", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("data_window 的历史窗口必须为非负整数。") from exc
    if lookback < 0 or minimum < 0:
        raise ValueError("data_window 的历史窗口不能为负数。")
    return max(lookback, minimum)


def _factor_input_dates(target_dates, trading_calendar, history_days):
    positions = {date: position for position, date in enumerate(trading_calendar)}
    required_dates = set()
    for target_date in target_dates:
        position = positions.get(target_date)
        if position is None:
            raise ValueError(f"目标日期 {target_date:%Y-%m-%d} 不在交易日历中。")
        start_position = position - history_days
        if start_position < 0:
            raise ValueError(
                f"目标日 {target_date:%Y-%m-%d} 前历史数据不足 {history_days} 个交易日。"
            )
        required_dates.update(trading_calendar[start_position : position + 1].tolist())
    return pd.DatetimeIndex(sorted(required_dates))


def _factor_value_column(metadata, factor_name):
    output_schema = metadata.get("output_schema", {})
    if isinstance(output_schema, Mapping):
        value_columns = [
            column for column in output_schema if column not in {"date", "instrument"}
        ]
        if factor_name in value_columns:
            return factor_name
        if len(value_columns) == 1:
            return value_columns[0]
        if len(value_columns) > 1:
            raise ValueError(
                f"因子 {factor_name!r} 声明多个数值输出 {value_columns}，"
                "候选筛选无法自动确定应评价哪一列。"
            )
    return factor_name


def _resolved_factor_params(requirements):
    resolved = requirements.get("resolved_factor_params", {})
    if not isinstance(resolved, Mapping):
        raise ValueError("loader 未返回有效的 resolved_factor_params。")
    return {
        name: value
        for name, value in resolved.items()
        if name not in _RESERVED_FACTOR_PARAMS
    }


def _prepare_labels(schedule, instruments, show_progress):
    dates = sorted(set(schedule["date"]) | set(schedule["return_end_date"]))
    price_data = load_daily_raw_data(
        standard_fields=["close"],
        dates=dates,
        instruments=instruments,
        show_progress=show_progress,
    )
    required = {"date", "instrument", "close"}
    missing = required - set(price_data.columns)
    if missing:
        raise ValueError(f"日频适配器缺少标签价格字段：{sorted(missing)}")

    prices = price_data.loc[:, ["date", "instrument", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.loc[
        prices["date"].notna()
        & prices["instrument"].notna()
        & prices["close"].gt(0)
    ].copy()
    if prices.duplicated(["date", "instrument"]).any():
        raise ValueError("标签价格存在重复的 date + instrument。")

    start_prices = prices.rename(columns={"close": "start_close"})
    end_prices = prices.rename(
        columns={"date": "return_end_date", "close": "end_close"}
    )
    labels = (
        schedule.merge(start_prices, on="date", how="inner", validate="one_to_many")
        .merge(
            end_prices,
            on=["return_end_date", "instrument"],
            how="inner",
            validate="one_to_one",
        )
    )
    labels["forward_return"] = labels["end_close"] / labels["start_close"] - 1.0
    labels = labels[
        ["date", "instrument", "forward_return", "return_end_date"]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    if labels.empty:
        raise ValueError("未来收益标签为空。")
    return labels


def _normalize_candidate_params(params, factor_name):
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError(f"候选因子 {factor_name!r} 的参数必须为字典。")
    result = dict(params)
    conflicts = sorted(_RESERVED_FACTOR_PARAMS.intersection(result))
    if conflicts:
        raise ValueError(
            f"候选因子 {factor_name!r} 的参数不能覆盖框架控制参数：{conflicts}。"
        )
    return result


def _metadata_candidate_instances(factor_name):
    metadata = get_factor_metadata(factor_name)
    _require_base_factor(metadata, factor_name)
    instances = metadata.get("candidate_instances")
    if instances is None:
        raise ValueError(
            f"因子 {factor_name!r} 未声明 FACTOR['candidate_instances']。"
            "请显式传入参数列表，或在后续 FACTOR 规范重写时登记常用实例。"
        )

    result = []
    if isinstance(instances, Mapping):
        iterable = instances.items()
        for instance_id, params in iterable:
            result.append(
                {
                    "instance_id": str(instance_id),
                    "params": _normalize_candidate_params(params, factor_name),
                }
            )
    elif isinstance(instances, Sequence) and not isinstance(instances, (str, bytes)):
        for position, item in enumerate(instances, start=1):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"{factor_name!r} 的 candidate_instances 第 {position} 项必须为字典。"
                )
            params = item.get("params", {})
            instance_id = item.get("id", f"instance_{position}")
            result.append(
                {
                    "instance_id": str(instance_id),
                    "params": _normalize_candidate_params(params, factor_name),
                }
            )
    else:
        raise TypeError(
            f"{factor_name!r} 的 candidate_instances 必须为字典或字典列表。"
        )
    if not result:
        raise ValueError(f"{factor_name!r} 的 candidate_instances 不能为空。")
    return result


def _require_base_factor(metadata, factor_name):
    """候选池只接收不依赖其他因子输出的基础因子。"""
    factor_type = metadata.get("factor_type")
    if factor_type != "base":
        raise ValueError(
            f"因子 {factor_name!r} 的 factor_type={factor_type!r}，"
            "候选因子分组仅接受 factor_type='base' 的基础因子。"
        )


def _resolve_candidates(candidate_spec):
    """将候选字典展开为稳定、可审计的单个因子实例列表。"""
    if not isinstance(candidate_spec, Mapping) or not candidate_spec:
        raise TypeError("candidate_spec 必须是非空字典：{因子名: 参数列表或 None}。")

    candidates = []
    seen_ids = set()
    for factor_name, instance_spec in candidate_spec.items():
        if not isinstance(factor_name, str) or not factor_name.strip():
            raise ValueError("candidate_spec 中存在无效因子名称。")
        factor_name = factor_name.strip()
        _require_base_factor(get_factor_metadata(factor_name), factor_name)

        if instance_spec is None:
            instances = _metadata_candidate_instances(factor_name)
        elif isinstance(instance_spec, Mapping):
            instances = [{"instance_id": "custom", "params": instance_spec}]
        elif isinstance(instance_spec, Sequence) and not isinstance(
            instance_spec, (str, bytes)
        ):
            instances = []
            for position, params in enumerate(instance_spec, start=1):
                instances.append(
                    {
                        "instance_id": f"custom_{position}",
                        "params": params,
                    }
                )
        else:
            raise TypeError(
                f"candidate_spec[{factor_name!r}] 必须为参数字典、参数字典列表或 None。"
            )

        for item in instances:
            params = _normalize_candidate_params(item["params"], factor_name)
            instance_id = str(item["instance_id"]).strip()
            if not instance_id:
                raise ValueError(f"{factor_name!r} 存在空的候选实例 id。")
            candidate_id = f"{factor_name}::{instance_id}"
            if candidate_id in seen_ids:
                raise ValueError(f"候选实例重复：{candidate_id}")
            seen_ids.add(candidate_id)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "factor_name": factor_name,
                    "instance_id": instance_id,
                    "factor_params": params,
                }
            )
    return candidates


def _newey_west_t(values, max_lag):
    """对 RankIC 均值计算 Newey-West(HAC) t 值，处理标签重叠的序列相关。"""
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    sample_count = len(series)
    if sample_count < 3:
        return np.nan, np.nan, 0

    values = series.to_numpy(dtype=float)
    mean = float(values.mean())
    centered = values - mean
    max_lag = min(max(0, int(max_lag)), sample_count - 1)
    long_run_variance = float(np.dot(centered, centered) / sample_count)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / sample_count)
        long_run_variance += 2.0 * weight * covariance

    standard_error = math.sqrt(max(long_run_variance, 0.0) / sample_count)
    if standard_error == 0.0:
        t_value = np.sign(mean) * np.inf if mean != 0.0 else np.nan
    else:
        t_value = mean / standard_error
    return t_value, standard_error, max_lag


def _evaluate_candidate_panel(
    factor_data,
    labels,
    factor_column,
    min_obs,
    hac_lags,
    show_progress,
    progress_every,
    started_at,
    candidate_id,
    candidate_position,
    candidate_total,
    stage_number,
    stage_total,
):
    required = {"date", "instrument", factor_column}
    missing = required - set(factor_data.columns)
    if missing:
        raise ValueError(f"因子计算结果缺少字段：{sorted(missing)}")

    factor_panel = factor_data[["date", "instrument", factor_column]].copy()
    factor_panel["date"] = pd.to_datetime(factor_panel["date"], errors="coerce").dt.normalize()
    factor_panel[factor_column] = pd.to_numeric(
        factor_panel[factor_column], errors="coerce"
    )
    factor_panel = factor_panel.dropna(subset=["date", "instrument"])
    if factor_panel.duplicated(["date", "instrument"]).any():
        raise ValueError("因子输出存在重复的 date + instrument。")

    panel = labels.merge(
        factor_panel,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    ).replace([np.inf, -np.inf], np.nan)

    records = []
    grouped = list(panel.groupby("date", sort=True))
    total_dates = len(grouped)
    for date_position, (date, cross_section) in enumerate(grouped, start=1):
        valid = cross_section[[factor_column, "forward_return", "return_end_date"]].dropna()
        sample_count = len(valid)
        rank_ic = np.nan
        if sample_count >= min_obs:
            rank_ic = valid[factor_column].corr(valid["forward_return"], method="spearman")
        records.append(
            {
                "signal_date": date,
                "return_end_date": valid["return_end_date"].max() if sample_count else pd.NaT,
                "sample_count": sample_count,
                "rank_ic": rank_ic,
            }
        )
        if show_progress and (
            date_position == 1
            or date_position % progress_every == 0
            or date_position == total_dates
        ):
            _render_progress(
                stage_number,
                stage_total,
                f"计算候选实例 RankIC（第 {candidate_position}/{candidate_total} 个）",
                started_at,
                completed=date_position,
                total=total_dates,
                current=f"{candidate_id}；{date:%Y-%m-%d}",
            )
    rankic_timeseries = pd.DataFrame(records)
    rank_ic_mean = rankic_timeseries["rank_ic"].mean()
    rank_ic_std = rankic_timeseries["rank_ic"].std(ddof=1)
    rank_icir = (
        rank_ic_mean / rank_ic_std
        if pd.notna(rank_ic_std) and rank_ic_std != 0
        else np.nan
    )
    hac_t, hac_se, used_lags = _newey_west_t(rankic_timeseries["rank_ic"], hac_lags)
    possible = len(labels)
    valid_pairs = panel[[factor_column, "forward_return"]].dropna().shape[0]
    coverage = valid_pairs / possible if possible else np.nan
    value_panel = factor_panel.rename(columns={factor_column: "factor_value"})
    return {
        "value_panel": value_panel,
        "rankic_timeseries": rankic_timeseries,
        "rank_ic_mean": rank_ic_mean,
        "rank_icir": rank_icir,
        "rank_ic_hac_t": hac_t,
        "rank_ic_hac_se": hac_se,
        "hac_lags": used_lags,
        "valid_date_count": int(rankic_timeseries["rank_ic"].notna().sum()),
        "coverage": coverage,
    }


def _pair_similarity(left_panel, right_panel, min_obs):
    merged = left_panel.merge(
        right_panel,
        on=["date", "instrument"],
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    ).replace([np.inf, -np.inf], np.nan)
    daily_correlations = []
    for _, cross_section in merged.groupby("date", sort=False):
        valid = cross_section[["factor_value_left", "factor_value_right"]].dropna()
        if len(valid) >= min_obs:
            correlation = valid["factor_value_left"].corr(
                valid["factor_value_right"], method="spearman"
            )
            if pd.notna(correlation):
                daily_correlations.append(float(correlation))
    if not daily_correlations:
        return np.nan, 0
    return float(np.median(np.abs(daily_correlations))), len(daily_correlations)


def _cluster_similarity(similarity, cluster_count):
    candidate_ids = list(similarity.index)
    if len(candidate_ids) == 1:
        return pd.Series([1], index=candidate_ids, name="group_id"), candidate_ids

    try:
        from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        raise ImportError("候选因子聚类需要 scipy；请在 BigQuant 环境安装或启用 scipy。") from exc

    numeric = similarity.to_numpy(dtype=float)
    # 没有共同有效截面的候选对，按“完全不相似”处理；同时保留 warning 列供研究者审计。
    numeric = np.where(np.isfinite(numeric), numeric, 0.0)
    numeric = np.clip((numeric + numeric.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(numeric, 1.0)
    distance = 1.0 - numeric
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    raw_groups = fcluster(linkage_matrix, t=cluster_count, criterion="maxclust")
    leaf_order = leaves_list(linkage_matrix).tolist()

    # fcluster 的编号没有阅读顺序；依树叶首次出现的位置稳定重编号为 1, 2, ...
    remapping = {}
    next_group = 1
    for position in leaf_order:
        raw_group = int(raw_groups[position])
        if raw_group not in remapping:
            remapping[raw_group] = next_group
            next_group += 1
    groups = pd.Series(
        [remapping[int(value)] for value in raw_groups],
        index=candidate_ids,
        name="group_id",
    )
    return groups, [candidate_ids[position] for position in leaf_order]


def _plot_similarity_heatmap(similarity, ordered_ids, title, show=True):
    import matplotlib.pyplot as plt

    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    matrix = similarity.loc[ordered_ids, ordered_ids]
    size = max(7.0, min(0.55 * len(ordered_ids) + 4.0, 24.0))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="YlOrRd")
    ax.set_xticks(range(len(ordered_ids)), labels=ordered_ids, rotation=75, ha="right")
    ax.set_yticks(range(len(ordered_ids)), labels=ordered_ids)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("每日截面 |Spearman 相关系数| 的中位数")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _display_dataframe(frame):
    try:
        from IPython.display import display

        display(frame)
    except ImportError:
        print(frame.to_string(index=False))


def _rank_ic_direction(value):
    """将原始 RankIC 符号转换为便于阅读的方向标签。"""
    if pd.isna(value) or abs(float(value)) < 1e-12:
        return "方向不明显"
    return "正向" if value > 0 else "负向"


def _evidence_level(hac_t):
    """仅供展示，不参与有效性分数或组内排序。"""
    if pd.isna(hac_t):
        return "无法判断"
    absolute_t = abs(float(hac_t))
    if absolute_t >= 3.0:
        return "强"
    if absolute_t >= 2.0:
        return "待验证"
    return "证据不足"


def _build_group_ranking_display(ranking):
    """构造 Notebook 阅读版排序表，保留原始参数以支持实例间比较。"""
    display = ranking.loc[
        :,
        [
            "group_id",
            "group_rank",
            "factor_name",
            "instance_id",
            "factor_params",
            "rank_ic_mean",
            "rank_ic_hac_t",
            "coverage",
            "effectiveness_score",
        ],
    ].copy()
    display.insert(
        5,
        "direction",
        display["rank_ic_mean"].map(_rank_ic_direction),
    )
    display.insert(
        8,
        "evidence_level",
        display["rank_ic_hac_t"].map(_evidence_level),
    )
    display = display.rename(
        columns={
            "group_id": "组别",
            "group_rank": "组内排名",
            "factor_name": "因子",
            "instance_id": "实例",
            "factor_params": "参数",
            "direction": "方向",
            "rank_ic_mean": "平均 RankIC",
            "rank_ic_hac_t": "HAC t 值",
            "evidence_level": "统计证据",
            "coverage": "覆盖率",
            "effectiveness_score": "有效性分数",
        }
    )
    display["平均 RankIC"] = display["平均 RankIC"].map(
        lambda value: "—" if pd.isna(value) else f"{float(value):+.4f}"
    )
    display["HAC t 值"] = display["HAC t 值"].map(
        lambda value: "—" if pd.isna(value) else f"{float(value):+.2f}"
    )
    display["覆盖率"] = display["覆盖率"].map(
        lambda value: "—" if pd.isna(value) else f"{float(value):.1%}"
    )
    display["有效性分数"] = display["有效性分数"].map(
        lambda value: "—" if pd.isna(value) else f"{float(value):.4f}"
    )
    return display


def _display_group_ranking(group_summary, ranking_display):
    """先展示组代表，再按组展示紧凑的组内候选清单。"""
    _display_dataframe(group_summary)
    try:
        from IPython.display import Markdown, display

        show_title = lambda text: display(Markdown(f"#### {text}"))
    except ImportError:
        show_title = print

    summary_by_group = group_summary.set_index("group_id")
    for group_id, group in ranking_display.groupby("组别", sort=True):
        representative = summary_by_group.loc[group_id]
        show_title(
            f"第 {group_id} 组｜代表："
            f"{representative['representative_candidate_id']}｜"
            f"成员：{int(representative['member_count'])} 个"
        )
        _display_dataframe(group.reset_index(drop=True))


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"无法序列化为 JSON：{type(value).__name__}")


def _frame_records(frame):
    """将 DataFrame 转换为 JSON 友好的 records，且把 NaN 写为 null。"""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _safe_file_stem(run_name):
    if run_name is None:
        return "factor_candidate_management"
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name 必须是非空字符串或 None。")
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", run_name.strip()).strip("._-")
    if not stem:
        raise ValueError("run_name 未包含可用于文件名的字符。")
    return stem


def _save_management_result(
    output_dir,
    run_name,
    run_parameters,
    group_summary,
    ranking,
    similarity,
    pair_counts,
    schedule,
    resolved_candidates,
    universe_panel,
):
    """显式保存单个、机器与 LLM 均可读取的 JSON 研究报告。"""
    directory = Path(output_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(
            f"输出目录不存在或不是文件夹：{directory}。"
            "请先创建 factor_management_outputs，再传入该目录。"
        )

    stem = _safe_file_stem(run_name)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{stem}_{timestamp}.json"
    serial = 1
    while path.exists():
        path = directory / f"{stem}_{timestamp}_{serial}.json"
        serial += 1

    payload = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now().isoformat(),
        "run_parameters": run_parameters,
        "group_summary": _frame_records(group_summary),
        "group_ranking": _frame_records(ranking),
        "resolved_candidates": _frame_records(resolved_candidates),
        "evaluation_schedule": _frame_records(schedule),
        "universe_panel": (
            None if universe_panel is None else _frame_records(universe_panel)
        ),
        "similarity_matrix": {
            "index": list(similarity.index),
            "columns": list(similarity.columns),
            "data": similarity.astype(object).where(similarity.notna(), None).values.tolist(),
        },
        "pairwise_observation_count": {
            "index": list(pair_counts.index),
            "columns": list(pair_counts.columns),
            "data": pair_counts.astype(object).where(pair_counts.notna(), None).values.tolist(),
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def cluster_and_rank_factor_candidates(
    candidate_spec,
    start_date,
    end_date,
    frequency,
    holding_period_days,
    cluster_count,
    instruments=None,
    universe=None,
    min_obs=30,
    hac_lags=None,
    significance_threshold=2.0,
    similarity_min_obs=None,
    plot=True,
    display_results=True,
    save_result=False,
    output_dir="factor_lib/factor_management_outputs",
    run_name=None,
    show_progress=True,
    progress_every=20,
):
    """对候选因子实例做暴露相似度聚类，并输出组内有效性排序。

    Parameters
    ----------
    candidate_spec : dict
        唯一的候选输入。键为因子名称，值为参数字典、参数字典列表或 None。
        ``None`` 表示展开该因子 ``FACTOR['candidate_instances']`` 中的全部实例。
    start_date, end_date : str or datetime
        研究区间；``end_date`` 同时是未来标签的截止日，未在其前结束的标签会剔除。
    frequency : int
        每隔多少个交易日取一个因子截面。
    holding_period_days : int
        标签收益的未来交易日跨度。它独立于 ``frequency``。
    cluster_count : int
        期望分成的候选因子组数，不能超过展开后的候选实例数。
    instruments : sequence[str] or None
        固定股票池；None 为适配器返回的全 A 股。
    universe : "all_a"、股票代码集合或 dict，可选
        点时股票池定义。指数股票池例如
        ``{"type": "index", "index_codes": ["000300.SH"]}``；每个信号日
        仅使用当日历史成分股计算 RankIC 和因子暴露相似度。传入时不得再传
        instruments。
    min_obs : int
        单截面计算 RankIC 的最小完整样本数。
    hac_lags : int or None
        RankIC 均值 HAC t 值的 Newey-West 滞后阶数。None 时按
        ``ceil(holding_period_days / frequency) - 1`` 自动处理标签重叠。
    significance_threshold : float
        有效性分数中“统计可信度达到饱和”的 |HAC t| 阈值，默认 2。
    similarity_min_obs : int or None
        两个因子在单日截面计算暴露相关性所需的最小共同股票数；默认沿用 min_obs。
    save_result : bool, default False
        True 时将本次运行的原始 ``group_ranking``、分组概要、相似度矩阵、
        运行参数和候选实例保存为单个 JSON 文件；默认不写文件。
    output_dir : str or path-like
        ``save_result=True`` 时使用的既有输出目录；默认是
        ``factor_lib/factor_management_outputs``。函数不会自动创建该目录。
    run_name : str or None
        保存文件的可读名称前缀；None 时使用 ``factor_candidate_management``。

    Returns
    -------
    dict
        ``group_summary``：每个组的代表候选实例；
        ``candidate_ranking``：所有候选实例的原始 RankIC、HAC t、覆盖率与组内名次；
        ``group_ranking``：与 ``candidate_ranking`` 相同的兼容别名；
        ``group_ranking_display``：Notebook 分组展示版，保留原始参数字典；
        ``saved_result_path``：显式保存时生成的 JSON 文件路径，否则为 None；
        ``similarity_matrix``：聚类依据；
        ``pairwise_observation_count``：每对因子的有效共同截面数；
        ``evaluation_schedule``、``universe_config``、``universe_panel``、
        ``heatmap_figure``、``heatmap_axis``。

    Notes
    -----
    相似度定义为 ``median(abs(daily cross-sectional Spearman correlation))``。
    组内 ``effectiveness_score`` 为：

    ``abs(mean RankIC) * min(1, abs(HAC t) / significance_threshold)``。

    因此它保留效应强度，不会因为 t 值特别大而无限加分；排序表仍展示原始
    RankIC 符号，方便识别该因子在研究期内的实际方向。
    """
    started_at = time.perf_counter()
    stage_total = 8 if save_result else 7
    try:
        start_date = _normalize_date(start_date, "start_date")
        end_date = _normalize_date(end_date, "end_date")
        if start_date > end_date:
            raise ValueError("start_date 不能晚于 end_date。")
        frequency = _positive_integer(frequency, "frequency")
        holding_period_days = _positive_integer(
            holding_period_days, "holding_period_days"
        )
        cluster_count = _positive_integer(cluster_count, "cluster_count")
        min_obs = _positive_integer(min_obs, "min_obs")
        progress_every = _positive_integer(progress_every, "progress_every")
        if similarity_min_obs is None:
            similarity_min_obs = min_obs
        similarity_min_obs = _positive_integer(
            similarity_min_obs, "similarity_min_obs"
        )
        if not isinstance(significance_threshold, (int, float, np.number)) or significance_threshold <= 0:
            raise ValueError("significance_threshold 必须为正数。")
        significance_threshold = float(significance_threshold)
        if not isinstance(save_result, (bool, np.bool_)):
            raise TypeError("save_result 必须为 bool。")
        if not isinstance(output_dir, (str, Path)):
            raise TypeError("output_dir 必须是路径字符串或 pathlib.Path。")
        if universe is not None and instruments is not None:
            raise ValueError("universe 与 instruments 互斥；请只指定一种股票池。")
        instruments = _normalize_instruments(instruments)
        candidates = _resolve_candidates(candidate_spec)
        if cluster_count > len(candidates):
            raise ValueError(
                f"cluster_count={cluster_count} 不能超过候选实例数 {len(candidates)}。"
            )
        if hac_lags is None:
            hac_lags = max(0, math.ceil(holding_period_days / frequency) - 1)
        else:
            hac_lags = _positive_integer(hac_lags + 1, "hac_lags + 1") - 1

        if show_progress:
            _render_progress(1, stage_total, "读取交易日历并生成完整标签计划", started_at)
        trading_calendar = _query_trading_calendar(end_date)
        schedule = _build_schedule(
            trading_calendar,
            start_date,
            end_date,
            frequency,
            holding_period_days,
        )

        target_dates = pd.DatetimeIndex(schedule["date"])
        if show_progress:
            _render_progress(
                2,
                stage_total,
                "读取并校验点时动态股票池" if universe is not None else "确认固定股票池范围",
                started_at,
                current=f"{len(schedule)} 个信号日",
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

        if show_progress:
            _render_progress(
                3,
                stage_total,
                "读取价格并构造未来收益标签",
                started_at,
                current=f"{len(schedule)} 个信号日，持有期 {holding_period_days} 日",
            )
        label_data = _prepare_labels(schedule, load_instruments, show_progress)
        label_data = _filter_panel_to_universe(
            label_data,
            universe_panel,
            "label_data",
        )
        if show_progress:
            _render_progress(
                3,
                stage_total,
                "未来收益标签构造完成",
                started_at,
                completed=len(schedule),
                total=len(schedule),
                current=f"{len(label_data):,} 条完整标签",
            )

        if show_progress:
            _render_progress(
                4,
                stage_total,
                "依次计算候选因子实例",
                started_at,
                completed=0,
                total=len(candidates),
            )

        candidate_results = []
        for position, candidate in enumerate(candidates, start=1):
            factor_name = candidate["factor_name"]
            metadata = get_factor_metadata(factor_name)
            requirements = get_factor_data_requirements(
                factor_name, candidate["factor_params"]
            )
            params = _resolved_factor_params(requirements)
            factor_dates = _factor_input_dates(
                target_dates,
                trading_calendar,
                _history_days(requirements),
            )
            if show_progress:
                _render_progress(
                    4,
                    stage_total,
                    "准备并计算候选因子实例",
                    started_at,
                    completed=position - 1,
                    total=len(candidates),
                    current=f"{candidate['candidate_id']}；输入 {len(factor_dates)} 日",
                )

            # loader 与基础因子函数保留各自的真实进度。
            raw_data = load_factor_raw_data(
                factor_name=factor_name,
                dates=factor_dates,
                factor_params=params,
                instruments=load_instruments,
                show_progress=show_progress,
            )
            factor_data = get_factor(
                factor_name,
                raw_data,
                target_dates=target_dates,
                as_of_date=end_date,
                **params,
                show_progress=show_progress,
                progress_every=progress_every,
            )
            factor_data = _filter_panel_to_universe(
                factor_data,
                universe_panel,
                "factor_data",
            )
            factor_column = _factor_value_column(metadata, factor_name)
            evaluation = _evaluate_candidate_panel(
                factor_data,
                label_data,
                factor_column,
                min_obs,
                hac_lags,
                show_progress,
                progress_every,
                started_at,
                candidate["candidate_id"],
                position,
                len(candidates),
                4,
                stage_total,
            )
            candidate_results.append({**candidate, **evaluation})
            if show_progress:
                _render_progress(
                    4,
                    stage_total,
                    "候选因子实例计算完成",
                    started_at,
                    completed=position,
                    total=len(candidates),
                    current=candidate["candidate_id"],
                )

        if show_progress:
            _render_progress(
                5,
                stage_total,
                "计算候选因子两两暴露相似度",
                started_at,
            )
        candidate_ids = [result["candidate_id"] for result in candidate_results]
        similarity = pd.DataFrame(np.eye(len(candidate_ids)), index=candidate_ids, columns=candidate_ids)
        pair_counts = pd.DataFrame(np.nan, index=candidate_ids, columns=candidate_ids)
        np.fill_diagonal(pair_counts.values, len(schedule))
        pair_total = len(candidate_results) * (len(candidate_results) - 1) // 2
        pair_position = 0
        for left_position in range(len(candidate_results)):
            for right_position in range(left_position + 1, len(candidate_results)):
                pair_position += 1
                left = candidate_results[left_position]
                right = candidate_results[right_position]
                value, count = _pair_similarity(
                    left["value_panel"], right["value_panel"], similarity_min_obs
                )
                similarity.loc[left["candidate_id"], right["candidate_id"]] = value
                similarity.loc[right["candidate_id"], left["candidate_id"]] = value
                pair_counts.loc[left["candidate_id"], right["candidate_id"]] = count
                pair_counts.loc[right["candidate_id"], left["candidate_id"]] = count
                if show_progress and (
                    pair_position == 1
                    or pair_position % progress_every == 0
                    or pair_position == pair_total
                ):
                    _render_progress(
                        5,
                        stage_total,
                        "计算候选因子两两暴露相似度",
                        started_at,
                        completed=pair_position,
                        total=pair_total,
                        current=f"{left['candidate_id']} × {right['candidate_id']}",
                    )

        if show_progress:
            _render_progress(6, stage_total, "层次聚类并进行组内排序", started_at)
        group_ids, leaf_order = _cluster_similarity(similarity, cluster_count)
        rows = []
        for result in candidate_results:
            confidence = (
                min(1.0, abs(result["rank_ic_hac_t"]) / significance_threshold)
                if pd.notna(result["rank_ic_hac_t"])
                else 0.0
            )
            score = abs(result["rank_ic_mean"]) * confidence if pd.notna(result["rank_ic_mean"]) else 0.0
            rows.append(
                {
                    "group_id": int(group_ids.loc[result["candidate_id"]]),
                    "candidate_id": result["candidate_id"],
                    "factor_name": result["factor_name"],
                    "instance_id": result["instance_id"],
                    "factor_params": result["factor_params"],
                    "rank_ic_mean": result["rank_ic_mean"],
                    "rank_icir": result["rank_icir"],
                    "rank_ic_hac_t": result["rank_ic_hac_t"],
                    "rank_ic_hac_se": result["rank_ic_hac_se"],
                    "hac_lags": result["hac_lags"],
                    "valid_date_count": result["valid_date_count"],
                    "coverage": result["coverage"],
                    "effectiveness_score": score,
                }
            )
        ranking = pd.DataFrame(rows)
        ranking["group_rank"] = (
            ranking.groupby("group_id")["effectiveness_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        ranking = ranking.sort_values(
            [
                "group_id",
                "group_rank",
                "rank_ic_mean",
                "rank_ic_hac_t",
                "coverage",
                "valid_date_count",
            ],
            ascending=[True, True, False, False, False, False],
            key=lambda values: values.abs()
            if values.name in {"rank_ic_mean", "rank_ic_hac_t"}
            else values,
            kind="mergesort",
        ).reset_index(drop=True)
        group_summary = (
            ranking.sort_values(["group_id", "group_rank"])
            .groupby("group_id", as_index=False)
            .first()[
                [
                    "group_id",
                    "candidate_id",
                    "factor_name",
                    "instance_id",
                    "effectiveness_score",
                    "rank_ic_mean",
                    "rank_ic_hac_t",
                    "coverage",
                ]
            ]
            .rename(
                columns={
                    "candidate_id": "representative_candidate_id",
                    "factor_name": "representative_factor_name",
                    "instance_id": "representative_instance_id",
                    "effectiveness_score": "representative_effectiveness_score",
                    "rank_ic_mean": "representative_rank_ic_mean",
                    "rank_ic_hac_t": "representative_rank_ic_hac_t",
                    "coverage": "representative_coverage",
                }
            )
        )
        group_sizes = ranking.groupby("group_id").size().rename("member_count")
        group_summary = group_summary.merge(group_sizes, on="group_id", how="left")
        order_map = {candidate_id: position for position, candidate_id in enumerate(leaf_order)}
        ordered_ids = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                int(group_ids.loc[candidate_id]),
                order_map[candidate_id],
            ),
        )

        figure = None
        axis = None
        if plot:
            if show_progress:
                _render_progress(7, stage_total, "绘制并展示候选因子相似度热力图", started_at)
            figure, axis = _plot_similarity_heatmap(
                similarity,
                ordered_ids,
                "候选因子暴露相似度热力图",
                show=True,
            )

        ranking_display = _build_group_ranking_display(ranking)
        ordered_similarity = similarity.loc[ordered_ids, ordered_ids]
        ordered_pair_counts = pair_counts.loc[ordered_ids, ordered_ids]
        resolved_candidates = pd.DataFrame(
            [
                {
                    "candidate_id": item["candidate_id"],
                    "factor_name": item["factor_name"],
                    "instance_id": item["instance_id"],
                    "factor_params": item["factor_params"],
                }
                for item in candidates
            ]
        )

        saved_result_path = None
        if save_result:
            if show_progress:
                _render_progress(8, stage_total, "保存候选因子管理 JSON 报告", started_at)
            saved_result_path = _save_management_result(
                output_dir=output_dir,
                run_name=run_name,
                run_parameters={
                    "candidate_spec": candidate_spec,
                    "start_date": start_date,
                    "end_date": end_date,
                    "frequency": frequency,
                    "holding_period_days": holding_period_days,
                    "cluster_count": cluster_count,
                    "instruments": instruments,
                    "universe": universe_config,
                    "min_obs": min_obs,
                    "hac_lags": hac_lags,
                    "significance_threshold": significance_threshold,
                    "similarity_min_obs": similarity_min_obs,
                },
                group_summary=group_summary,
                ranking=ranking,
                similarity=ordered_similarity,
                pair_counts=ordered_pair_counts,
                schedule=schedule,
                resolved_candidates=resolved_candidates,
                universe_panel=universe_panel,
            )
        if display_results:
            _display_group_ranking(group_summary, ranking_display)
        if show_progress:
            _render_progress(
                stage_total,
                stage_total,
                "完成",
                started_at,
                completed=len(candidates),
                total=len(candidates),
                current=(
                    f"{len(group_summary)} 个因子组"
                    if saved_result_path is None
                    else f"{len(group_summary)} 个因子组；已保存 {saved_result_path}"
                ),
            )
        return {
            "group_summary": group_summary,
            "candidate_ranking": ranking,
            "group_ranking": ranking,
            "group_ranking_display": ranking_display,
            "similarity_matrix": ordered_similarity,
            "pairwise_observation_count": ordered_pair_counts,
            "evaluation_schedule": schedule,
            "universe_config": universe_config,
            "universe_panel": universe_panel,
            "heatmap_figure": figure,
            "heatmap_axis": axis,
            "resolved_candidates": resolved_candidates,
            "saved_result_path": saved_result_path,
        }
    finally:
        if show_progress:
            print()
