# -*- coding: utf-8 -*-
"""BigQuant 指数增强回测策略。

本模块仅面向历史回测。它预先读取回测区间内可能用到的数据，在每个
信号日收盘后计算因子和目标权重，并在下一交易日使用 BigTrader 下单。

支持三种组合构建方法：
1. benchmark_tilt：基准权重连续倾斜；
2. stratified_sampling：行业 × 市值分层抽样；
3. constrained_optimization：基准相对约束优化。

支持固定交易日间隔、指定执行日、偏离触发和混合触发。指数模式只允许
交易指数历史成分股，且历史指数权重缺失时直接报错；不会静默退化成等权。
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


_RESERVED_FACTOR_PARAMS = {
    "data",
    "domain_data",
    "target_dates",
    "as_of_date",
    "show_progress",
    "progress_every",
}
_PRICE_FIELDS = {"open", "close", "vwap"}
_INDUSTRY_FIELDS = {
    "citic_l1": "cs_level1",
    "citic_l2": "cs_level2",
    "citic_l3": "cs_level3",
    "sw2014_l1": "sw2014_level1",
    "sw2014_l2": "sw2014_level2",
    "sw2014_l3": "sw2014_level3",
    "sw2021_l1": "sw2021_level1",
    "sw2021_l2": "sw2021_level2",
    "sw2021_l3": "sw2021_level3",
}
_STYLE_FIELDS = {
    "BETA",
    "SIZE",
    "SIZENL",
    "BTOP",
    "MOMENTUM",
    "RESVOL",
    "LIQUIDTY",
    "EARNYILD",
    "GROWTH",
    "LEVERAGE",
}


def _timestamp(value, name):
    try:
        result = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是可解析日期：{value!r}") from exc
    if pd.isna(result):
        raise ValueError(f"{name} 不允许为空。")
    return result


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是正整数。")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正整数。") from exc
    if result <= 0 or float(value) != result:
        raise ValueError(f"{name} 必须是正整数。")
    return result


def _mapping(value, name, default=None):
    if value is None:
        return {} if default is None else dict(default)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是字典或 None。")
    result = {} if default is None else dict(default)
    result.update(value)
    return result


def _instruments(values, name="instruments"):
    if isinstance(values, str):
        values = [values]
    if isinstance(values, (bytes, bytearray)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} 必须是股票代码序列。")
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 中存在无效股票代码：{value!r}")
        code = value.strip()
        if code not in result:
            result.append(code)
    if not result:
        raise ValueError(f"{name} 不能为空。")
    return result


def _factor_params(value):
    result = _mapping(value, "factor_params")
    conflicts = sorted(_RESERVED_FACTOR_PARAMS.intersection(result))
    if conflicts:
        raise ValueError(
            "factor_params 包含由策略控制的保留参数：" f"{conflicts}。"
        )
    return result


def _quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _progress(
    stage_number,
    stage_total,
    stage,
    started_at,
    completed=None,
    total=None,
    current=None,
    detail="",
):
    elapsed = time.perf_counter() - started_at
    parts = [f"[指数增强回测] [{stage_number}/{stage_total}] {stage}"]
    if completed is not None and total:
        ratio = completed / total
        parts.append(f"{completed}/{total} ({ratio:.1%})")
        if 0 < completed < total:
            parts.append(f"预计剩余 {elapsed / completed * (total - completed):.1f}s")
    if current:
        parts.append(f"当前 {current}")
    if detail:
        parts.append(str(detail))
    parts.append(f"已耗时 {elapsed:.1f}s")
    print("\r" + " | ".join(parts).ljust(240), end="", flush=True)


def _heartbeat(action, stage_number, stage_total, stage, started_at, enabled, **kwargs):
    if not enabled:
        return action()
    stop = threading.Event()

    def beat():
        while not stop.wait(2.0):
            _progress(
                stage_number,
                stage_total,
                f"{stage}（仍在运行）",
                started_at,
                **kwargs,
            )

    worker = threading.Thread(target=beat, daemon=True)
    worker.start()
    try:
        return action()
    finally:
        stop.set()
        worker.join(timeout=2.1)


def _query_calendar(end_date, show_progress, started_at):
    import dai

    sql = f"""
    SELECT DISTINCT date
    FROM cn_stock_bar1d
    WHERE date <= '{end_date:%Y-%m-%d}'
    ORDER BY date
    """
    frame = _heartbeat(
        lambda: dai.query(sql).df(),
        1,
        9,
        "读取A股交易日历",
        started_at,
        show_progress,
        current=f"截止{end_date:%Y-%m-%d}",
    )
    if frame.empty or "date" not in frame:
        raise ValueError("未读取到有效的 A 股交易日历。")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce"))
    if dates.isna().any():
        raise ValueError("交易日历中存在无效日期。")
    return dates.normalize().unique().sort_values()


def _execution_pairs(calendar, start_date, end_date):
    execution_dates = calendar[(calendar >= start_date) & (calendar <= end_date)]
    if execution_dates.empty:
        raise ValueError("回测区间内没有交易日。")
    positions = {date: index for index, date in enumerate(calendar)}
    rows = []
    for execution_date in execution_dates:
        position = positions[execution_date]
        if position == 0:
            raise ValueError("首次执行日前没有交易日，无法形成前一日收盘信号。")
        rows.append(
            {
                "signal_date": calendar[position - 1],
                "execution_date": execution_date,
                "execution_position": position,
            }
        )
    return pd.DataFrame(rows)


def _normalize_reference(reference_portfolio):
    config = _mapping(reference_portfolio, "reference_portfolio")
    reference_type = str(config.get("type", "")).strip().lower()
    if reference_type == "index":
        code = config.get("index_code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("指数模式必须提供非空 index_code。")
        return {"type": "index", "index_code": code.strip()}
    if reference_type != "custom":
        raise ValueError("reference_portfolio['type'] 仅支持 index 或 custom。")
    members = _instruments(config.get("instruments", []))
    method = str(config.get("base_weight_method", "equal")).strip().lower()
    if method not in {"equal", "market_cap", "explicit"}:
        raise ValueError(
            "custom.base_weight_method 仅支持 equal、market_cap、explicit。"
        )
    result = {"type": "custom", "instruments": members, "base_weight_method": method}
    if method == "explicit":
        weights = config.get("weights")
        if not isinstance(weights, Mapping):
            raise TypeError("explicit 模式的 weights 必须是股票代码到权重的字典。")
        if set(weights) != set(members):
            raise ValueError("explicit weights 的股票集合必须与 instruments 完全一致。")
        normalized = {key: float(value) for key, value in weights.items()}
        if any(not np.isfinite(v) or v < 0 for v in normalized.values()):
            raise ValueError("explicit weights 必须是有限非负数。")
        if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-8):
            raise ValueError("explicit weights 必须精确加总为 1，不会静默归一化。")
        result["weights"] = normalized
    return result


def _query_index_weights(index_code, signal_dates, show_progress, started_at):
    import dai

    if signal_dates.empty:
        raise ValueError("指数权重查询日期不能为空。")
    sql = f"""
    SELECT
        date,
        instrument AS index_code,
        member_code AS instrument,
        weight
    FROM cn_stock_index_weight
    WHERE instrument = {_quote(index_code)}
      AND date BETWEEN '{signal_dates.min():%Y-%m-%d}'
                   AND '{signal_dates.max():%Y-%m-%d}'
    ORDER BY date, member_code
    """
    filters = {
        "date": [
            signal_dates.min().strftime("%Y-%m-%d"),
            signal_dates.max().strftime("%Y-%m-%d"),
        ]
    }
    frame = _heartbeat(
        lambda: dai.query(sql, filters=filters).df(),
        2,
        9,
        "读取历史指数成分与权重",
        started_at,
        show_progress,
        current=index_code,
        detail=f"{len(signal_dates)}个信号日",
    )
    if frame.empty:
        raise ValueError(f"未读取到指数 {index_code} 的历史权重。")
    required = {"date", "index_code", "instrument", "weight"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"指数权重结果缺少字段：{sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    if frame[["date", "instrument", "weight"]].isna().any().any():
        raise ValueError("指数权重中存在空日期、股票代码或权重。")
    if frame.duplicated(["date", "instrument"]).any():
        raise ValueError("指数权重表中存在重复 date + instrument。")
    covered = pd.DatetimeIndex(frame["date"].unique())
    missing_dates = signal_dates.difference(covered)
    if not missing_dates.empty:
        preview = [d.strftime("%Y-%m-%d") for d in missing_dates[:5]]
        raise ValueError(
            "历史指数权重缺少需要日期，策略不会退化成近似权重：" f"{preview}"
        )

    parts = []
    scale_records = []
    for date, group in frame.groupby("date", sort=True):
        group = group.copy()
        total = float(group["weight"].sum())
        if total > 2.0:
            group["weight"] = group["weight"] / 100.0
            source_scale = "percent"
        else:
            source_scale = "fraction"
        total = float(group["weight"].sum())
        if not 0.98 <= total <= 1.02:
            raise ValueError(
                f"{date:%Y-%m-%d} 的指数权重合计为 {total:.6f}，"
                "超出允许的舍入误差范围。"
            )
        # 官方权重可能因小数位截断轻微偏离 1；仅在已验证误差范围内修正。
        group["weight"] = group["weight"] / total
        scale_records.append(
            {"date": date, "source_scale": source_scale, "source_sum": total}
        )
        parts.append(group[["date", "instrument", "weight"]])
    return pd.concat(parts, ignore_index=True), pd.DataFrame(scale_records)


def _membership_change_dates(weight_panel):
    result = []
    previous = None
    for date, group in weight_panel.groupby("date", sort=True):
        current = frozenset(group["instrument"])
        if previous is not None and current != previous:
            result.append(pd.Timestamp(date))
        previous = current
    return pd.DatetimeIndex(result)


def _normalize_rebalance_rule(rule):
    config = _mapping(rule, "rebalance_rule", {"type": "fixed_interval", "interval_trading_days": 20})
    rule_type = str(config.get("type", "fixed_interval")).strip().lower()
    if rule_type == "fixed_interval":
        return {"type": rule_type, "interval_trading_days": _positive_int(config.get("interval_trading_days", 20), "interval_trading_days")}
    if rule_type == "custom_dates":
        dates = config.get("execution_dates")
        if isinstance(dates, (str, pd.Timestamp)):
            dates = [dates]
        if not isinstance(dates, Iterable):
            raise TypeError("custom_dates.execution_dates 必须是日期序列。")
        normalized = sorted({_timestamp(value, "execution_date") for value in dates})
        if not normalized:
            raise ValueError("custom_dates.execution_dates 不能为空。")
        policy = str(config.get("non_trading_date_policy", "error")).lower()
        if policy not in {"error", "next_trading_day", "previous_trading_day"}:
            raise ValueError("non_trading_date_policy 无效。")
        return {"type": rule_type, "execution_dates": normalized, "non_trading_date_policy": policy}
    if rule_type in {"weight_drift", "risk_drift"}:
        minimum = _positive_int(config.get("min_interval_trading_days", 5), "min_interval_trading_days")
        logic = str(config.get("trigger_logic", "any")).lower()
        if logic not in {"any", "all"}:
            raise ValueError("trigger_logic 仅支持 any 或 all。")
        result = dict(config)
        result.update({"type": rule_type, "min_interval_trading_days": minimum, "trigger_logic": logic})
        if rule_type == "weight_drift":
            thresholds = _mapping(config.get("thresholds"), "thresholds")
            allowed = {"max_abs_weight_drift", "active_share"}
            if not thresholds or not set(thresholds).issubset(allowed):
                raise ValueError("weight_drift.thresholds 必须声明 max_abs_weight_drift 和/或 active_share。")
            result["thresholds"] = {key: float(value) for key, value in thresholds.items()}
            result["include_cash"] = bool(config.get("include_cash", True))
        else:
            thresholds = _mapping(config.get("thresholds"), "thresholds")
            allowed = {
                "predicted_tracking_error",
                "max_industry_active_weight",
                "max_style_active_exposure",
            }
            if not thresholds or not set(thresholds).issubset(allowed):
                raise ValueError(
                    "risk_drift.thresholds 必须声明 predicted_tracking_error、"
                    "max_industry_active_weight、max_style_active_exposure 中的"
                    "至少一个。"
                )
            result["thresholds"] = {
                key: float(value) for key, value in thresholds.items()
            }
        if any(
            not np.isfinite(value) or value <= 0
            for value in result["thresholds"].values()
        ):
            raise ValueError("偏离触发阈值必须是有限正数。")
        return result
    if rule_type == "hybrid":
        scheduled = _normalize_rebalance_rule(config.get("scheduled_rule"))
        deviation = _normalize_rebalance_rule(config.get("deviation_rule"))
        if scheduled["type"] not in {"fixed_interval", "custom_dates"}:
            raise ValueError("hybrid.scheduled_rule 必须是固定间隔或指定日期。")
        if deviation["type"] not in {"weight_drift", "risk_drift"}:
            raise ValueError("hybrid.deviation_rule 必须是权重或风险偏离。")
        return {"type": "hybrid", "scheduled_rule": scheduled, "deviation_rule": deviation}
    raise ValueError("rebalance_rule.type 仅支持 fixed_interval、custom_dates、weight_drift、risk_drift、hybrid。")


def _resolve_custom_execution_dates(config, execution_dates):
    available = pd.DatetimeIndex(execution_dates)
    resolved = []
    for requested in config["execution_dates"]:
        if requested in available:
            actual = requested
        elif config["non_trading_date_policy"] == "error":
            raise ValueError(f"指定调仓日 {requested:%Y-%m-%d} 不是回测区间内交易日。")
        elif config["non_trading_date_policy"] == "next_trading_day":
            candidates = available[available > requested]
            if candidates.empty:
                raise ValueError(f"{requested:%Y-%m-%d} 之后没有可用交易日。")
            actual = candidates[0]
        else:
            candidates = available[available < requested]
            if candidates.empty:
                raise ValueError(f"{requested:%Y-%m-%d} 之前没有可用交易日。")
            actual = candidates[-1]
        resolved.append(actual)
    return pd.DatetimeIndex(sorted(set(resolved)))


def _scheduled_execution_dates(rule, all_execution_dates):
    if rule["type"] == "fixed_interval":
        return pd.DatetimeIndex(all_execution_dates[:: rule["interval_trading_days"]])
    if rule["type"] == "custom_dates":
        return _resolve_custom_execution_dates(rule, all_execution_dates)
    if rule["type"] == "hybrid":
        return _scheduled_execution_dates(rule["scheduled_rule"], all_execution_dates)
    return pd.DatetimeIndex([])


def _uses_deviation(rule):
    return rule["type"] in {"weight_drift", "risk_drift", "hybrid"}


def _deviation_rule(rule):
    return rule["deviation_rule"] if rule["type"] == "hybrid" else rule


def _resolve_factor_column(metadata, factor_name):
    schema = metadata.get("output_schema", {})
    if isinstance(schema, Mapping):
        columns = [key for key in schema if key not in {"date", "instrument"}]
        if factor_name in columns:
            return factor_name
        if len(columns) == 1:
            return columns[0]
        if len(columns) > 1:
            raise ValueError(f"因子 {factor_name!r} 声明多个输出字段，无法自动选择。")
    return factor_name


def _history_days(requirements):
    window = requirements.get("data_window", {})
    if not isinstance(window, Mapping):
        raise ValueError("FACTOR['data_window'] 必须是字典。")
    lookback = int(window.get("lookback_trading_days", 0))
    minimum = int(window.get("minimum_history_observations", 0))
    if lookback < 0 or minimum < 0:
        raise ValueError("因子历史窗口不能为负数。")
    return max(lookback, minimum)


def _factor_windows(signal_dates, calendar, history_days):
    positions = {date: index for index, date in enumerate(calendar)}
    windows = {}
    union = set()
    for date in signal_dates:
        position = positions.get(date)
        if position is None or position - history_days < 0:
            raise ValueError(f"{date:%Y-%m-%d} 前没有足够的 {history_days} 个预热交易日。")
        window = calendar[position - history_days : position + 1]
        windows[date] = window
        union.update(window.tolist())
    return windows, pd.DatetimeIndex(sorted(union))


def _query_industry(signal_dates, instruments, scheme, show_progress, started_at):
    import dai

    if scheme not in _INDUSTRY_FIELDS:
        raise ValueError(f"industry_scheme 仅支持 {sorted(_INDUSTRY_FIELDS)}。")
    field = _INDUSTRY_FIELDS[scheme]
    instrument_sql = ", ".join(_quote(code) for code in instruments)
    sql = f"""
    SELECT date, instrument, {field} AS industry
    FROM cn_stock_factors_industry
    WHERE date BETWEEN '{signal_dates.min():%Y-%m-%d}' AND '{signal_dates.max():%Y-%m-%d}'
      AND instrument IN ({instrument_sql})
    ORDER BY date, instrument
    """
    filters = {"date": [signal_dates.min().strftime("%Y-%m-%d"), signal_dates.max().strftime("%Y-%m-%d")]}
    frame = _heartbeat(
        lambda: dai.query(sql, filters=filters).df(),
        4,
        9,
        "读取点时行业分类",
        started_at,
        show_progress,
        current=scheme,
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame.drop_duplicates(["date", "instrument"])


def _query_style(signal_dates, instruments, fields, show_progress, started_at):
    import dai

    normalized = [str(field).upper() for field in fields]
    invalid = sorted(set(normalized) - _STYLE_FIELDS)
    if invalid:
        raise ValueError(f"不支持的风格暴露字段：{invalid}")
    if not normalized:
        return pd.DataFrame(columns=["date", "instrument"])
    instrument_sql = ", ".join(_quote(code) for code in instruments)
    sql = f"""
    SELECT date, instrument, {', '.join(normalized)}
    FROM cn_stock_factors_exposure
    WHERE date BETWEEN '{signal_dates.min():%Y-%m-%d}' AND '{signal_dates.max():%Y-%m-%d}'
      AND instrument IN ({instrument_sql})
    ORDER BY date, instrument
    """
    filters = {"date": [signal_dates.min().strftime("%Y-%m-%d"), signal_dates.max().strftime("%Y-%m-%d")]}
    frame = _heartbeat(
        lambda: dai.query(sql, filters=filters).df(),
        4,
        9,
        "读取点时风格暴露",
        started_at,
        show_progress,
        current=",".join(normalized),
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame.drop_duplicates(["date", "instrument"])


def _validate_panel(frame, name, columns):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} 必须是 pandas.DataFrame。")
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{name} 缺少字段：{sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any() or result["instrument"].isna().any():
        raise ValueError(f"{name} 存在无效主键。")
    if result.duplicated(["date", "instrument"]).any():
        raise ValueError(f"{name} 存在重复 date + instrument。")
    return result


def _execution_price(frame, field):
    if field == "vwap":
        amount = pd.to_numeric(frame["amount"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        return amount / volume.where(volume > 0)
    return pd.to_numeric(frame[field], errors="coerce")


def _execution_state(frame, buy_field, sell_field):
    result = frame.copy()
    for column in ["volume", "upper_limit", "lower_limit", "is_risk_warning", "suspended"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["buy_price"] = _execution_price(result, buy_field)
    result["sell_price"] = _execution_price(result, sell_field)
    st = result["is_risk_warning"].fillna(1).ne(0)
    suspended = result["suspended"].fillna(1).ne(0) | result["volume"].fillna(0).le(0)
    result["can_buy"] = (
        ~st
        & ~suspended
        & result["buy_price"].gt(0)
        & result["upper_limit"].gt(0)
        & (result["buy_price"] < result["upper_limit"] - 1e-8)
    )
    result["can_sell"] = (
        ~suspended
        & result["sell_price"].gt(0)
        & result["lower_limit"].gt(0)
        & (result["sell_price"] > result["lower_limit"] + 1e-8)
    )

    def reasons(row, side):
        values = []
        if side == "buy" and bool(st.loc[row.name]):
            values.append("st_or_risk_warning")
        if bool(suspended.loc[row.name]):
            values.append("suspended_or_zero_volume")
        price = row[f"{side}_price"]
        if not np.isfinite(price) or price <= 0:
            values.append(f"invalid_{side}_price")
        if side == "buy" and (not np.isfinite(row["upper_limit"]) or row["upper_limit"] <= 0):
            values.append("invalid_upper_limit")
        elif side == "buy" and np.isfinite(price) and price >= row["upper_limit"] - 1e-8:
            values.append("at_or_above_upper_limit")
        if side == "sell" and (not np.isfinite(row["lower_limit"]) or row["lower_limit"] <= 0):
            values.append("invalid_lower_limit")
        elif side == "sell" and np.isfinite(price) and price <= row["lower_limit"] + 1e-8:
            values.append("at_or_below_lower_limit")
        return "|".join(values)

    result["buy_blocked_reason"] = result.apply(lambda row: reasons(row, "buy"), axis=1)
    result["sell_blocked_reason"] = result.apply(lambda row: reasons(row, "sell"), axis=1)
    return result


def _normalize_scores(values, method, clip):
    series = pd.to_numeric(values, errors="coerce")
    if method == "zscore":
        std = float(series.std(ddof=0))
        result = (series - series.mean()) / std if np.isfinite(std) and std > 0 else series * 0.0
    elif method == "rank":
        result = series.rank(method="average", pct=True) - 0.5
    else:
        raise ValueError("score_transform 仅支持 zscore 或 rank。")
    if clip is not None:
        limit = float(clip)
        if limit <= 0:
            raise ValueError("score_clip 必须大于 0 或为 None。")
        result = result.clip(-limit, limit)
    return result


def _normalize_weights(series, name="weights", exposure=1.0):
    result = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if result.isna().any() or (result < 0).any():
        raise ValueError(f"{name} 必须是有限非负数。")
    total = float(result.sum())
    if total <= 0:
        raise ValueError(f"{name} 合计必须大于 0。")
    return result / total * float(exposure)


def _benchmark_tilt(base, score, params):
    transform = str(params.get("score_transform", "zscore")).lower()
    transformed = _normalize_scores(score, transform, params.get("score_clip", 3.0))
    function = str(params.get("tilt_function", "exponential")).lower()
    if function == "exponential":
        multiplier = np.exp(float(params.get("tilt_strength", 0.5)) * transformed)
    elif function == "linear":
        multiplier = (1.0 + float(params.get("tilt_strength", 0.5)) * transformed).clip(lower=0.0)
    elif function == "bucket":
        count = _positive_int(params.get("bucket_count", 5), "bucket_count")
        multipliers = params.get("bucket_weight_multipliers")
        if not isinstance(multipliers, Iterable) or isinstance(multipliers, (str, bytes)):
            raise TypeError("bucket_weight_multipliers 必须是非负数序列。")
        multipliers = [float(value) for value in multipliers]
        if len(multipliers) != count or any(value < 0 for value in multipliers) or not any(value > 0 for value in multipliers):
            raise ValueError("bucket_weight_multipliers 长度必须等于 bucket_count，且非负并至少一个为正。")
        ranks = transformed.rank(method="first", pct=True)
        buckets = np.minimum((ranks * count).apply(math.ceil), count).clip(lower=1).astype(int)
        multiplier = buckets.map({index + 1: value for index, value in enumerate(multipliers)})
    elif function == "custom":
        custom = params.get("custom_tilt_function")
        if not callable(custom):
            raise TypeError("custom_tilt_function 必须是可调用函数。")
        multiplier = custom(transformed.copy(), **_mapping(params.get("custom_tilt_params"), "custom_tilt_params"))
        if not isinstance(multiplier, pd.Series) or not multiplier.index.equals(transformed.index):
            raise ValueError("自定义倾斜函数必须返回与输入索引完全相同的 pandas.Series。")
    else:
        raise ValueError("tilt_function 仅支持 exponential、linear、bucket、custom。")
    multiplier = pd.to_numeric(multiplier, errors="coerce")
    if multiplier.isna().any() or (multiplier < 0).any() or not (multiplier > 0).any():
        raise ValueError("倾斜乘数必须有限、非负且至少一个大于 0。")
    return _normalize_weights(base * multiplier, "倾斜后的原始权重")


def _stratified_sampling(base, score, state, params):
    if "industry_scheme" not in params:
        raise ValueError("stratified_sampling 必须显式提供 industry_scheme。")
    groups = _positive_int(params.get("market_cap_group_count", 5), "market_cap_group_count")
    mode = str(params.get("selection_mode", "top_fraction")).lower()
    if mode not in {"top_n", "top_fraction"}:
        raise ValueError("selection_mode 仅支持 top_n 或 top_fraction。")
    count = _positive_int(params.get("selection_count", 1), "selection_count") if mode == "top_n" else None
    fraction = float(params.get("selection_fraction", 0.2)) if mode == "top_fraction" else None
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("selection_fraction 必须满足 0 < value <= 1。")
    minimum = _positive_int(params.get("min_selected_per_bucket", 1), "min_selected_per_bucket")
    weight_method = str(params.get("within_bucket_weight_method", "benchmark_proportional")).lower()
    if weight_method not in {"benchmark_proportional", "equal", "score_proportional"}:
        raise ValueError("within_bucket_weight_method 无效。")
    if str(params.get("empty_bucket_policy", "benchmark_fallback")).lower() != "benchmark_fallback":
        raise ValueError("当前仅支持 empty_bucket_policy='benchmark_fallback'。")

    panel = state[
        ["instrument", "industry", "total_market_cap", "eligible"]
    ].copy()
    panel["base"] = panel["instrument"].map(base)
    panel["score"] = panel["instrument"].map(score)
    panel = panel.dropna(subset=["base", "industry", "total_market_cap"])
    panel = panel[panel["total_market_cap"] > 0]
    if panel.empty:
        raise ValueError("分层抽样没有可用股票。")
    result = pd.Series(0.0, index=base.index)
    for _, industry_group in panel.groupby("industry", sort=True):
        ranked = industry_group["total_market_cap"].rank(method="first", pct=True)
        industry_group = industry_group.assign(
            cap_bucket=np.minimum(np.ceil(ranked * groups), groups).astype(int)
        )
        for _, bucket in industry_group.groupby("cap_bucket", sort=True):
            bucket_weight = float(bucket["base"].sum())
            ordered = bucket[
                bucket["eligible"].fillna(False) & bucket["score"].notna()
            ].sort_values("score", ascending=False)
            wanted = count if mode == "top_n" else math.ceil(len(ordered) * fraction)
            wanted = min(len(ordered), max(minimum, wanted)) if len(ordered) else 0
            selected = ordered.head(wanted)
            fallback = selected.empty
            if fallback:
                selected = bucket
            if fallback or weight_method == "benchmark_proportional":
                local = _normalize_weights(selected.set_index("instrument")["base"], exposure=bucket_weight)
            elif weight_method == "equal":
                local = pd.Series(bucket_weight / len(selected), index=selected["instrument"])
            else:
                positive = _normalize_scores(selected.set_index("instrument")["score"], "rank", None) + 0.5
                local = _normalize_weights(positive.clip(lower=1e-12), exposure=bucket_weight)
            result.loc[local.index] += local
    return _normalize_weights(result, "分层抽样原始权重")


def _covariance(close_panel, signal_date, instruments, risk_model):
    lookback = _positive_int(risk_model.get("lookback_trading_days", 60), "risk lookback")
    minimum = _positive_int(risk_model.get("min_observations", 30), "risk min_observations")
    history = close_panel[(close_panel["date"] <= signal_date) & close_panel["instrument"].isin(instruments)]
    pivot = history.pivot(index="date", columns="instrument", values="close").sort_index().tail(lookback + 1)
    returns = pivot.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if len(returns) < minimum:
        raise ValueError("风险协方差历史观测不足。")
    method = str(risk_model.get("covariance_method", "exponential")).lower()
    if method == "historical":
        covariance = returns.cov(min_periods=minimum)
    elif method == "exponential":
        half_life = float(risk_model.get("half_life", 20.0))
        if half_life <= 0:
            raise ValueError("half_life 必须大于 0。")
        clean = returns.fillna(returns.mean()).fillna(0.0)
        ages = np.arange(len(clean) - 1, -1, -1)
        weights = np.power(0.5, ages / half_life)
        weights /= weights.sum()
        values = clean.to_numpy(float)
        mean = np.average(values, axis=0, weights=weights)
        centered = values - mean
        matrix = (centered * weights[:, None]).T @ centered
        covariance = pd.DataFrame(matrix, index=clean.columns, columns=clean.columns)
    else:
        raise ValueError("covariance_method 仅支持 historical 或 exponential。")
    covariance = covariance.reindex(index=instruments, columns=instruments)
    diagonal = np.diag(covariance.to_numpy(float))
    fallback = np.nanmedian(diagonal[np.isfinite(diagonal) & (diagonal > 0)])
    if not np.isfinite(fallback):
        fallback = 1e-6
    matrix = covariance.to_numpy(float)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    diagonal = np.diag(matrix).copy()
    diagonal[~np.isfinite(diagonal) | (diagonal <= 0)] = fallback
    np.fill_diagonal(matrix, diagonal)
    shrinkage = float(risk_model.get("shrinkage", 0.1))
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage 必须位于 [0, 1]。")
    matrix = (1 - shrinkage) * matrix + shrinkage * np.diag(np.diag(matrix))
    return matrix


def _constraint_defaults(value):
    defaults = {
        "target_stock_exposure": 1.0,
        "min_stock_exposure": 0.95,
        "max_stock_weight": None,
        "max_active_weight": None,
        "max_turnover": None,
        "industry_active_weight_limit": None,
        "style_active_exposure_limit": None,
        "max_tracking_error": None,
    }
    result = _mapping(value, "portfolio_constraints", defaults)
    for key in defaults:
        if result[key] is not None:
            result[key] = float(result[key])
    if not 0 < result["min_stock_exposure"] <= result["target_stock_exposure"] <= 1:
        raise ValueError("股票仓位必须满足 0 < min <= target <= 1。")
    for key in ["max_stock_weight", "max_active_weight", "max_turnover", "industry_active_weight_limit", "style_active_exposure_limit", "max_tracking_error"]:
        if result[key] is not None and result[key] <= 0:
            raise ValueError(f"{key} 必须大于 0 或为 None。")
    return result


def _feasibility_defaults(value):
    defaults = {"mode": "staged_relaxation", "stop_at_first_feasible": True, "final_action": "hold_previous"}
    result = _mapping(value, "feasibility_policy", defaults)
    if result["mode"] not in {"strict", "staged_relaxation"}:
        raise ValueError("feasibility_policy.mode 仅支持 strict 或 staged_relaxation。")
    if result["final_action"] != "hold_previous":
        raise ValueError("当前仅支持 final_action='hold_previous'。")
    return result


def _solve_weights(raw, base, previous, industry, style, covariance, constraints, objective_spec):
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("当前组合约束/优化模式需要 BigQuant 环境提供 scipy.optimize。") from exc

    instruments = list(raw.index)
    wb = base.reindex(instruments).fillna(0.0).to_numpy(float)
    wr = raw.reindex(instruments).fillna(0.0).to_numpy(float)
    wp = previous.reindex(instruments).fillna(0.0).to_numpy(float)
    exposure = float(constraints["target_stock_exposure"])
    initial = _normalize_weights(pd.Series(wr, index=instruments), exposure=exposure).to_numpy(float)

    bounds = []
    for index in range(len(instruments)):
        upper = 1.0 if constraints["max_stock_weight"] is None else constraints["max_stock_weight"]
        if constraints["max_active_weight"] is not None:
            upper = min(upper, wb[index] + constraints["max_active_weight"])
            lower = max(0.0, wb[index] - constraints["max_active_weight"])
        else:
            lower = 0.0
        if lower > upper + 1e-12:
            return None, "empty_bound"
        bounds.append((lower, upper))

    scipy_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - exposure}]
    if constraints["max_turnover"] is not None:
        limit = float(constraints["max_turnover"])
        scipy_constraints.append({"type": "ineq", "fun": lambda w, limit=limit: limit - 0.5 * np.abs(w - wp).sum()})
    if industry is not None and constraints["industry_active_weight_limit"] is not None:
        matrix = industry.reindex(index=instruments).fillna(0.0).to_numpy(float)
        limit = float(constraints["industry_active_weight_limit"])
        scipy_constraints.extend([
            {"type": "ineq", "fun": lambda w, m=matrix, l=limit: l - m.T @ (w - wb)},
            {"type": "ineq", "fun": lambda w, m=matrix, l=limit: l + m.T @ (w - wb)},
        ])
    if style is not None and constraints["style_active_exposure_limit"] is not None:
        matrix = style.reindex(index=instruments).fillna(0.0).to_numpy(float)
        limit = float(constraints["style_active_exposure_limit"])
        scipy_constraints.extend([
            {"type": "ineq", "fun": lambda w, m=matrix, l=limit: l - m.T @ (w - wb)},
            {"type": "ineq", "fun": lambda w, m=matrix, l=limit: l + m.T @ (w - wb)},
        ])
    if covariance is not None and constraints["max_tracking_error"] is not None:
        limit = float(constraints["max_tracking_error"])
        scipy_constraints.append({"type": "ineq", "fun": lambda w, c=covariance, l=limit: l - math.sqrt(max(0.0, float((w - wb) @ c @ (w - wb)) * 252.0))})

    if objective_spec["type"] == "optimization":
        alpha = objective_spec["alpha"].reindex(instruments).fillna(0.0).to_numpy(float)
        risk_aversion = float(objective_spec.get("risk_aversion", 1.0))
        cost_aversion = float(objective_spec.get("transaction_cost_aversion", 0.0))

        def objective(w):
            delta = w - wb
            risk = float(delta @ covariance @ delta) if covariance is not None else 0.0
            return -(alpha @ w) + risk_aversion * risk + cost_aversion * np.abs(w - wp).sum()
    else:
        objective = lambda w: float(np.square(w - wr).sum())

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        options={"ftol": float(objective_spec.get("solver_tolerance", 1e-9)), "maxiter": int(objective_spec.get("solver_max_iterations", 1000)), "disp": False},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return None, str(result.message)
    weights = pd.Series(np.clip(result.x, 0.0, None), index=instruments)
    if abs(weights.sum() - exposure) > 1e-5:
        return None, "exposure_validation_failed"
    return weights, "ok"


def _feasible_target(raw, base, previous, industry, style, covariance, constraints, objective_spec, policy):
    attempts = []
    current = dict(constraints)

    optional_constraints = [
        "max_stock_weight",
        "max_active_weight",
        "max_turnover",
        "industry_active_weight_limit",
        "style_active_exposure_limit",
        "max_tracking_error",
    ]
    if (
        objective_spec["type"] == "projection"
        and all(current[name] is None for name in optional_constraints)
    ):
        direct = _normalize_weights(
            raw,
            "无附加组合约束的目标权重",
            exposure=current["target_stock_exposure"],
        )
        attempts.append(
            {
                "attempt": 1,
                "step": "direct_without_numeric_solver",
                "constraints": dict(current),
                "feasible": True,
                "solver_message": "not_required",
            }
        )
        return direct, attempts

    def attempt(label):
        target, message = _solve_weights(raw, base, previous, industry, style, covariance, current, objective_spec)
        attempts.append({"attempt": len(attempts) + 1, "step": label, "constraints": dict(current), "feasible": target is not None, "solver_message": message})
        return target

    target = attempt("strict")
    if target is not None or policy["mode"] == "strict":
        return target, attempts

    steps = [
        ("max_turnover", 0.05, min(1.0, constraints["max_turnover"] + 0.20) if constraints["max_turnover"] is not None else None),
        ("max_active_weight", 0.005, constraints["max_active_weight"] + 0.02 if constraints["max_active_weight"] is not None else None),
        ("industry_active_weight_limit", 0.01, constraints["industry_active_weight_limit"] + 0.03 if constraints["industry_active_weight_limit"] is not None else None),
        ("style_active_exposure_limit", 0.1, constraints["style_active_exposure_limit"] + 0.3 if constraints["style_active_exposure_limit"] is not None else None),
        ("max_tracking_error", 0.005, constraints["max_tracking_error"] + 0.02 if constraints["max_tracking_error"] is not None else None),
    ]
    for field, increment, cap in steps:
        if current[field] is None:
            continue
        while current[field] + 1e-12 < cap:
            old = current[field]
            current[field] = min(cap, current[field] + increment)
            target = attempt(f"relax {field}: {old:.6g} -> {current[field]:.6g}")
            if target is not None and policy["stop_at_first_feasible"]:
                return target, attempts

    while current["target_stock_exposure"] - 1e-12 > current["min_stock_exposure"]:
        old = current["target_stock_exposure"]
        current["target_stock_exposure"] = max(current["min_stock_exposure"], old - 0.01)
        target = attempt(f"allow_cash: {old:.2%} -> {current['target_stock_exposure']:.2%}")
        if target is not None and policy["stop_at_first_feasible"]:
            return target, attempts
    return target, attempts


def _position_quantity(position):
    for name in ("current_qty", "amount", "quantity"):
        value = getattr(position, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _position_price(position):
    for name in ("last_price", "last_sale_price", "market_price", "cost_price"):
        value = getattr(position, name, None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    return np.nan


def _current_weights(context):
    positions = context.get_account_positions()
    value = getattr(context.portfolio, "portfolio_value", np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan
    result = {}
    if np.isfinite(value) and value > 0:
        for instrument, position in positions.items():
            quantity = _position_quantity(position)
            price = _position_price(position)
            if quantity > 0 and np.isfinite(price):
                result[instrument] = quantity * price / value
    return positions, pd.Series(result, dtype=float), value


def _attribute(obj, names, default=None):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _drift_metrics(current, target, include_cash):
    index = current.index.union(target.index)
    actual = current.reindex(index).fillna(0.0)
    desired = target.reindex(index).fillna(0.0)
    delta = actual - desired
    max_drift = float(delta.abs().max()) if len(delta) else 0.0
    active_share = 0.5 * float(delta.abs().sum())
    if include_cash:
        active_share += 0.5 * abs((1.0 - actual.sum()) - (1.0 - desired.sum()))
    return {"max_abs_weight_drift": max_drift, "active_share": active_share}


def _trigger_from_thresholds(metrics, thresholds, logic):
    tests = {key: metrics.get(key, -np.inf) >= float(value) for key, value in thresholds.items()}
    return (any(tests.values()) if logic == "any" else all(tests.values())), tests


def run_index_enhancement_backtest(
    start_date,
    end_date,
    reference_portfolio,
    factor_name,
    factor_params=None,
    signal_direction=1,
    construction_method="benchmark_tilt",
    construction_params=None,
    rebalance_rule=None,
    portfolio_constraints=None,
    risk_model=None,
    feasibility_policy=None,
    execution_config=None,
    trading_costs=None,
    initial_cash=1_000_000,
    performance_benchmark=None,
    show_progress=True,
    progress_every=20,
):
    """运行 BigQuant 指数增强历史回测。

    核心参数
    --------
    reference_portfolio : dict
        指数模式：``{"type":"index", "index_code":"000300.SH"}``。
        自定义模式：``{"type":"custom", "instruments":[...],
        "base_weight_method":"equal|market_cap|explicit", "weights":{...}}``。
    factor_name, factor_params : str, dict
        因子中心名称和因子内部参数。策略通过 loader 自动准备原始数据。
    signal_direction : 1 或 -1
        仅控制因子值到组合偏好的映射。1 表示值越大越偏好，-1 相反。
    construction_method : str
        benchmark_tilt、stratified_sampling 或 constrained_optimization。
    construction_params : dict
        倾斜法可配置 score_transform、score_clip、tilt_function、
        tilt_strength；bucket 模式另需 bucket_count 与非负乘数；custom
        模式传 custom_tilt_function 和 custom_tilt_params。
        分层抽样必须显式传 industry_scheme，并配置市值组数、选股方式和
        桶内权重方式。约束优化可配置 alpha_transform、alpha_scale、
        risk_aversion、transaction_cost_aversion 和求解器精度。
    rebalance_rule : dict
        fixed_interval、custom_dates、weight_drift、risk_drift 或 hybrid。
        固定/指定日期不使用冷静期；偏离触发使用
        min_interval_trading_days。指数成分调整由
        rebalance_on_index_reconstitution（默认 True）独立触发。
    portfolio_constraints : dict
        target/min_stock_exposure，以及可选的单股权重、主动权重、换手、
        行业偏离、风格偏离和跟踪误差上限。值为 None 表示不施加该策略
        约束，但做多、停牌、涨跌停和成交量等真实交易约束不会被关闭。
    risk_model : dict
        风险协方差方法、回看交易日、半衰期、最少观测、收缩强度和
        style_fields。优化、跟踪误差约束或风险偏离触发时使用。
    feasibility_policy : dict
        strict 或 staged_relaxation。默认依次放宽换手、单股主动偏离、
        行业、风格、跟踪误差，再允许股票仓位降至 min_stock_exposure；
        仍无解则保持上期组合，首次建仓无上期组合时保持现金。
    execution_config : dict
        order_price_field_buy/sell（open、close、vwap）、volume_limit、
        slippage_value、weight_tolerance、rebalance_on_index_reconstitution。

    时序口径
    --------
    回测区间第一个交易日必须建仓；信号使用其前一交易日收盘后可得数据，
    并在执行日由 BigTrader 撮合。后续同样是 t 日收盘信号、t+1 日执行。
    执行日 ST、停牌、零成交量、涨停买入和跌停卖出会阻止相应订单；不会
    使用替补股票重新分配权重。BigTrader 缓存固定关闭。

    返回
    ----
    dict
        performance、schedule、trigger_audit、factor_signals、raw_weights、
        target_weights、actual_weights、active_weights、feasibility_audit、
        risk_audit、rebalance_audit、execution_audit、order_audit、
        trade_audit、data_diagnostics、resolved_config。
    """
    started_at = time.perf_counter()
    start_date = _timestamp(start_date, "start_date")
    end_date = _timestamp(end_date, "end_date")
    if start_date > end_date:
        raise ValueError("start_date 不能晚于 end_date。")
    if not isinstance(factor_name, str) or not factor_name.strip():
        raise ValueError("factor_name 必须是非空字符串。")
    factor_name = factor_name.strip()
    factor_params = _factor_params(factor_params)
    if signal_direction not in (-1, 1):
        raise ValueError("signal_direction 只能是 1 或 -1。")
    method = str(construction_method).strip().lower()
    if method not in {"benchmark_tilt", "stratified_sampling", "constrained_optimization"}:
        raise ValueError("construction_method 无效。")
    construction = _mapping(construction_params, "construction_params")
    if method == "constrained_optimization":
        solver = str(construction.get("solver", "auto")).strip().lower()
        if solver not in {"auto", "slsqp"}:
            raise ValueError(
                "当前约束优化实现的 solver 仅支持 'auto' 或 'slsqp'。"
            )
        construction["solver"] = solver
    reference = _normalize_reference(reference_portfolio)
    rule = _normalize_rebalance_rule(rebalance_rule)
    constraints = _constraint_defaults(portfolio_constraints)
    risk = _mapping(risk_model, "risk_model", {
        "covariance_method": "exponential",
        "lookback_trading_days": 60,
        "half_life": 20.0,
        "min_observations": 30,
        "shrinkage": 0.1,
        "style_fields": [],
    })
    policy = _feasibility_defaults(feasibility_policy)
    execution = _mapping(execution_config, "execution_config", {
        "order_price_field_buy": "open",
        "order_price_field_sell": "open",
        "volume_limit": 0.025,
        "slippage_value": 0.001,
        "weight_tolerance": 1e-4,
        "rebalance_on_index_reconstitution": True,
    })
    buy_field = str(execution["order_price_field_buy"]).lower()
    sell_field = str(execution["order_price_field_sell"]).lower()
    if buy_field not in _PRICE_FIELDS or sell_field not in _PRICE_FIELDS:
        raise ValueError("订单价格字段仅支持 open、close、vwap。")
    volume_limit = float(execution["volume_limit"])
    if not 0 < volume_limit <= 1:
        raise ValueError("volume_limit 必须满足 0 < value <= 1。")
    weight_tolerance = float(execution["weight_tolerance"])
    if weight_tolerance < 0:
        raise ValueError("weight_tolerance 不能为负。")
    progress_every = _positive_int(progress_every, "progress_every")
    initial_cash = float(initial_cash)
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash 必须大于 0。")
    costs = _mapping(trading_costs, "trading_costs", {"buy_cost": 0.0003, "sell_cost": 0.0003, "min_cost": 5.0, "tax_ratio": 0.0005})
    for key in ["buy_cost", "sell_cost", "min_cost", "tax_ratio"]:
        costs[key] = float(costs[key])
        if costs[key] < 0:
            raise ValueError(f"trading_costs[{key!r}] 不能为负。")

    from factor_lib.common.data_adapters.bigquant_adapters.daily import load_daily_raw_data
    from factor_lib.common.data_adapters.bigquant_adapters.loader import get_factor_data_requirements, get_factor_metadata, load_factor_raw_data
    from factor_lib.factor_hub.get_factor import get_factor

    if show_progress:
        _progress(1, 9, "读取交易日历并建立候选执行日", started_at, current=f"{start_date:%Y-%m-%d}至{end_date:%Y-%m-%d}")
    calendar = _query_calendar(end_date, show_progress, started_at)
    all_pairs = _execution_pairs(calendar, start_date, end_date)
    all_signal_dates = pd.DatetimeIndex(all_pairs["signal_date"])
    all_execution_dates = pd.DatetimeIndex(all_pairs["execution_date"])

    index_scale_audit = pd.DataFrame()
    if reference["type"] == "index":
        reference_panel, index_scale_audit = _query_index_weights(reference["index_code"], all_signal_dates, show_progress, started_at)
        reference_instruments = sorted(reference_panel["instrument"].unique())
        change_dates = _membership_change_dates(reference_panel) if execution["rebalance_on_index_reconstitution"] else pd.DatetimeIndex([])
    else:
        reference_instruments = reference["instruments"]
        reference_panel = None
        change_dates = pd.DatetimeIndex([])

    scheduled_exec = _scheduled_execution_dates(rule, all_execution_dates)
    first_execution = all_execution_dates[0]
    scheduled_exec = scheduled_exec.union(pd.DatetimeIndex([first_execution]))
    if _uses_deviation(rule):
        possible_exec = all_execution_dates
    else:
        possible_exec = scheduled_exec
        if len(change_dates):
            change_exec = all_pairs.loc[all_pairs["signal_date"].isin(change_dates), "execution_date"]
            possible_exec = possible_exec.union(pd.DatetimeIndex(change_exec))
    possible_pairs = all_pairs[all_pairs["execution_date"].isin(possible_exec)].copy()
    possible_signal_dates = pd.DatetimeIndex(possible_pairs["signal_date"])
    if show_progress:
        _progress(2, 9, "参考组合与触发日准备完成", started_at, completed=1, total=1, detail=f"{len(possible_pairs)}个潜在信号日，{len(reference_instruments)}只参考股票")

    metadata = get_factor_metadata(factor_name)
    requirements = get_factor_data_requirements(factor_name, factor_params)
    resolved_factor_params = {key: value for key, value in requirements["resolved_factor_params"].items() if key not in _RESERVED_FACTOR_PARAMS}
    factor_column = _resolve_factor_column(metadata, factor_name)
    history_days = _history_days(requirements)
    windows, factor_dates = _factor_windows(possible_signal_dates, calendar, history_days)
    if show_progress:
        _progress(3, 9, "预存因子原始数据", started_at, completed=0, total=1, current=factor_name, detail=f"{len(factor_dates)}个日期")
    factor_bundle = load_factor_raw_data(factor_name=factor_name, dates=factor_dates, factor_params=resolved_factor_params, instruments=reference_instruments, show_progress=show_progress)
    security_daily = _validate_panel(factor_bundle.get_security_daily(), "factor security_daily", ["date", "instrument"])
    factor_bundle = factor_bundle.with_domain("security_daily", security_daily, key_columns=("date", "instrument"))

    needs_industry = method == "stratified_sampling" or constraints["industry_active_weight_limit"] is not None
    deviation = _deviation_rule(rule) if _uses_deviation(rule) else None
    if deviation and deviation["type"] == "risk_drift" and "max_industry_active_weight" in deviation.get("thresholds", {}):
        needs_industry = True
    industry_scheme = construction.get("industry_scheme") or risk.get("industry_scheme")
    if needs_industry and not industry_scheme:
        raise ValueError("当前构建/约束需要行业数据，必须显式提供 industry_scheme。")
    style_fields = [str(value).upper() for value in risk.get("style_fields", [])]
    needs_style = constraints["style_active_exposure_limit"] is not None or (deviation and deviation["type"] == "risk_drift" and "max_style_active_exposure" in deviation.get("thresholds", {}))
    if needs_style and not style_fields:
        raise ValueError("风格约束或风险偏离触发需要 risk_model.style_fields。")
    needs_covariance = method == "constrained_optimization" or constraints["max_tracking_error"] is not None or (deviation and deviation["type"] == "risk_drift" and "predicted_tracking_error" in deviation.get("thresholds", {}))

    signal_fields = ["total_market_cap", "is_risk_warning", "suspended", "volume", "list_date"]
    signal_panel = load_daily_raw_data(standard_fields=signal_fields, dates=possible_signal_dates, instruments=reference_instruments, show_progress=show_progress)
    signal_panel = _validate_panel(signal_panel, "signal_panel", ["date", "instrument", *signal_fields])
    for column in ["total_market_cap", "is_risk_warning", "suspended", "volume"]:
        signal_panel[column] = pd.to_numeric(signal_panel[column], errors="coerce")
    signal_panel["eligible"] = signal_panel["is_risk_warning"].fillna(1).eq(0) & signal_panel["suspended"].fillna(1).eq(0) & signal_panel["volume"].fillna(0).gt(0)
    first_signal = possible_signal_dates.min()
    first_custom = signal_panel[signal_panel["date"] == first_signal]
    if reference["type"] == "custom":
        missing_listed = sorted(set(reference_instruments) - set(first_custom["instrument"]))
        invalid_listing = first_custom[pd.to_datetime(first_custom["list_date"], errors="coerce") > start_date]["instrument"].tolist()
        if missing_listed or invalid_listing:
            raise ValueError(f"自定义股票必须在回测开始日前全部上市；缺失={missing_listed[:5]}，晚于开始日上市={invalid_listing[:5]}")

    industry_panel = _query_industry(possible_signal_dates, reference_instruments, industry_scheme, show_progress, started_at) if needs_industry else pd.DataFrame(columns=["date", "instrument", "industry"])
    style_panel = _query_style(possible_signal_dates, reference_instruments, style_fields, show_progress, started_at) if needs_style else pd.DataFrame(columns=["date", "instrument", *style_fields])

    risk_close = None
    if needs_covariance:
        risk_lookback = _positive_int(risk.get("lookback_trading_days", 60), "risk lookback")
        positions = {date: index for index, date in enumerate(calendar)}
        earliest = positions[possible_signal_dates.min()] - risk_lookback - 1
        if earliest < 0:
            raise ValueError("风险模型历史窗口不足。")
        risk_dates = calendar[earliest : positions[possible_signal_dates.max()] + 1]
        risk_close = load_daily_raw_data(standard_fields=["close"], dates=risk_dates, instruments=reference_instruments, show_progress=show_progress)
        risk_close = _validate_panel(risk_close, "risk_close", ["date", "instrument", "close"])
        risk_close["close"] = pd.to_numeric(risk_close["close"], errors="coerce")

    execution_fields = ["volume", "upper_limit", "lower_limit", "is_risk_warning", "suspended"]
    for field in (buy_field, sell_field):
        execution_fields.extend(["amount", "volume"] if field == "vwap" else [field])
    execution_fields = list(dict.fromkeys(execution_fields))
    execution_panel = load_daily_raw_data(standard_fields=execution_fields, dates=pd.DatetimeIndex(possible_pairs["execution_date"]), instruments=reference_instruments, show_progress=show_progress)
    execution_panel = _execution_state(_validate_panel(execution_panel, "execution_panel", ["date", "instrument", *execution_fields]), buy_field, sell_field)
    execution_map = {(row.date, row.instrument): row._asdict() for row in execution_panel.itertuples(index=False)}

    signal_by_date = {date: group.copy() for date, group in signal_panel.groupby("date", sort=False)}
    industry_by_date = {date: group.set_index("instrument")["industry"] for date, group in industry_panel.groupby("date", sort=False)}
    style_by_date = {date: group.set_index("instrument")[style_fields] for date, group in style_panel.groupby("date", sort=False)}
    reference_by_date = {}
    if reference["type"] == "index":
        if reference_panel is None:
            raise RuntimeError(
                "指数参考组合已经通过参数校验，但历史指数权重面板未建立。"
            )
        reference_by_date = {
            date: group.set_index("instrument")["weight"]
            for date, group in reference_panel.groupby("date", sort=False)
        }
    else:
        for date, state in signal_by_date.items():
            if reference["base_weight_method"] == "equal":
                base = pd.Series(1.0 / len(reference_instruments), index=reference_instruments)
            elif reference["base_weight_method"] == "explicit":
                base = pd.Series(reference["weights"], dtype=float)
            else:
                caps = state.set_index("instrument")["total_market_cap"].reindex(reference_instruments)
                if caps.isna().any() or (caps <= 0).any():
                    raise ValueError(f"{date:%Y-%m-%d} 自定义市值权重缺少有效总市值。")
                base = _normalize_weights(caps)
            reference_by_date[date] = base

    pair_by_signal = {row.signal_date: row for row in possible_pairs.itertuples(index=False)}
    scheduled_set = set(scheduled_exec)
    change_set = set(change_dates)
    engine_instruments = reference_instruments
    factor_records, raw_records, target_records, actual_records, active_records = [], [], [], [], []
    trigger_records, feasibility_records, risk_records, rebalance_records = [], [], [], []
    execution_records, order_records, trade_records = [], [], []
    last_target = pd.Series(dtype=float)
    last_rebalance_position = None
    processed = 0
    successful = 0

    if show_progress:
        _progress(5, 9, "数据预存与快速索引完成", started_at, completed=1, total=1, detail=f"因子股票域{len(security_daily):,}行，执行约束{len(execution_panel):,}行")
        _progress(6, 9, "启动 BigTrader 原生回测", started_at, completed=0, total=len(possible_pairs), detail=f"{len(possible_pairs)}个潜在信号日，{len(engine_instruments)}只股票")
        print()

    from bigmodule import M

    def initialize(context):
        from bigtrader.finance.commission import PerOrder
        context.set_commission(PerOrder(**costs))
        slippage = execution.get("slippage_value")
        if slippage is not None:
            context.set_slippage_value(slippage_type=2, slippage_value=float(slippage))

    def handle_data(context, data):
        nonlocal last_target, last_rebalance_position, processed, successful
        signal_date = pd.Timestamp(data.current_dt).normalize()
        pair = pair_by_signal.get(signal_date)
        if pair is None:
            return
        processed += 1
        execution_date = pair.execution_date
        _, current, portfolio_value = _current_weights(context)
        base = reference_by_date.get(signal_date)
        if base is None:
            raise ValueError(f"{signal_date:%Y-%m-%d} 缺少参考权重。")
        current = current.reindex(current.index.union(base.index)).fillna(0.0)
        is_first = last_rebalance_position is None
        scheduled = execution_date in scheduled_set
        reconstitution = signal_date in change_set
        deviation_triggered = False
        trigger_tests = {}
        metrics = {}
        cooldown_ok = True
        if _uses_deviation(rule) and not is_first and not last_target.empty:
            d_rule = _deviation_rule(rule)
            cooldown_ok = pair.execution_position - last_rebalance_position >= d_rule["min_interval_trading_days"]
            if d_rule["type"] == "weight_drift":
                metrics = _drift_metrics(current, last_target, d_rule.get("include_cash", True))
                deviation_triggered, trigger_tests = _trigger_from_thresholds(metrics, d_rule["thresholds"], d_rule["trigger_logic"])
            else:
                members = list(base.index)
                covariance = None
                if "predicted_tracking_error" in d_rule.get("thresholds", {}):
                    if risk_close is None:
                        raise RuntimeError("风险偏离触发缺少用于计算协方差的收盘价数据。")
                    if show_progress:
                        _progress(
                            6,
                            9,
                            "回测中：计算风险偏离协方差",
                            started_at,
                            completed=processed - 1,
                            total=len(possible_pairs),
                            current=f"{signal_date:%Y-%m-%d}",
                            detail=f"{len(members)}只股票",
                        )
                    covariance = _heartbeat(
                        lambda: _covariance(
                            risk_close,
                            signal_date,
                            members,
                            risk,
                        ),
                        6,
                        9,
                        "回测中：计算风险偏离协方差",
                        started_at,
                        show_progress,
                        completed=processed - 1,
                        total=len(possible_pairs),
                        current=f"{signal_date:%Y-%m-%d}",
                        detail=f"{len(members)}只股票",
                    )
                delta = current.reindex(members).fillna(0.0) - base
                if covariance is not None:
                    metrics["predicted_tracking_error"] = math.sqrt(max(0.0, float(delta.to_numpy() @ covariance @ delta.to_numpy()) * 252.0))
                industry = industry_by_date.get(signal_date)
                if industry is not None:
                    exposure = pd.get_dummies(industry.reindex(members).fillna("unknown"), dtype=float)
                    metrics["max_industry_active_weight"] = float(np.abs(exposure.to_numpy().T @ delta.to_numpy()).max())
                style = style_by_date.get(signal_date)
                if style is not None and len(style.columns):
                    metrics["max_style_active_exposure"] = float(np.abs(style.reindex(members).fillna(0.0).to_numpy().T @ delta.to_numpy()).max())
                deviation_triggered, trigger_tests = _trigger_from_thresholds(metrics, d_rule.get("thresholds", {}), d_rule["trigger_logic"])
            deviation_triggered = deviation_triggered and cooldown_ok
        should_rebalance = is_first or scheduled or reconstitution or deviation_triggered
        reasons = [name for name, flag in [("initial", is_first), ("scheduled", scheduled), ("index_reconstitution", reconstitution), ("deviation", deviation_triggered)] if flag]
        trigger_records.append({"signal_date": signal_date, "execution_date": execution_date, "triggered": should_rebalance, "trigger_reasons": "|".join(reasons), "cooldown_ok": cooldown_ok, **metrics, "threshold_tests": trigger_tests})
        if not should_rebalance:
            if show_progress and (processed == 1 or processed % progress_every == 0 or processed == len(possible_pairs)):
                _progress(6, 9, "回测中：检查调仓触发", started_at, completed=processed, total=len(possible_pairs), current=f"{signal_date:%Y-%m-%d}", detail="未触发")
            return

        status, error = "ok", ""
        target = None
        try:
            if show_progress:
                _progress(6, 9, "回测中：计算因子信号", started_at, completed=processed - 1, total=len(possible_pairs), current=f"{signal_date:%Y-%m-%d} {factor_name}")
            required_dates = windows[signal_date]
            factor_input = factor_bundle.select_dates(required_dates)
            factor_frame = get_factor(factor_name, factor_input, target_dates=[signal_date], as_of_date=signal_date, show_progress=show_progress, progress_every=progress_every, **resolved_factor_params)
            factor_frame = _validate_panel(factor_frame, f"{factor_name}@{signal_date:%Y-%m-%d}", ["date", "instrument", factor_column])
            state = signal_by_date[signal_date]
            panel = base.rename("base_weight").to_frame().reset_index().rename(columns={"index": "instrument"})
            panel = panel.merge(factor_frame[["instrument", factor_column]], on="instrument", how="left", validate="one_to_one")
            panel = panel.merge(state[["instrument", "eligible", "total_market_cap"]], on="instrument", how="left", validate="one_to_one")
            if needs_industry:
                panel["industry"] = panel["instrument"].map(industry_by_date.get(signal_date))
            panel["factor_value"] = pd.to_numeric(panel[factor_column], errors="coerce")
            panel["oriented_score"] = signal_direction * panel["factor_value"]
            panel["eligible"] = panel["eligible"].fillna(False)
            eligible = panel[panel["eligible"] & panel["oriented_score"].notna()].copy()
            if eligible.empty:
                raise ValueError("信号日没有可用于构建组合的股票。")
            eligible_base = base.reindex(eligible["instrument"]).fillna(0.0)
            eligible_base.index = eligible["instrument"].values
            eligible_base = _normalize_weights(eligible_base)
            score = eligible.set_index("instrument")["oriented_score"]
            if method == "benchmark_tilt":
                raw = _benchmark_tilt(eligible_base, score, construction)
                raw = raw.reindex(base.index).fillna(0.0)
                objective_spec = {"type": "projection"}
            elif method == "stratified_sampling":
                raw = _stratified_sampling(base, score, panel, construction)
                objective_spec = {"type": "projection"}
            else:
                alpha = _normalize_scores(score, construction.get("alpha_transform", "zscore"), construction.get("score_clip", 3.0)) * float(construction.get("alpha_scale", 1.0))
                raw = _benchmark_tilt(eligible_base, alpha, {"score_transform": "zscore", "score_clip": construction.get("score_clip", 3.0), "tilt_function": "exponential", "tilt_strength": construction.get("initial_tilt_strength", 0.25)})
                raw = raw.reindex(base.index).fillna(0.0)
                objective_spec = {"type": "optimization", "alpha": alpha, "risk_aversion": construction.get("risk_aversion", 1.0), "transaction_cost_aversion": construction.get("transaction_cost_aversion", 0.0), "solver_tolerance": construction.get("solver_tolerance", 1e-9), "solver_max_iterations": construction.get("solver_max_iterations", 1000)}
            members = list(base.index)
            covariance = None
            if needs_covariance:
                if risk_close is None:
                    raise RuntimeError("组合约束需要协方差矩阵，但未加载收盘价数据。")
                if show_progress:
                    _progress(
                        6,
                        9,
                        "回测中：计算目标组合风险协方差",
                        started_at,
                        completed=processed - 1,
                        total=len(possible_pairs),
                        current=f"{signal_date:%Y-%m-%d}",
                        detail=f"{len(members)}只股票",
                    )
                covariance = _heartbeat(
                    lambda: _covariance(
                        risk_close,
                        signal_date,
                        members,
                        risk,
                    ),
                    6,
                    9,
                    "回测中：计算目标组合风险协方差",
                    started_at,
                    show_progress,
                    completed=processed - 1,
                    total=len(possible_pairs),
                    current=f"{signal_date:%Y-%m-%d}",
                    detail=f"{len(members)}只股票",
                )
            industry_matrix = None
            if needs_industry:
                industries = industry_by_date[signal_date].reindex(members).fillna("unknown")
                industry_matrix = pd.get_dummies(industries, dtype=float)
            style_matrix = None
            if needs_style:
                style_for_date = style_by_date.get(signal_date)
                if style_for_date is None:
                    raise ValueError(
                        f"{signal_date:%Y-%m-%d} 缺少所需的点时风格暴露。"
                    )
                style_matrix = style_for_date.reindex(members)
            previous = current.reindex(members).fillna(0.0)
            if show_progress:
                _progress(
                    6,
                    9,
                    "回测中：构建并校验目标组合",
                    started_at,
                    completed=processed - 1,
                    total=len(possible_pairs),
                    current=f"{signal_date:%Y-%m-%d}",
                    detail=f"{method}，{len(members)}只股票",
                )
            target, attempts = _heartbeat(
                lambda: _feasible_target(
                    raw,
                    base.reindex(members).fillna(0.0),
                    previous,
                    industry_matrix,
                    style_matrix,
                    covariance,
                    constraints,
                    objective_spec,
                    policy,
                ),
                6,
                9,
                "回测中：构建并校验目标组合",
                started_at,
                show_progress,
                completed=processed - 1,
                total=len(possible_pairs),
                current=f"{signal_date:%Y-%m-%d}",
                detail=f"{method}，{len(members)}只股票",
            )
            for attempt in attempts:
                feasibility_records.append({"signal_date": signal_date, "execution_date": execution_date, **attempt})
            if target is None:
                status = "infeasible_hold_previous"
                target = last_target.copy() if not last_target.empty else pd.Series(dtype=float)
            else:
                successful += 1
            panel_records = panel[
                ["instrument", "factor_value", "oriented_score", "eligible"]
            ].copy()
            panel_records["signal_date"] = signal_date
            panel_records["execution_date"] = execution_date
            factor_records.extend(panel_records.to_dict("records"))
            raw_records.extend({"signal_date": signal_date, "execution_date": execution_date, "instrument": key, "raw_weight": value} for key, value in raw.items())
            target_records.extend({"signal_date": signal_date, "execution_date": execution_date, "instrument": key, "target_weight": value} for key, value in target.items())
            active_index = target.index.union(base.index)
            active = target.reindex(active_index).fillna(0.0) - base.reindex(active_index).fillna(0.0)
            active_records.extend({"signal_date": signal_date, "execution_date": execution_date, "instrument": key, "active_weight": value} for key, value in active.items())
            if covariance is not None:
                delta = target.reindex(members).fillna(0.0) - base.reindex(members).fillna(0.0)
                risk_records.append({"signal_date": signal_date, "execution_date": execution_date, "predicted_tracking_error": math.sqrt(max(0.0, float(delta.to_numpy() @ covariance @ delta.to_numpy()) * 252.0))})
        except Exception as exc:
            status = "skipped_error"
            error = f"{type(exc).__name__}: {exc}"
            target = None

        if target is None:
            rebalance_records.append({"signal_date": signal_date, "execution_date": execution_date, "status": status, "error_message": error, "target_count": 0})
            return
        last_target = target.copy()
        last_rebalance_position = pair.execution_position
        all_codes = current.index.union(target.index)
        sells, buys = [], []
        for instrument in all_codes:
            current_weight = float(current.get(instrument, 0.0))
            target_weight = float(target.get(instrument, 0.0))
            delta = target_weight - current_weight
            if abs(delta) <= weight_tolerance:
                continue
            (sells if delta < 0 else buys).append((instrument, current_weight, target_weight))
        intents = sells + buys
        for intent_number, (instrument, current_weight, target_weight) in enumerate(intents, start=1):
            side = "sell" if target_weight < current_weight else "buy"
            state = execution_map.get((execution_date, instrument))
            tradable = bool(state and state[f"can_{side}"])
            blocked = "missing_execution_data" if state is None else state[f"{side}_blocked_reason"]
            submit_result = None
            submitted = False
            if tradable:
                submit_result = context.order_target_percent(instrument, target_weight)
                try:
                    submitted = int(submit_result) >= 0
                except (TypeError, ValueError):
                    submitted = submit_result is not None
                if not submitted:
                    blocked = f"order_submit_failed:{submit_result}"
            execution_records.append({"signal_date": signal_date, "execution_date": execution_date, "instrument": instrument, "current_weight": current_weight, "target_weight": target_weight, "order_direction": side, "tradable": tradable, "blocked_reason": blocked, "order_attempted": tradable, "order_submitted": submitted, "submit_result": submit_result})
            if show_progress:
                _progress(6, 9, "回测中：提交调仓订单", started_at, completed=processed, total=len(possible_pairs), current=f"{signal_date:%Y-%m-%d}", detail=f"订单{intent_number}/{len(intents)} {side}:{instrument}")
        rebalance_records.append({"signal_date": signal_date, "execution_date": execution_date, "status": status, "error_message": error, "target_count": len(target), "order_intent_count": len(intents), "portfolio_value_before": portfolio_value})

    def handle_order(context, order):
        order_records.append({"trading_day": _attribute(order, ["trading_day", "insert_date"]), "instrument": _attribute(order, ["instrument", "symbol"]), "direction": str(_attribute(order, ["direction"], "")), "order_qty": _attribute(order, ["order_qty", "quantity"], np.nan), "filled_qty": _attribute(order, ["filled_qty", "trade_qty"], np.nan), "order_price": _attribute(order, ["order_price", "price"], np.nan), "order_status": str(_attribute(order, ["order_status", "status"], "")), "status_msg": _attribute(order, ["status_msg", "message"], ""), "order_key": _attribute(order, ["order_key", "order_id"])})

    def handle_trade(context, trade):
        trade_records.append({"trading_day": _attribute(trade, ["trading_day", "trade_date"]), "trade_time": _attribute(trade, ["trade_time", "datetime"]), "instrument": _attribute(trade, ["instrument", "symbol"]), "direction": str(_attribute(trade, ["direction"], "")), "filled_qty": _attribute(trade, ["filled_qty", "trade_qty", "quantity"], np.nan), "filled_price": _attribute(trade, ["filled_price", "trade_price", "price"], np.nan), "filled_money": _attribute(trade, ["filled_money", "trade_amount", "amount"], np.nan), "commission": _attribute(trade, ["commission", "fee"], np.nan), "order_key": _attribute(trade, ["order_key", "order_id"])})

    def after_trading(context, data):
        date = pd.Timestamp(data.current_dt).normalize()
        _, weights, value = _current_weights(context)
        cash_weight = max(0.0, 1.0 - float(weights.sum())) if np.isfinite(value) and value > 0 else np.nan
        actual_records.extend({"date": date, "instrument": key, "actual_weight": val, "cash_weight": cash_weight, "portfolio_value": value} for key, val in weights.items())

    kwargs = {
        "data": {"instruments": engine_instruments},
        "start_date": possible_pairs["signal_date"].min().strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initialize": initialize,
        "handle_data": handle_data,
        "handle_order": handle_order,
        "handle_trade": handle_trade,
        "after_trading": after_trading,
        "capital_base": initial_cash,
        "frequency": "daily",
        "volume_limit": volume_limit,
        "order_price_field_buy": buy_field,
        "order_price_field_sell": sell_field,
        "m_cached": False,
    }
    if performance_benchmark is not None:
        kwargs["benchmark"] = performance_benchmark
    performance = M.bigtrader.v35(**kwargs)

    if show_progress:
        _progress(8, 9, "BigTrader运行完成，整理审计结果", started_at, completed=processed, total=len(possible_pairs), detail=f"成功构建{successful}次")

    schedule = possible_pairs.copy()
    schedule["is_scheduled_execution"] = schedule["execution_date"].isin(scheduled_set)
    schedule["is_index_reconstitution_signal"] = schedule["signal_date"].isin(change_set)
    result_frames = {
        "trigger_audit": pd.DataFrame(trigger_records),
        "factor_signals": pd.DataFrame(factor_records),
        "raw_weights": pd.DataFrame(raw_records),
        "target_weights": pd.DataFrame(target_records),
        "actual_weights": pd.DataFrame(actual_records),
        "active_weights": pd.DataFrame(active_records),
        "feasibility_audit": pd.DataFrame(feasibility_records),
        "risk_audit": pd.DataFrame(risk_records),
        "rebalance_audit": pd.DataFrame(rebalance_records),
        "execution_audit": pd.DataFrame(execution_records),
        "order_audit": pd.DataFrame(order_records),
        "trade_audit": pd.DataFrame(trade_records),
    }
    elapsed = time.perf_counter() - started_at
    if show_progress:
        _progress(9, 9, "回测与审计结果整理完成", started_at, completed=1, total=1, detail=f"订单{len(order_records):,}条，成交{len(trade_records):,}条")
        print()
    resolved_config = {
        "start_date": start_date,
        "end_date": end_date,
        "reference_portfolio": reference,
        "factor_name": factor_name,
        "factor_params": resolved_factor_params,
        "signal_direction": signal_direction,
        "construction_method": method,
        "construction_params": construction,
        "rebalance_rule": rule,
        "portfolio_constraints": constraints,
        "risk_model": risk,
        "feasibility_policy": policy,
        "execution_config": execution,
        "trading_costs": costs,
    }
    data_diagnostics = {
        "requested_start_date": start_date,
        "actual_first_execution_date": all_execution_dates[0],
        "engine_start_date": possible_pairs["signal_date"].min(),
        "end_date": end_date,
        "factor_history_days": history_days,
        "factor_domain_rows": factor_bundle.row_counts(),
        "reference_instrument_count": len(reference_instruments),
        "possible_signal_count": len(possible_pairs),
        "processed_signal_count": processed,
        "successful_rebalance_count": successful,
        "index_weight_scale_audit": index_scale_audit,
        "total_runtime_seconds": elapsed,
    }
    return {"performance": performance, "schedule": schedule, **result_frames, "data_diagnostics": data_diagnostics, "resolved_config": resolved_config}
