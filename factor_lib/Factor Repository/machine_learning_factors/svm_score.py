# -*- coding: utf-8 -*-
"""固定模型支持向量机选股评分因子。

本模块不查询 BigQuant，也不包含回测或交易逻辑。外层研究代码应准备：

* ``feature_panel``：date、instrument 和 ``feature_spec`` 中每个特征列；
* ``price_panel``：date、instrument、open，用于训练标签；
* ``trading_calendar``：严格递增的交易日历；
* ``universe_panel``：指数股票池时按日期的历史成员面板。

训练是一次性的，模型包可显式保存为 joblib。推理函数仅使用冻结模型，输出
``date | instrument | svm_score``；分数越高表示越靠近训练标签中的 +1 类。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "svm_score"]
SUPPORTED_KERNELS = {"linear", "rbf", "poly", "sigmoid"}
DEFAULT_LABEL_CONFIG = {
    "return_definition": "universe_equal_weight_excess",
    "entry_price_field": "open",
    "entry_offset_trading_days": 1,
    "exit_price_field": "open",
    "positive_quantile": 0.30,
    "negative_quantile": 0.30,
    "min_cross_section_size": 30,
}
DEFAULT_PREPROCESSING_CONFIG = {
    "winsorize_mad": True,
    "mad_limit": 5.0,
    "missing_policy": "industry_median_then_cross_section_median",
    "neutralize_size_industry": False,
    "zscore": True,
    "global_standard_scaler": True,
}


def _require_sklearn():
    """延迟导入，给 BigQuant 环境提供明确的依赖错误。"""
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ImportError(
            "svm_score 需要 scikit-learn。请先在 BigQuant 内核执行 "
            "`import sklearn; print(sklearn.__version__)` 确认可用性。"
        ) from exc
    return SVC, roc_auc_score


def _normalize_dates(values, name, *, allow_empty=False):
    if values is None:
        raise ValueError(f"{name} 不能为空。")
    if isinstance(values, (str, pd.Timestamp, np.datetime64)):
        values = [values]
    result = pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise"))
    result = result.normalize().unique().sort_values()
    if not allow_empty and len(result) == 0:
        raise ValueError(f"{name} 不能为空。")
    return result


def _normalize_calendar(trading_calendar):
    calendar = _normalize_dates(trading_calendar, "trading_calendar")
    if len(calendar) < 3:
        raise ValueError("trading_calendar 至少需要 3 个交易日。")
    return calendar


def _as_date(value, name):
    if value is None:
        raise ValueError(f"{name} 不能为空。")
    date = pd.Timestamp(value).normalize()
    if pd.isna(date):
        raise ValueError(f"{name} 不是有效日期。")
    return date


def _normalize_feature_spec(feature_spec):
    if not isinstance(feature_spec, Sequence) or isinstance(feature_spec, str):
        raise TypeError("feature_spec 必须是非空列表。")
    if not feature_spec:
        raise ValueError("feature_spec 不能为空。")
    normalized, names = [], set()
    for position, item in enumerate(feature_spec, start=1):
        if not isinstance(item, Mapping):
            raise TypeError(f"feature_spec 第 {position} 项必须是字典。")
        required = {"factor_name", "params", "feature_name"}
        missing = sorted(required - set(item))
        unknown = sorted(set(item) - required)
        if missing or unknown:
            raise ValueError(
                f"feature_spec 第 {position} 项字段错误，"
                f"缺少={missing}，未知={unknown}。"
            )
        factor_name = item["factor_name"]
        feature_name = item["feature_name"]
        params = item["params"]
        if not isinstance(factor_name, str) or not factor_name.strip():
            raise ValueError(f"feature_spec 第 {position} 项 factor_name 无效。")
        if not isinstance(feature_name, str) or not feature_name.strip():
            raise ValueError(f"feature_spec 第 {position} 项 feature_name 无效。")
        if feature_name in names:
            raise ValueError(f"feature_name 重复：{feature_name!r}。")
        if not isinstance(params, Mapping):
            raise TypeError(f"{feature_name!r} 的 params 必须是字典。")
        names.add(feature_name)
        normalized.append(
            {
                "factor_name": factor_name.strip(),
                "params": dict(params),
                "feature_name": feature_name.strip(),
            }
        )
    return normalized


def _json_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_universe(universe):
    if not isinstance(universe, Mapping):
        raise TypeError("universe 必须是字典。")
    result = dict(universe)
    kind = result.get("type")
    if kind not in {"all_a", "custom", "index"}:
        raise ValueError("universe.type 必须是 all_a、custom 或 index。")
    if kind == "custom":
        instruments = result.get("instruments")
        if not isinstance(instruments, Sequence) or isinstance(instruments, str):
            raise TypeError("custom universe 必须提供 instruments 列表。")
        instruments = sorted({str(item).strip() for item in instruments if str(item).strip()})
        if not instruments:
            raise ValueError("custom universe.instruments 不能为空。")
        result["instruments"] = instruments
    if kind == "index":
        codes = result.get("index_codes")
        if not isinstance(codes, Sequence) or isinstance(codes, str):
            raise TypeError("index universe 必须提供 index_codes 列表。")
        codes = [str(code).strip() for code in codes if str(code).strip()]
        if not codes:
            raise ValueError("index universe.index_codes 不能为空。")
        result["index_codes"] = codes
    return result


def _normalize_label_config(label_config, prediction_label_window_trading_days):
    config = dict(DEFAULT_LABEL_CONFIG)
    if label_config is not None:
        if not isinstance(label_config, Mapping):
            raise TypeError("label_config 必须是字典或 None。")
        unknown = sorted(set(label_config) - set(DEFAULT_LABEL_CONFIG))
        if unknown:
            raise ValueError(f"label_config 包含未知字段：{unknown}。")
        config.update(label_config)
    horizon = prediction_label_window_trading_days
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("prediction_label_window_trading_days 必须是正整数。")
    for key in ("positive_quantile", "negative_quantile"):
        value = float(config[key])
        if not 0 < value < 0.5:
            raise ValueError(f"label_config[{key!r}] 必须位于 (0, 0.5)。")
        config[key] = value
    if config["positive_quantile"] + config["negative_quantile"] >= 1.0:
        raise ValueError("正负类别分位数之和必须小于 1。")
    if config["return_definition"] != "universe_equal_weight_excess":
        raise ValueError("首版仅支持 universe_equal_weight_excess 标签收益。")
    if config["entry_price_field"] != "open" or config["exit_price_field"] != "open":
        raise ValueError("首版标签固定使用开盘价，不接受其他价格字段。")
    if int(config["entry_offset_trading_days"]) != 1:
        raise ValueError("首版标签固定在信号日下一交易日开盘开始。")
    minimum = config["min_cross_section_size"]
    if not isinstance(minimum, (int, np.integer)) or isinstance(minimum, bool) or minimum < 3:
        raise ValueError("label_config.min_cross_section_size 必须是不小于 3 的整数。")
    config["min_cross_section_size"] = int(minimum)
    config["prediction_label_window_trading_days"] = int(horizon)
    config["exit_offset_trading_days"] = int(horizon) + 1
    return config


def _normalize_preprocessing_config(preprocessing_config):
    config = dict(DEFAULT_PREPROCESSING_CONFIG)
    if preprocessing_config is not None:
        if not isinstance(preprocessing_config, Mapping):
            raise TypeError("preprocessing_config 必须是字典或 None。")
        unknown = sorted(set(preprocessing_config) - set(config))
        if unknown:
            raise ValueError(f"preprocessing_config 包含未知字段：{unknown}。")
        config.update(preprocessing_config)
    config["mad_limit"] = float(config["mad_limit"])
    if not np.isfinite(config["mad_limit"]) or config["mad_limit"] <= 0:
        raise ValueError("preprocessing_config.mad_limit 必须是正数。")
    allowed_missing = {"industry_median_then_cross_section_median", "cross_section_median"}
    if config["missing_policy"] not in allowed_missing:
        raise ValueError(f"missing_policy 必须是 {sorted(allowed_missing)} 之一。")
    for key in ("winsorize_mad", "neutralize_size_industry", "zscore", "global_standard_scaler"):
        if not isinstance(config[key], (bool, np.bool_)):
            raise TypeError(f"preprocessing_config[{key!r}] 必须是 bool。")
        config[key] = bool(config[key])
    return config


def _normalize_model_config(model_config):
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config 必须是字典。")
    config = dict(model_config)
    kernel = config.get("kernel")
    if kernel not in SUPPORTED_KERNELS:
        raise ValueError(f"kernel 必须是 {sorted(SUPPORTED_KERNELS)} 之一。")
    has_fixed = any(key in config for key in ("C", "gamma", "degree", "coef0"))
    has_search = "hyperparameter_search" in config
    if has_fixed and has_search:
        raise ValueError("固定超参数与 hyperparameter_search 不能同时传入。")
    allowed = {"kernel", "C", "gamma", "degree", "coef0", "class_weight", "tol", "max_iter", "cache_size", "hyperparameter_search"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"model_config 包含未知字段：{unknown}。")
    if has_search:
        search = config["hyperparameter_search"]
        if not isinstance(search, Mapping):
            raise TypeError("hyperparameter_search 必须是字典。")
        return {"kernel": kernel, "search": _normalize_search_config(kernel, search), "base": config}

    fixed = {"kernel": kernel, "class_weight": config.get("class_weight"), "tol": float(config.get("tol", 1e-3)), "max_iter": int(config.get("max_iter", -1)), "cache_size": float(config.get("cache_size", 200.0))}
    fixed["C"] = _positive_float(config.get("C", 1.0), "model_config.C")
    if kernel in {"rbf", "poly", "sigmoid"}:
        fixed["gamma"] = _positive_float(config.get("gamma", "scale"), "model_config.gamma", allow_scale=True)
    if kernel == "poly":
        fixed["degree"] = _positive_int(config.get("degree", 3), "model_config.degree")
        fixed["coef0"] = float(config.get("coef0", 0.0))
    if kernel == "sigmoid":
        fixed["coef0"] = float(config.get("coef0", 0.0))
    return {"kernel": kernel, "search": None, "base": fixed}


def _positive_float(value, name, *, allow_scale=False):
    if allow_scale and value == "scale":
        return value
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是正数。")
    return value


def _positive_int(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数。")
    return int(value)


def _normalize_search_config(kernel, search):
    allowed = {"C_values", "gamma_values", "degree_values", "coef0_values", "metric"}
    unknown = sorted(set(search) - allowed)
    if unknown:
        raise ValueError(f"hyperparameter_search 包含未知字段：{unknown}。")
    metric = search.get("metric", "auc")
    if metric not in {"auc", "rank_ic"}:
        raise ValueError("hyperparameter_search.metric 必须是 auc 或 rank_ic。")
    result = {"metric": metric, "C_values": _positive_values(search.get("C_values"), "C_values")}
    if kernel in {"rbf", "poly", "sigmoid"}:
        result["gamma_values"] = _positive_values(search.get("gamma_values"), "gamma_values")
    if kernel == "poly":
        values = search.get("degree_values")
        if values is None:
            raise ValueError("poly 核的网格搜索必须提供 degree_values。")
        result["degree_values"] = [_positive_int(value, "degree_values") for value in values]
        result["coef0_values"] = _finite_values(search.get("coef0_values", [0.0]), "coef0_values")
    if kernel == "sigmoid":
        result["coef0_values"] = _finite_values(search.get("coef0_values", [0.0]), "coef0_values")
    return result


def _positive_values(values, name):
    if not isinstance(values, Sequence) or isinstance(values, str) or not values:
        raise ValueError(f"{name} 必须是非空候选值列表。")
    return [_positive_float(value, name) for value in values]


def _finite_values(values, name):
    if not isinstance(values, Sequence) or isinstance(values, str) or not values:
        raise ValueError(f"{name} 必须是非空候选值列表。")
    normalized = [float(value) for value in values]
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} 必须都是有限数。")
    return normalized


def _validate_feature_panel(feature_panel, feature_names, preprocessing_config):
    if not isinstance(feature_panel, pd.DataFrame):
        raise TypeError("feature_panel/data 必须是 pandas.DataFrame。")
    required = {"date", "instrument", *feature_names}
    if preprocessing_config["missing_policy"] == "industry_median_then_cross_section_median":
        # 行业字段可选；缺失时退化为截面中位数填补，保证基础因子不因该辅助字段失效。
        required -= {"industry"}
    missing = sorted(required - set(feature_panel.columns))
    if missing:
        raise ValueError(f"feature_panel 缺少字段：{missing}。")
    result = feature_panel.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result[["date", "instrument"]].isna().any().any():
        raise ValueError("feature_panel 的 date/instrument 不能缺失。")
    if result.duplicated(["date", "instrument"]).any():
        raise ValueError("feature_panel 存在重复的 date + instrument。")
    for name in feature_names:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    if "total_market_cap" in result:
        result["total_market_cap"] = pd.to_numeric(result["total_market_cap"], errors="coerce")
    return result


def _resolve_universe_panel(universe, dates, feature_panel, universe_panel=None):
    """生成 date、instrument、is_member，指数必须由外层提供历史成员表。"""
    config = _normalize_universe(universe)
    date_values = _normalize_dates(dates, "universe dates")
    available = feature_panel[["date", "instrument"]].drop_duplicates().copy()
    available = available.loc[available["date"].isin(date_values)]
    if config["type"] == "all_a":
        result = available.assign(is_member=True)
    elif config["type"] == "custom":
        result = available.loc[available["instrument"].astype(str).isin(config["instruments"])].copy()
        result["is_member"] = True
    else:
        if not isinstance(universe_panel, pd.DataFrame):
            raise TypeError("指数股票池必须传入按日期的 universe_panel。")
        required = {"date", "instrument"}
        missing = sorted(required - set(universe_panel.columns))
        if missing:
            raise ValueError(f"universe_panel 缺少字段：{missing}。")
        membership = universe_panel[["date", "instrument"] + (["is_member"] if "is_member" in universe_panel else [])].copy()
        membership["date"] = pd.to_datetime(membership["date"], errors="coerce").dt.normalize()
        membership = membership.dropna(subset=["date", "instrument"])
        if "is_member" not in membership:
            membership["is_member"] = True
        membership["is_member"] = membership["is_member"].astype(bool)
        membership = membership.loc[membership["date"].isin(date_values) & membership["is_member"]]
        if membership.duplicated(["date", "instrument"]).any():
            raise ValueError("universe_panel 存在重复的 date + instrument。")
        result = available.merge(membership, on=["date", "instrument"], how="inner", validate="one_to_one")
    return result[["date", "instrument", "is_member"]].sort_values(["date", "instrument"]).reset_index(drop=True)


def _apply_cross_section_preprocessing(
    panel,
    feature_names,
    config,
    progress_callback=None,
):
    """仅用当日截面数据变换；不拟合任何跨期统计量。"""
    result = panel.copy()
    transformed = []
    date_groups = list(result.groupby("date", sort=True))
    for position, (date, group) in enumerate(date_groups, start=1):
        group = group.copy()
        values = group[feature_names].astype(float)
        if config["winsorize_mad"]:
            med = values.median(axis=0, skipna=True)
            mad = (values - med).abs().median(axis=0, skipna=True)
            lower = med - config["mad_limit"] * mad
            upper = med + config["mad_limit"] * mad
            values = values.clip(lower=lower, upper=upper, axis=1)
        if config["missing_policy"] == "industry_median_then_cross_section_median" and "industry" in group:
            for name in feature_names:
                values[name] = values[name].fillna(group.assign(_value=values[name]).groupby("industry")["_value"].transform("median"))
        values = values.fillna(values.median(axis=0, skipna=True))
        if values.isna().any().any():
            raise ValueError(f"{date:%Y-%m-%d} 截面存在全缺失特征，无法训练或推理。")
        if config["neutralize_size_industry"]:
            controls = [np.ones(len(group), dtype=float)]
            if "total_market_cap" in group:
                cap = pd.to_numeric(group["total_market_cap"], errors="coerce")
                log_cap = np.log(cap.where(cap > 0))
                log_cap = log_cap.fillna(log_cap.median())
                if log_cap.isna().any():
                    raise ValueError(f"{date:%Y-%m-%d} 缺少可用 total_market_cap。")
                controls.append(log_cap.to_numpy(dtype=float))
            if "industry" in group:
                dummies = pd.get_dummies(group["industry"].astype(str), dtype=float)
                if dummies.shape[1] > 1:
                    controls.extend(dummies.iloc[:, 1:].to_numpy(dtype=float).T)
            design = np.column_stack(controls)
            for name in feature_names:
                y = values[name].to_numpy(dtype=float)
                beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                values[name] = y - design @ beta
        if config["zscore"]:
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=0).replace(0.0, np.nan)
            values = (values - mean) / std
            values = values.fillna(0.0)
        group.loc[:, feature_names] = values.to_numpy(dtype=float)
        transformed.append(group)
        if progress_callback is not None:
            progress_callback(position, len(date_groups), date)
    return pd.concat(transformed, ignore_index=True).sort_values(["date", "instrument"]).reset_index(drop=True)


def _fit_scaler(frame, feature_names, enabled):
    values = frame[feature_names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("训练特征经过截面预处理后仍包含非有限值。")
    if not enabled:
        return {"enabled": False, "mean": np.zeros(len(feature_names)), "std": np.ones(len(feature_names))}
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    std = np.where(std > 0, std, 1.0)
    return {"enabled": True, "mean": mean.astype(float), "std": std.astype(float)}


def _apply_scaler(frame, feature_names, scaler):
    result = frame.copy()
    mean = np.asarray(scaler["mean"], dtype=float)
    std = np.asarray(scaler["std"], dtype=float)
    if len(mean) != len(feature_names) or len(std) != len(feature_names) or (std <= 0).any():
        raise ValueError("模型包中的 scaler 与 feature_spec 不一致。")
    result.loc[:, feature_names] = (result[feature_names].to_numpy(dtype=float) - mean) / std
    return result


def _make_signal_dates(calendar, start, end, interval):
    if not isinstance(interval, (int, np.integer)) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("signal_interval_trading_days 必须是正整数。")
    candidates = calendar[(calendar >= start) & (calendar <= end)]
    if len(candidates) == 0:
        raise ValueError("训练区间内没有交易日。")
    return candidates[:: int(interval)]


def _build_labels(
    price_panel,
    signal_dates,
    calendar,
    membership,
    label_config,
    anchor_date,
    progress_callback=None,
):
    if not isinstance(price_panel, pd.DataFrame):
        raise TypeError("price_panel 必须是 pandas.DataFrame。")
    required = {"date", "instrument", "open"}
    missing = sorted(required - set(price_panel.columns))
    if missing:
        raise ValueError(f"price_panel 缺少字段：{missing}。")
    prices = price_panel[["date", "instrument", "open"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["open"] = pd.to_numeric(prices["open"], errors="coerce")
    prices = prices.dropna(subset=["date", "instrument"]).drop_duplicates(["date", "instrument"])
    positions = {date: index for index, date in enumerate(calendar)}
    rows = []
    horizon = label_config["prediction_label_window_trading_days"]
    for date in signal_dates:
        position = positions.get(date)
        exit_position = None if position is None else position + label_config["exit_offset_trading_days"]
        if exit_position is None or exit_position >= len(calendar):
            continue
        entry_date = calendar[position + label_config["entry_offset_trading_days"]]
        exit_date = calendar[exit_position]
        if exit_date > anchor_date:
            raise ValueError("训练标签结束日晚于 training_anchor_date，存在未来信息。")
        rows.append((date, entry_date, exit_date, horizon))
    mapping = pd.DataFrame(rows, columns=["date", "entry_date", "exit_date", "prediction_label_window_trading_days"])
    if mapping.empty:
        raise ValueError("没有可构造完整标签的训练信号日。")
    start = prices.rename(columns={"date": "entry_date", "open": "entry_open"})
    end = prices.rename(columns={"date": "exit_date", "open": "exit_open"})
    labels = membership.merge(mapping, on="date", how="inner")
    labels = labels.merge(start, on=["entry_date", "instrument"], how="left")
    labels = labels.merge(end, on=["exit_date", "instrument"], how="left")
    labels["raw_forward_return"] = labels["exit_open"] / labels["entry_open"] - 1.0
    labels = labels.replace([np.inf, -np.inf], np.nan).dropna(subset=["raw_forward_return"])
    grouped_labels = {
        date: group for date, group in labels.groupby("date", sort=True)
    }
    valid_rows = []
    mapped_dates = pd.DatetimeIndex(mapping["date"].unique()).sort_values()
    for position, date in enumerate(mapped_dates, start=1):
        group = grouped_labels.get(date)
        if group is None:
            if progress_callback is not None:
                progress_callback(position, len(mapped_dates), date)
            continue
        if len(group) < label_config["min_cross_section_size"]:
            if progress_callback is not None:
                progress_callback(position, len(mapped_dates), date)
            continue
        group = group.copy()
        group["forward_excess_return"] = group["raw_forward_return"] - group["raw_forward_return"].mean()
        count = len(group)
        positive_count = max(1, int(math.ceil(count * label_config["positive_quantile"])))
        negative_count = max(1, int(math.ceil(count * label_config["negative_quantile"])))
        ordered = group.sort_values("forward_excess_return", kind="mergesort")
        ordered["label"] = np.nan
        ordered.iloc[:negative_count, ordered.columns.get_loc("label")] = -1
        ordered.iloc[-positive_count:, ordered.columns.get_loc("label")] = 1
        valid_rows.append(ordered)
        if progress_callback is not None:
            progress_callback(position, len(mapped_dates), date)
    if not valid_rows:
        raise ValueError("没有达到最小截面样本数的完整训练标签。")
    return pd.concat(valid_rows, ignore_index=True)


def _prepare_samples(transformed_features, labels, feature_names):
    merged = labels.merge(
        transformed_features[["date", "instrument", *feature_names]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    classified = merged.dropna(subset=["label"]).copy()
    if classified.empty:
        raise ValueError("训练/验证合并后没有二分类样本。")
    if not set(classified["label"].astype(int).unique()) == {-1, 1}:
        raise ValueError("训练或验证样本必须同时包含 +1 与 -1 类。")
    return classified


def _grid_candidates(normalized_config):
    base = normalized_config["base"]
    search = normalized_config["search"]
    if search is None:
        return [base]
    kernel = normalized_config["kernel"]
    rows = []
    gamma_values = search.get("gamma_values", [None])
    degree_values = search.get("degree_values", [None])
    coef0_values = search.get("coef0_values", [None])
    for c_value in search["C_values"]:
        for gamma in gamma_values:
            for degree in degree_values:
                for coef0 in coef0_values:
                    candidate = {
                        "kernel": kernel,
                        "C": c_value,
                        "class_weight": base.get("class_weight"),
                        "tol": float(base.get("tol", 1e-3)),
                        "max_iter": int(base.get("max_iter", -1)),
                        "cache_size": float(base.get("cache_size", 200.0)),
                    }
                    if gamma is not None:
                        candidate["gamma"] = gamma
                    if degree is not None:
                        candidate["degree"] = degree
                    if coef0 is not None:
                        candidate["coef0"] = coef0
                    rows.append(candidate)
    return rows


def _fit_svc(parameters, x_train, y_train):
    SVC, _ = _require_sklearn()
    return SVC(**parameters).fit(x_train, y_train)


def _positive_class_margin(model, x_values):
    raw = np.asarray(model.decision_function(x_values), dtype=float).reshape(-1)
    predicted = np.asarray(model.predict(x_values), dtype=int).reshape(-1)
    if set(model.classes_) != {-1, 1}:
        raise ValueError("SVM 模型类别必须精确为 {-1, +1}。")
    positive_if_raw_positive = np.where(raw >= 0, 1, -1)
    if np.array_equal(positive_if_raw_positive, predicted):
        return raw, 1
    if np.array_equal(-positive_if_raw_positive, predicted):
        return -raw, -1
    raise RuntimeError("无法依据模型类别编码确定朝 +1 类的决策分数方向。")


def _validation_metric(model, validation_frame, feature_names, metric):
    _, roc_auc_score = _require_sklearn()
    x_values = validation_frame[feature_names].to_numpy(dtype=float)
    y_values = validation_frame["label"].to_numpy(dtype=int)
    scores, sign = _positive_class_margin(model, x_values)
    auc = float(roc_auc_score(y_values, scores))
    rank_ic = float(pd.Series(scores).corr(validation_frame["forward_excess_return"], method="spearman"))
    selected = auc if metric == "auc" else rank_ic
    return {"selected_metric": selected, "auc": auc, "rank_ic": rank_ic, "decision_value_positive_sign": sign}


def _render_progress(stage, completed, total, started_at, detail=""):
    elapsed = time.perf_counter() - started_at
    percentage = 100.0 * completed / total if total else 100.0
    eta = ""
    if 0 < completed < total:
        eta = f"，预计剩余 {elapsed / completed * (total - completed):.1f}s"
    print(
        f"\r[SVM] {stage} {completed}/{total}（{percentage:6.2f}%），"
        f"耗时 {elapsed:.1f}s{eta} {detail}".ljust(180),
        end="",
        flush=True,
    )


def _bundle_metadata(bundle):
    return {
        "model_version": bundle["model_version"],
        "feature_spec": bundle["feature_spec"],
        "feature_schema_hash": bundle["feature_schema_hash"],
        "label_config": bundle["label_config"],
        "preprocessing_config": bundle["preprocessing_config"],
        "selected_model_config": bundle["selected_model_config"],
        "training_universe_config": bundle["training_universe_config"],
        "training_metadata": bundle["training_metadata"],
        "decision_score_definition": bundle["decision_score_definition"],
    }


def save_svm_model_bundle(model_bundle, model_artifact_dir):
    """显式保存模型包；目标目录已存在时拒绝覆盖。"""
    bundle = validate_svm_model_bundle(model_bundle)
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("保存 SVM 模型包需要 joblib。") from exc
    directory = Path(model_artifact_dir).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"模型目录已存在，拒绝覆盖：{directory}")
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model_bundle.joblib"
    metadata_path = directory / "metadata.json"
    joblib.dump(bundle, model_path, compress=3)
    metadata = _bundle_metadata(bundle)
    metadata["model_bundle_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"model_path": str(model_path), "metadata_path": str(metadata_path)}


def load_svm_model_bundle(model_artifact_dir):
    """加载受信任的本项目模型包，并校验可推理结构。"""
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("加载 SVM 模型包需要 joblib。") from exc
    path = Path(model_artifact_dir).expanduser().resolve()
    model_path = path / "model_bundle.joblib" if path.is_dir() else path
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型文件：{model_path}")
    return validate_svm_model_bundle(joblib.load(model_path))


def validate_svm_model_bundle(model_bundle):
    if not isinstance(model_bundle, Mapping):
        raise TypeError("model_bundle 必须是字典。")
    required = {
        "fitted_svm", "feature_spec", "feature_schema_hash", "preprocessing_config",
        "scaler", "label_config", "selected_model_config", "training_universe_config",
        "training_metadata", "model_version", "decision_score_definition",
        "decision_value_positive_sign",
    }
    missing = sorted(required - set(model_bundle))
    if missing:
        raise ValueError(f"model_bundle 缺少字段：{missing}。")
    bundle = dict(model_bundle)
    feature_spec = _normalize_feature_spec(bundle["feature_spec"])
    if bundle["feature_schema_hash"] != _json_hash(feature_spec):
        raise ValueError("model_bundle 的 feature_schema_hash 校验失败。")
    if bundle["decision_score_definition"] != "margin_toward_positive_class":
        raise ValueError("model_bundle 的决策分数定义不受支持。")
    if bundle["decision_value_positive_sign"] not in {-1, 1}:
        raise ValueError("decision_value_positive_sign 必须为 -1 或 1。")
    _normalize_universe(bundle["training_universe_config"])
    _normalize_preprocessing_config(bundle["preprocessing_config"])
    stored_label_config = dict(bundle["label_config"])
    try:
        stored_horizon = stored_label_config.pop("prediction_label_window_trading_days")
        stored_exit_offset = stored_label_config.pop("exit_offset_trading_days")
    except KeyError as exc:
        raise ValueError(f"model_bundle.label_config 缺少字段：{exc.args[0]}。") from exc
    normalized_label_config = _normalize_label_config(stored_label_config, stored_horizon)
    if int(stored_exit_offset) != normalized_label_config["exit_offset_trading_days"]:
        raise ValueError("model_bundle.label_config 的 exit_offset_trading_days 不一致。")
    model = bundle["fitted_svm"]
    if not callable(getattr(model, "decision_function", None)) or not callable(getattr(model, "predict", None)):
        raise TypeError("model_bundle.fitted_svm 不是可用的 SVM 分类器。")
    feature_count = len(feature_spec)
    scaler = bundle["scaler"]
    if not isinstance(scaler, Mapping):
        raise TypeError("model_bundle.scaler 必须是字典。")
    if len(np.asarray(scaler.get("mean"), dtype=float)) != feature_count or len(np.asarray(scaler.get("std"), dtype=float)) != feature_count:
        raise ValueError("model_bundle.scaler 维度与 feature_spec 不一致。")
    return bundle


def train_svm_model(
    *,
    feature_panel,
    price_panel,
    trading_calendar,
    feature_spec,
    training_start_date,
    training_end_date,
    training_anchor_date,
    validation_start_date,
    validation_end_date,
    universe,
    universe_panel=None,
    signal_interval_trading_days=20,
    prediction_label_window_trading_days=20,
    label_config=None,
    preprocessing_config=None,
    model_config=None,
    model_version="svm_fixed_v1",
    persist_model_bundle=False,
    model_artifact_dir=None,
    refit_on_train_and_validation=True,
    show_progress=True,
    progress_every=20,
):
    """一次性训练固定 SVM，并可显式保存完整模型包。

    ``feature_panel``、``price_panel``、``trading_calendar`` 与指数历史成员表均
    由外层研究/数据适配器准备。``training_anchor_date`` 必须不早于所有训练及
    验证标签的结束日期，从而保证训练时未来标签已经完整实现。
    """
    started_at = time.perf_counter()
    feature_spec = _normalize_feature_spec(feature_spec)
    feature_names = [item["feature_name"] for item in feature_spec]
    label_cfg = _normalize_label_config(label_config, prediction_label_window_trading_days)
    preprocess_cfg = _normalize_preprocessing_config(preprocessing_config)
    normalized_model = _normalize_model_config(model_config or {"kernel": "rbf", "C": 1.0, "gamma": "scale"})
    calendar = _normalize_calendar(trading_calendar)
    train_start = _as_date(training_start_date, "training_start_date")
    train_end = _as_date(training_end_date, "training_end_date")
    validation_start = _as_date(validation_start_date, "validation_start_date")
    validation_end = _as_date(validation_end_date, "validation_end_date")
    anchor = _as_date(training_anchor_date, "training_anchor_date")
    if not train_start <= train_end < validation_start <= validation_end <= anchor:
        raise ValueError("日期必须满足 training_start <= training_end < validation_start <= validation_end <= anchor。")
    if not isinstance(persist_model_bundle, (bool, np.bool_)):
        raise TypeError("persist_model_bundle 必须是 bool。")
    if persist_model_bundle and not model_artifact_dir:
        raise ValueError("persist_model_bundle=True 时必须提供 model_artifact_dir。")
    if not isinstance(refit_on_train_and_validation, (bool, np.bool_)):
        raise TypeError("refit_on_train_and_validation 必须是 bool。")
    if not isinstance(progress_every, (int, np.integer)) or isinstance(progress_every, bool) or progress_every <= 0:
        raise ValueError("progress_every 必须是正整数。")

    raw_features = _validate_feature_panel(feature_panel, feature_names, preprocess_cfg)
    train_signals = _make_signal_dates(calendar, train_start, train_end, signal_interval_trading_days)
    validation_signals = _make_signal_dates(calendar, validation_start, validation_end, signal_interval_trading_days)
    all_signals = train_signals.union(validation_signals).sort_values()
    membership = _resolve_universe_panel(universe, all_signals, raw_features, universe_panel)
    if membership.empty:
        raise ValueError("训练股票池在全部信号日没有可用股票。")

    positions = {date: position for position, date in enumerate(calendar)}
    validation_first = positions[validation_signals.min()]
    train_signals = pd.DatetimeIndex(
        [date for date in train_signals if positions[date] + label_cfg["exit_offset_trading_days"] < validation_first]
    )
    if len(train_signals) < 2:
        raise ValueError("按预测标签窗口净化后，训练信号日不足 2 个。")
    all_signals = train_signals.union(validation_signals).sort_values()
    membership = membership.loc[membership["date"].isin(all_signals)].copy()

    def label_progress(completed, total, date):
        if show_progress and (
            completed == 1
            or completed % int(progress_every) == 0
            or completed == total
        ):
            _render_progress(
                "构造点时标签",
                completed,
                total,
                started_at,
                f"当前 {date:%Y-%m-%d}",
            )

    if show_progress:
        _render_progress("构造点时标签", 0, len(all_signals), started_at, "准备开盘价收益标签")
    labels = _build_labels(
        price_panel,
        all_signals,
        calendar,
        membership,
        label_cfg,
        anchor,
        progress_callback=label_progress if show_progress else None,
    )
    labels = labels.loc[labels["date"].isin(all_signals)].copy()
    if show_progress:
        _render_progress("构造点时标签", 1, 1, started_at, f"完整标签 {len(labels):,} 条")

    features = raw_features.loc[raw_features["date"].isin(all_signals)].copy()
    features = features.merge(membership[["date", "instrument"]], on=["date", "instrument"], how="inner", validate="one_to_one")
    def preprocessing_progress(completed, total, date):
        if show_progress and (
            completed == 1
            or completed % int(progress_every) == 0
            or completed == total
        ):
            _render_progress(
                "截面预处理",
                completed,
                total,
                started_at,
                f"当前 {date:%Y-%m-%d}，特征 {len(feature_names)} 个",
            )

    if show_progress:
        _render_progress("截面预处理", 0, len(all_signals), started_at, f"特征 {len(feature_names)} 个")
    transformed = _apply_cross_section_preprocessing(
        features,
        feature_names,
        preprocess_cfg,
        progress_callback=preprocessing_progress if show_progress else None,
    )
    train_labels = labels.loc[labels["date"].isin(train_signals)]
    validation_labels = labels.loc[labels["date"].isin(validation_signals)]
    train_frame = _prepare_samples(transformed, train_labels, feature_names)
    validation_frame = _prepare_samples(transformed, validation_labels, feature_names)
    if show_progress:
        _render_progress("截面预处理", 1, 1, started_at, f"训练样本 {len(train_frame):,}，验证样本 {len(validation_frame):,}")

    scaler = _fit_scaler(train_frame, feature_names, preprocess_cfg["global_standard_scaler"])
    train_scaled = _apply_scaler(train_frame, feature_names, scaler)
    validation_scaled = _apply_scaler(validation_frame, feature_names, scaler)
    x_train = train_scaled[feature_names].to_numpy(dtype=float)
    y_train = train_scaled["label"].to_numpy(dtype=int)
    candidates = _grid_candidates(normalized_model)
    metric_name = normalized_model["search"]["metric"] if normalized_model["search"] else "auc"
    rows, best = [], None
    for position, candidate in enumerate(candidates, start=1):
        if show_progress:
            _render_progress("训练并验证超参数", position - 1, len(candidates), started_at, str(candidate))
        model = _fit_svc(candidate, x_train, y_train)
        metrics = _validation_metric(model, validation_scaled, feature_names, metric_name)
        record = {**candidate, **metrics}
        rows.append(record)
        if best is None or record["selected_metric"] > best["selected_metric"]:
            best = record
        if show_progress:
            _render_progress("训练并验证超参数", position, len(candidates), started_at, f"{metric_name}={record['selected_metric']:.6f}")
    if show_progress:
        print()

    # _grid_candidates 理论上已保证至少有一个候选组合；这里仍显式保护，
    # 既防止运行时出现空搜索，也让静态类型检查明确 best 不再是 None。
    if best is None:
        raise RuntimeError("超参数搜索未产生有效候选模型，无法继续训练最终模型。")

    selected_config = {key: value for key, value in best.items() if key in {"kernel", "C", "gamma", "degree", "coef0", "class_weight", "tol", "max_iter", "cache_size"}}
    final_frame = pd.concat([train_frame, validation_frame], ignore_index=True) if refit_on_train_and_validation else train_frame.copy()
    final_scaler = _fit_scaler(final_frame, feature_names, preprocess_cfg["global_standard_scaler"])
    final_scaled = _apply_scaler(final_frame, feature_names, final_scaler)
    if show_progress:
        _render_progress("拟合最终冻结模型", 0, 1, started_at, f"样本 {len(final_scaled):,}")
    fitted = _fit_svc(selected_config, final_scaled[feature_names].to_numpy(dtype=float), final_scaled["label"].to_numpy(dtype=int))
    if show_progress:
        _render_progress("拟合最终冻结模型", 1, 1, started_at, f"kernel={selected_config['kernel']}")
        print()
    _, positive_sign = _positive_class_margin(fitted, final_scaled[feature_names].to_numpy(dtype=float))
    feature_hash = _json_hash(feature_spec)
    bundle = {
        "fitted_svm": fitted,
        "feature_spec": feature_spec,
        "feature_schema_hash": feature_hash,
        "preprocessing_config": preprocess_cfg,
        "scaler": final_scaler,
        "label_config": label_cfg,
        "selected_model_config": selected_config,
        "training_universe_config": _normalize_universe(universe),
        "training_metadata": {
            "training_start_date": str(train_start.date()),
            "training_end_date": str(train_end.date()),
            "training_anchor_date": str(anchor.date()),
            "validation_start_date": str(validation_start.date()),
            "validation_end_date": str(validation_end.date()),
            "signal_interval_trading_days": int(signal_interval_trading_days),
            "training_signal_dates": [str(date.date()) for date in train_signals],
            "validation_signal_dates": [str(date.date()) for date in validation_signals],
            "training_sample_count": int(len(train_frame)),
            "validation_sample_count": int(len(validation_frame)),
            "refit_on_train_and_validation": bool(refit_on_train_and_validation),
            "validation_metric": metric_name,
            "validation_selected_metric": float(best["selected_metric"]),
            "validation_auc": float(best["auc"]),
            "validation_rank_ic": float(best["rank_ic"]),
            "hyperparameter_results": rows,
        },
        "model_version": str(model_version),
        "decision_score_definition": "margin_toward_positive_class",
        "decision_value_positive_sign": int(positive_sign),
    }
    bundle = validate_svm_model_bundle(bundle)
    if persist_model_bundle:
        bundle["artifact_paths"] = save_svm_model_bundle(bundle, model_artifact_dir)
    return bundle


def calc_svm_score(
    data,
    target_dates=None,
    as_of_date=None,
    *,
    universe,
    universe_panel=None,
    model_bundle=None,
    model_artifact_dir=None,
    show_progress=True,
    progress_every=20,
):
    """使用固定模型包计算目标日的 SVM 截面分数。

    ``data`` 是外层准备好的宽表特征面板。它必须含有模型包 feature_spec 中的
    每个 feature_name，指数股票池需要 ``universe_panel`` 提供每日历史成员。
    """
    if model_bundle is not None and model_artifact_dir is not None:
        raise ValueError("model_bundle 与 model_artifact_dir 只能提供一个。")
    if model_bundle is None:
        if model_artifact_dir is None:
            raise ValueError("推理必须提供 model_bundle 或 model_artifact_dir。")
        model_bundle = load_svm_model_bundle(model_artifact_dir)
    bundle = validate_svm_model_bundle(model_bundle)
    if not isinstance(progress_every, (int, np.integer)) or isinstance(progress_every, bool) or progress_every <= 0:
        raise ValueError("progress_every 必须是正整数。")
    feature_spec = bundle["feature_spec"]
    feature_names = [item["feature_name"] for item in feature_spec]
    raw = _validate_feature_panel(data, feature_names, bundle["preprocessing_config"])
    targets = _normalize_dates(target_dates if target_dates is not None else raw["date"].unique(), "target_dates")
    if as_of_date is not None and (targets > _as_date(as_of_date, "as_of_date")).any():
        raise ValueError("target_dates 不得晚于 as_of_date。")
    raw = raw.loc[raw["date"].isin(targets)].copy()
    membership = _resolve_universe_panel(universe, targets, raw, universe_panel)
    section = raw.merge(membership[["date", "instrument"]], on=["date", "instrument"], how="inner", validate="one_to_one")
    if section.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    model = bundle["fitted_svm"]
    groups = {date: group for date, group in section.groupby("date", sort=True)}
    result_parts = []
    started = time.perf_counter()
    if show_progress:
        _render_progress("固定模型推理", 0, len(targets), started, "准备逐日预处理与评分")
    for position, date in enumerate(targets, start=1):
        group = groups.get(date)
        if group is None or group.empty:
            valid_count = 0
        else:
            transformed = _apply_cross_section_preprocessing(
                group,
                feature_names,
                bundle["preprocessing_config"],
            )
            standardized = _apply_scaler(
                transformed,
                feature_names,
                bundle["scaler"],
            )
            raw_score = np.asarray(
                model.decision_function(
                    standardized[feature_names].to_numpy(dtype=float)
                ),
                dtype=float,
            ).reshape(-1)
            score_part = standardized[["date", "instrument"]].copy()
            score_part["svm_score"] = (
                raw_score * int(bundle["decision_value_positive_sign"])
            )
            score_part = score_part.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["svm_score"]
            )
            valid_count = len(score_part)
            if not score_part.empty:
                result_parts.append(score_part)
        if show_progress and (
            position == 1
            or position % int(progress_every) == 0
            or position == len(targets)
        ):
            _render_progress(
                "固定模型推理",
                position,
                len(targets),
                started,
                f"当前 {date:%Y-%m-%d}，有效股票 {valid_count}",
            )
    result = (
        pd.concat(result_parts, ignore_index=True)
        if result_parts else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    if show_progress:
        print()
    return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)


FACTOR = {
    "name": "svm_score",
    "func": calc_svm_score,
    "category": "machine_learning",
    "direction": 1,
    "description": "固定模型支持向量机评分因子；分数越高越接近训练标签中的未来超额收益高组。",
    "formula": "svm_score 是冻结 SVM 对经过同一预处理的特征向量输出的、朝 +1 类方向的决策边界 margin。",
    "input_schema": {
        "required": {
            "date": {"dtype": "datetime64[ns]", "frequency": "daily", "meaning": "目标因子截面日期。"},
            "instrument": {"dtype": "string", "frequency": "daily", "meaning": "证券唯一标识。"},
            "dynamic_feature_columns": {"dtype": "float", "frequency": "daily", "meaning": "名称和顺序由 model_bundle.feature_spec 决定。"},
        },
        "conditional": {
            "industry": {"dtype": "string", "frequency": "daily", "meaning": "行业中位数填补或行业中性化时使用。", "required_when": {"preprocessing_config": "uses_industry"}},
            "total_market_cap": {"dtype": "float", "frequency": "daily", "meaning": "市值中性化时使用。", "required_when": {"preprocessing_config": "neutralize_size_industry"}},
        },
    },
    "parameters": {
        "target_dates": {"default": None, "accepted_values": "日期或日期序列", "effect": "指定输出截面。", "changes_data_requirements": False},
        "as_of_date": {"default": None, "accepted_values": "日期", "effect": "推理可使用的信息截止日。", "changes_data_requirements": False},
        "universe": {"default": None, "accepted_values": "all_a/custom/index 配置", "effect": "限制目标日输出的点时股票池。", "changes_data_requirements": True},
        "universe_panel": {"default": None, "accepted_values": "date+instrument 历史成员面板", "effect": "指数股票池的历史成员输入。", "changes_data_requirements": True},
        "model_bundle": {"default": None, "accepted_values": "完整冻结模型包", "effect": "提供内存模型包。", "changes_data_requirements": True},
        "model_artifact_dir": {"default": None, "accepted_values": "模型文件或目录路径", "effect": "加载受信任的固定模型包。", "changes_data_requirements": False},
        "show_progress": {"default": True, "accepted_values": "bool", "effect": "显示推理进度。", "changes_data_requirements": False},
        "progress_every": {"default": 20, "accepted_values": "正整数", "effect": "推理进度刷新频率。", "changes_data_requirements": False},
    },
    "data_window": {
        "lookback_trading_days": 0,
        "requires_target_date_data": True,
        "minimum_history_observations": 1,
        "preheating_required": False,
        "insufficient_window_behavior": "缺失模型所需特征的股票不输出 svm_score。",
    },
    "output_schema": {
        "date": {"dtype": "datetime64[ns]", "meaning": "目标截面日期。"},
        "instrument": {"dtype": "string", "meaning": "证券唯一标识。"},
        "svm_score": {"dtype": "float64", "meaning": "朝训练正类 +1 方向的 SVM 决策分数。"},
    },
    "usage_notes": [
        "训练与推理均要求显式 universe；指数股票池必须逐日期提供历史成员。",
        "训练函数不直接查询数据，调用方必须提供点时特征、开盘价和交易日历。",
        "模型包只能由受信任来源加载；joblib/pickle 不适合不可信文件。",
    ],
    "pit_notes": [
        "训练标签从 T+1 开盘到 T+1+H 开盘；training_anchor_date 必须晚于标签结束日。",
        "验证与最终测试必须按时间切分；测试期不参与超参数选择。",
        "推理不构造未来标签，也不根据样本外表现翻转分数。",
    ],
    "status": "research",
    "version": "0.1.0",
}


FACTOR_INFO = """# svm_score

固定模型的支持向量机选股评分因子。训练函数 ``train_svm_model`` 只应在研究期
调用；``calc_svm_score`` 只做推理。高斯核、线性核、多项式核和 Sigmoid 核均受
支持。模型分数是指向训练标签 +1 类的决策边界 margin，数值越高代表模型越倾向于
预测该股票属于未来超额收益前分位组。
"""
