# -*- coding: utf-8 -*-
"""GAN-GRU 机器学习因子。

因子计算只消费已经由 loader 准备好的依赖因子数据。滚动训练由本文件中的
模型状态提供器负责，并通过 loader 按各依赖因子自己的预热窗口读取数据。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import re
import threading
import time
import uuid
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["date", "instrument", "gan_gru_score"]
_RESERVED_CHILD_PARAMS = {
    "data",
    "domain_data",
    "target_dates",
    "as_of_date",
    "show_progress",
    "progress_every",
}


DEFAULT_FEATURE_SPEC = (
    {"factor_name": "open_gap", "params": {}, "feature_name": "open_gap"},
    {
        "factor_name": "intraday_return",
        "params": {},
        "feature_name": "intraday_return",
    },
    {
        "factor_name": "high_relative_return",
        "params": {},
        "feature_name": "high_relative_return",
    },
    {
        "factor_name": "low_relative_return",
        "params": {},
        "feature_name": "low_relative_return",
    },
    {
        "factor_name": "volume_relative_ma_nd",
        "params": {"window": 20},
        "feature_name": "volume_relative_ma_20d",
    },
    {
        "factor_name": "amount_relative_ma_nd",
        "params": {"window": 20},
        "feature_name": "amount_relative_ma_20d",
    },
    {
        "factor_name": "return_nd",
        "params": {"window": 1},
        "feature_name": "return_1d",
    },
    {
        "factor_name": "return_nd",
        "params": {"window": 5},
        "feature_name": "return_5d",
    },
    {
        "factor_name": "return_nd",
        "params": {"window": 10},
        "feature_name": "return_10d",
    },
    {
        "factor_name": "return_nd",
        "params": {"window": 20},
        "feature_name": "return_20d",
    },
    {
        "factor_name": "price_ma_distance_nd",
        "params": {"window": 5},
        "feature_name": "price_ma_distance_5d",
    },
    {
        "factor_name": "price_ma_distance_nd",
        "params": {"window": 10},
        "feature_name": "price_ma_distance_10d",
    },
    {
        "factor_name": "price_ma_distance_nd",
        "params": {"window": 20},
        "feature_name": "price_ma_distance_20d",
    },
    {
        "factor_name": "volume_ma_ratio_nd",
        "params": {"short_window": 5, "long_window": 20},
        "feature_name": "volume_ma_5d_to_20d",
    },
    {
        "factor_name": "return_std_nd",
        "params": {"window": 5},
        "feature_name": "return_std_5d",
    },
    {
        "factor_name": "return_std_nd",
        "params": {"window": 20},
        "feature_name": "return_std_20d",
    },
    {
        "factor_name": "rsi_nd",
        "params": {"window": 14},
        "feature_name": "rsi_14d",
    },
    {
        "factor_name": "macd_hist_relative",
        "params": {
            "fast_window": 12,
            "slow_window": 26,
            "signal_window": 9,
            "warmup_multiplier": 5,
        },
        "feature_name": "macd_hist_relative",
    },
    {
        "factor_name": "williams_r_nd",
        "params": {"window": 14},
        "feature_name": "williams_r_14d",
    },
    {
        "factor_name": "volume_ratio_nd",
        "params": {"window": 5},
        "feature_name": "volume_ratio_5d",
    },
)


DEFAULT_TRAINING_CONFIG = {
    "sequence_length": 40,
    "training_window_days": 336,
    "retrain_interval_days": 84,
    "label_horizon_days": 5,
    "validation_ratio": 0.20,
    "purge_trading_days": 5,
    "minimum_validation_dates": 20,
    "minimum_rankic_stocks": 30,
    "minimum_training_samples": 1000,
    "latent_dim": 10,
    "discriminator_hidden_size": 32,
    "lambda_reconstruction": 0.5,
    "gan_epochs": 4,
    "gru_max_epochs": 20,
    "early_stop_patience": 4,
    "rankic_min_delta": 1e-4,
    "batch_size": 1024,
    # 与研究稿一致：显式限制 PyTorch CPU 训练线程数，避免平台默认值过低。
    "cpu_threads": 4,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "gan_learning_rate": 0.0002,
    "gru_learning_rate": 0.001,
    "random_seed": 42,
}


class _FactorDataBundleLike(Protocol):
    """gan_gru_score 实际使用的数据容器最小接口。"""

    def get_dependency(self, dependency_name: str) -> Any:
        ...

    def get_dependency_target_dates(
        self,
        dependency_name: str,
        final_dates: Any,
    ) -> pd.DatetimeIndex:
        ...


class _ModelStateProvider(Protocol):
    """滚动模型状态提供器的调用约定。"""

    def __call__(
        self,
        *,
        target_date: Any,
        feature_spec: Any,
        instruments: Any,
        training_config: Any,
        show_progress: bool,
    ) -> Mapping:
        ...


def _normalize_dates(dates, field_name="target_dates"):
    if isinstance(dates, (str, pd.Timestamp, np.datetime64)):
        dates = [dates]
    if dates is None:
        raise ValueError(f"{field_name} 不能为空。")
    result = pd.DatetimeIndex(
        pd.to_datetime(list(dates), errors="raise")
    ).normalize().unique().sort_values()
    if len(result) == 0:
        raise ValueError(f"{field_name} 不能为空。")
    return result


def _print_training_progress(
    stage,
    started_at,
    completed=None,
    total=None,
    detail="",
):
    """在 BigQuant 终端使用单行刷新显示训练阶段进度。"""
    elapsed = time.perf_counter() - started_at
    progress_text = ""
    if completed is not None and total is not None and total > 0:
        ratio = min(max(float(completed) / float(total), 0.0), 1.0)
        remaining = 0.0 if completed <= 0 else elapsed * (1.0 - ratio) / ratio
        progress_text = (
            f" | {completed:,}/{total:,} ({ratio:.1%})"
            f" | 预计剩余 {remaining:.1f}s"
        )
    suffix = f" | {detail}" if detail else ""
    message = (
        f"\r[GAN-GRU训练] {stage}{progress_text}{suffix}"
        f" | 已耗时 {elapsed:.1f}s"
    )
    print(
        message.ljust(190),
        end="",
        flush=True,
    )


def _print_feature_progress(
    scope,
    feature_index,
    feature_total,
    feature_name,
    stage,
    started_at,
    *,
    completed_features=0,
    detail="",
):
    """输出 GAN-GRU 内部特征准备进度，不依赖子因子的日志实现。"""
    elapsed = time.perf_counter() - started_at
    parts = [
        f"[{scope}] 特征 {feature_index}/{feature_total}",
        f"当前 {feature_name}",
        stage,
    ]
    if completed_features > 0:
        ratio = completed_features / feature_total
        parts.append(f"总完成度 {ratio:.1%}")
        if completed_features < feature_total:
            remaining = elapsed / completed_features * (
                feature_total - completed_features
            )
            parts.append(f"预计剩余 {remaining:.1f}s")
    if detail:
        parts.append(str(detail))
    parts.append(f"已耗时 {elapsed:.1f}s")
    print("\r" + " | ".join(parts).ljust(220), end="", flush=True)


def _run_feature_stage_with_heartbeat(
    action,
    *,
    scope,
    feature_index,
    feature_total,
    feature_name,
    stage,
    started_at,
    show_progress,
    completed_features,
    detail="",
):
    """为不透明查询或单次因子计算提供两秒一次的“仍在运行”心跳。"""
    if not show_progress:
        return action()

    stop = threading.Event()

    def heartbeat():
        while not stop.wait(2.0):
            _print_feature_progress(
                scope,
                feature_index,
                feature_total,
                feature_name,
                f"{stage}（仍在运行）",
                started_at,
                completed_features=completed_features,
                detail=detail,
            )

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        return action()
    finally:
        stop.set()
        worker.join(timeout=2.1)


def _json_hash(value):
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("用于模型校验的配置必须可以被 JSON 序列化。") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_feature_spec(feature_spec):
    source = DEFAULT_FEATURE_SPEC if feature_spec is None else feature_spec
    if not isinstance(source, (list, tuple)) or not source:
        raise ValueError("feature_spec 必须是非空列表。")

    normalized = []
    feature_names = set()
    allowed_keys = {"factor_name", "params", "feature_name"}
    for position, item in enumerate(source, start=1):
        if not isinstance(item, Mapping):
            raise TypeError(f"feature_spec 第 {position} 项必须是字典。")
        unknown = sorted(set(item) - allowed_keys)
        if unknown:
            raise ValueError(
                f"feature_spec 第 {position} 项包含未知字段：{unknown}。"
            )
        factor_name = item.get("factor_name")
        feature_name = item.get("feature_name")
        params = item.get("params", {})
        if not isinstance(factor_name, str) or not factor_name.strip():
            raise ValueError(f"feature_spec 第 {position} 项缺少 factor_name。")
        if factor_name.strip() == "gan_gru_score":
            raise ValueError("gan_gru_score 不能依赖自身。")
        if not isinstance(feature_name, str) or not feature_name.strip():
            raise ValueError(f"feature_spec 第 {position} 项缺少 feature_name。")
        if not isinstance(params, Mapping):
            raise TypeError(
                f"特征 {feature_name!r} 的 params 必须是字典。"
            )
        reserved = sorted(set(params) & _RESERVED_CHILD_PARAMS)
        if reserved:
            raise ValueError(
                f"特征 {feature_name!r} 不得覆盖系统参数：{reserved}。"
            )
        feature_name = feature_name.strip()
        if feature_name in feature_names:
            raise ValueError(f"feature_name 重复：{feature_name!r}。")
        normalized.append(
            {
                "factor_name": factor_name.strip(),
                "params": dict(params),
                "feature_name": feature_name,
            }
        )
        feature_names.add(feature_name)

    _json_hash(normalized)
    return normalized


def _normalize_training_config(training_config):
    if training_config is None:
        supplied = {}
    elif isinstance(training_config, Mapping):
        supplied = dict(training_config)
    else:
        raise TypeError("training_config 必须是字典或 None。")

    unknown = sorted(set(supplied) - set(DEFAULT_TRAINING_CONFIG))
    if unknown:
        raise ValueError(f"training_config 包含未知参数：{unknown}。")
    config = dict(DEFAULT_TRAINING_CONFIG)
    config.update(supplied)

    positive_integer_fields = (
        "sequence_length",
        "training_window_days",
        "retrain_interval_days",
        "label_horizon_days",
        "minimum_validation_dates",
        "minimum_rankic_stocks",
        "minimum_training_samples",
        "latent_dim",
        "discriminator_hidden_size",
        "gan_epochs",
        "gru_max_epochs",
        "early_stop_patience",
        "batch_size",
        "cpu_threads",
        "hidden_size",
        "num_layers",
    )
    for name in positive_integer_fields:
        value = config[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or int(value) < 1:
            raise ValueError(f"training_config[{name!r}] 必须是正整数。")
        config[name] = int(value)

    purge = config["purge_trading_days"]
    if isinstance(purge, (bool, np.bool_)) or not isinstance(
        purge, (int, np.integer)
    ) or int(purge) < 0:
        raise ValueError("purge_trading_days 必须是非负整数。")
    config["purge_trading_days"] = int(purge)

    ratio = float(config["validation_ratio"])
    if not 0.0 < ratio < 0.5:
        raise ValueError("validation_ratio 必须位于 (0, 0.5) 内。")
    config["validation_ratio"] = ratio

    dropout = float(config["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout 必须位于 [0, 1) 内。")
    config["dropout"] = dropout

    for name in (
        "lambda_reconstruction",
        "rankic_min_delta",
        "gan_learning_rate",
        "gru_learning_rate",
    ):
        value = float(config[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"training_config[{name!r}] 必须为正数。")
        config[name] = value

    seed = config["random_seed"]
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("random_seed 必须是整数。")
    config["random_seed"] = int(seed)
    return config


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("gan_gru_score 需要 BigQuant 环境中的 PyTorch。") from exc
    return torch, nn, optim, DataLoader, TensorDataset


def _build_gru_model(model_config):
    torch, nn, _, _, _ = _require_torch()

    class StockGRU(nn.Module):
        def __init__(self):
            super().__init__()
            layer_count = int(model_config["num_layers"])
            self.gru = nn.GRU(
                int(model_config["input_size"]),
                int(model_config["hidden_size"]),
                layer_count,
                batch_first=True,
                dropout=(
                    float(model_config["dropout"])
                    if layer_count > 1
                    else 0.0
                ),
            )
            self.fc = nn.Linear(int(model_config["hidden_size"]), 1)

        def forward(self, values):
            return self.fc(self.gru(values)[1][-1])

    return StockGRU(), torch


def _normalize_model_config(model_config, feature_count):
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config 必须是字典。")
    required = {
        "input_size",
        "sequence_length",
        "hidden_size",
        "num_layers",
        "dropout",
    }
    missing = sorted(required - set(model_config))
    if missing:
        raise ValueError(f"model_config 缺少字段：{missing}。")
    normalized = {
        "input_size": int(model_config["input_size"]),
        "sequence_length": int(model_config["sequence_length"]),
        "hidden_size": int(model_config["hidden_size"]),
        "num_layers": int(model_config["num_layers"]),
        "dropout": float(model_config["dropout"]),
    }
    if normalized["input_size"] != feature_count:
        raise ValueError(
            "model_config.input_size 与 feature_spec 特征数量不一致。"
        )
    if min(
        normalized["input_size"],
        normalized["sequence_length"],
        normalized["hidden_size"],
        normalized["num_layers"],
    ) < 1:
        raise ValueError("model_config 中的整数参数必须为正整数。")
    if not 0.0 <= normalized["dropout"] < 1.0:
        raise ValueError("model_config.dropout 必须位于 [0, 1) 内。")
    return normalized


def _validate_model_bundle(model_bundle):
    if not isinstance(model_bundle, Mapping):
        raise TypeError("模型包必须是字典。")
    required = {
        "model_state_dict",
        "scaler",
        "model_config",
        "feature_spec",
        "feature_schema_hash",
    }
    missing = sorted(required - set(model_bundle))
    if missing:
        raise ValueError(f"模型包缺少字段：{missing}。")

    feature_spec = _normalize_feature_spec(model_bundle["feature_spec"])
    expected_hash = _json_hash(feature_spec)
    if model_bundle["feature_schema_hash"] != expected_hash:
        raise ValueError("模型包的 feature_schema_hash 校验失败。")
    model_config = _normalize_model_config(
        model_bundle["model_config"],
        len(feature_spec),
    )

    scaler = model_bundle["scaler"]
    if not isinstance(scaler, Mapping):
        raise TypeError("模型包 scaler 必须是字典。")
    mean = np.asarray(scaler.get("mean"), dtype=np.float32).reshape(-1)
    std = np.asarray(scaler.get("std"), dtype=np.float32).reshape(-1)
    if len(mean) != len(feature_spec) or len(std) != len(feature_spec):
        raise ValueError("scaler 维度与 feature_spec 不一致。")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("scaler 包含非有限值。")
    if (std <= 0).any():
        raise ValueError("scaler.std 必须全部大于 0。")

    state_dict = model_bundle["model_state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("model_state_dict 必须是非空映射。")
    model, _ = _build_gru_model(model_config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError("model_state_dict 与 model_config 不匹配。") from exc

    return {
        "model_state_dict": dict(state_dict),
        "scaler": {"mean": mean, "std": std},
        "model_config": model_config,
        "feature_spec": feature_spec,
        "feature_schema_hash": expected_hash,
        "training_metadata": copy.deepcopy(
            model_bundle.get("training_metadata", {})
        ),
    }


def _resolve_runtime_mode(
    feature_spec,
    fixed_model_bundle,
    model_state_provider,
    training_config,
):
    uses_fixed = fixed_model_bundle is not None
    uses_provider = model_state_provider is not None
    if uses_fixed == uses_provider:
        raise ValueError(
            "fixed_model_bundle 与 model_state_provider 必须且只能提供一个。"
        )

    if uses_fixed:
        if feature_spec is not None:
            raise ValueError(
                "固定模型模式不得另传 feature_spec；"
                "特征定义必须来自 fixed_model_bundle。"
            )
        if training_config is not None:
            raise ValueError("固定模型模式不接受 training_config。")
        bundle = _validate_model_bundle(fixed_model_bundle)
        return bundle["feature_spec"], bundle["model_config"][
            "sequence_length"
        ]

    if not callable(model_state_provider):
        raise TypeError("model_state_provider 必须是可调用对象。")
    normalized_features = _normalize_feature_spec(feature_spec)
    config = _normalize_training_config(training_config)
    return normalized_features, config["sequence_length"]


def _resolve_gan_gru_dependencies(params):
    feature_spec, sequence_length = _resolve_runtime_mode(
        params.get("feature_spec"),
        params.get("fixed_model_bundle"),
        params.get("model_state_provider"),
        params.get("training_config"),
    )
    return {
        "sequence_length": sequence_length,
        "items": [
            {
                "factor_name": item["factor_name"],
                "factor_params": dict(item["params"]),
                "feature_name": item["feature_name"],
            }
            for item in feature_spec
        ],
    }


def _factor_output_column(factor_name):
    from factor_lib.common.data_adapters.bigquant_adapters.loader import (
        get_factor_metadata,
    )

    metadata = get_factor_metadata(factor_name)
    output_schema = metadata.get("output_schema")
    if not isinstance(output_schema, Mapping):
        raise ValueError(f"因子 {factor_name!r} 缺少 output_schema。")
    columns = [
        name
        for name in output_schema
        if name not in {"date", "instrument"}
    ]
    if len(columns) != 1:
        raise ValueError(
            f"依赖因子 {factor_name!r} 必须且只能有一个因子值列。"
        )
    return columns[0]


def _calculate_feature_panel(
    domain_data: _FactorDataBundleLike | None,
    feature_spec,
    final_dates,
    show_progress=False,
):
    from factor_lib.factor_hub.get_factor import get_factor

    if domain_data is None:
        raise TypeError(
            "gan_gru_score 必须接收 loader 返回的 FactorDataBundle。"
        )
    if not callable(getattr(domain_data, "get_dependency", None)) or not callable(
        getattr(domain_data, "get_dependency_target_dates", None)
    ):
        raise TypeError(
            "domain_data 缺少 FactorDataBundle 的依赖读取接口。"
        )
    data_bundle = domain_data

    pieces = []
    expected_dates = None
    total = len(feature_spec)
    started_at = time.perf_counter()
    try:
        for index, item in enumerate(feature_spec, start=1):
            feature_name = item["feature_name"]
            child_bundle = data_bundle.get_dependency(feature_name)
            child_target_dates = data_bundle.get_dependency_target_dates(
                feature_name,
                final_dates,
            )
            if expected_dates is None:
                expected_dates = child_target_dates
            elif not expected_dates.equals(child_target_dates):
                raise ValueError(
                    f"依赖 {feature_name!r} 的序列日期与其他特征不一致。"
                )

            if show_progress:
                _print_feature_progress(
                    "gan_gru_score",
                    index,
                    total,
                    feature_name,
                    "准备预存的依赖数据",
                    started_at,
                    completed_features=index - 1,
                    detail=f"{len(child_target_dates)} 个目标日",
                )

            factor_result = _run_feature_stage_with_heartbeat(
                lambda: get_factor(
                    item["factor_name"],
                    child_bundle,
                    target_dates=child_target_dates,
                    as_of_date=final_dates.max(),
                    show_progress=show_progress,
                    progress_every=1,
                    **item["params"],
                ),
                scope="gan_gru_score",
                feature_index=index,
                feature_total=total,
                feature_name=feature_name,
                stage="计算因子截面",
                started_at=started_at,
                show_progress=show_progress,
                completed_features=index - 1,
                detail=f"{len(child_target_dates)} 个目标日",
            )
            value_column = _factor_output_column(item["factor_name"])
            required = {"date", "instrument", value_column}
            missing = sorted(required - set(factor_result.columns))
            if missing:
                raise ValueError(
                    f"依赖因子 {item['factor_name']!r} 输出缺少：{missing}。"
                )
            piece = factor_result[
                ["date", "instrument", value_column]
            ].copy()
            piece.rename(
                columns={value_column: feature_name},
                inplace=True,
            )
            pieces.append(piece)

            if show_progress:
                _print_feature_progress(
                    "gan_gru_score",
                    index,
                    total,
                    feature_name,
                    "因子计算完成，已加入特征面板",
                    started_at,
                    completed_features=index,
                    detail=f"{len(piece):,} 条因子记录",
                )

        panel = pieces[0]
        merge_total = max(1, len(pieces) - 1)
        for merge_index, piece in enumerate(pieces[1:], start=1):
            if show_progress:
                _print_feature_progress(
                    "gan_gru_score",
                    merge_index,
                    merge_total,
                    str(piece.columns[-1]),
                    "合并并校验特征面板",
                    started_at,
                    completed_features=total,
                    detail=f"合并 {merge_index}/{merge_total}",
                )
            panel = panel.merge(
                piece,
                on=["date", "instrument"],
                how="inner",
                validate="one_to_one",
            )
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        return panel.sort_values(
            ["instrument", "date"], kind="mergesort"
        ).reset_index(drop=True)
    finally:
        if show_progress:
            print()


def _build_prediction_sequences(
    feature_panel,
    feature_names,
    sequence_dates,
):
    section = feature_panel.loc[
        feature_panel["date"].isin(sequence_dates)
    ]
    if section.empty:
        return [], np.empty(
            (0, len(sequence_dates), len(feature_names)),
            dtype=np.float32,
        )

    instruments = []
    sequences = []
    for instrument, group in section.groupby("instrument", sort=False):
        values = (
            group.set_index("date")[feature_names]
            .reindex(sequence_dates)
            .to_numpy(dtype=np.float32)
        )
        if values.shape != (len(sequence_dates), len(feature_names)):
            continue
        if not np.isfinite(values).all():
            continue
        instruments.append(str(instrument))
        sequences.append(values)
    if not sequences:
        return [], np.empty(
            (0, len(sequence_dates), len(feature_names)),
            dtype=np.float32,
        )
    return instruments, np.stack(sequences).astype(np.float32, copy=False)


def _predict_with_bundle(model_bundle, sequences):
    bundle = _validate_model_bundle(model_bundle)
    expected_shape = (
        bundle["model_config"]["sequence_length"],
        bundle["model_config"]["input_size"],
    )
    if sequences.ndim != 3 or tuple(sequences.shape[1:]) != expected_shape:
        raise ValueError(
            f"推理序列形状应为 (样本数, {expected_shape[0]}, "
            f"{expected_shape[1]})，实际为 {sequences.shape}。"
        )

    model, torch = _build_gru_model(bundle["model_config"])
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    model.eval()
    standardized = (
        sequences - bundle["scaler"]["mean"].reshape(1, 1, -1)
    ) / bundle["scaler"]["std"].reshape(1, 1, -1)
    with torch.no_grad():
        scores = model(
            torch.tensor(standardized, dtype=torch.float32)
        ).cpu().numpy().reshape(-1)
    return scores.astype(float, copy=False)


def calc_gan_gru_score(
    data,
    target_dates=None,
    as_of_date=None,
    feature_spec=None,
    fixed_model_bundle: Mapping | None = None,
    model_state_provider: _ModelStateProvider | None = None,
    training_config: Mapping | None = None,
    domain_data: _FactorDataBundleLike | None = None,
    show_progress=False,
    progress_every=20,
):
    """使用固定模型包或滚动模型状态计算 GAN-GRU 截面分数。"""
    if not isinstance(progress_every, int) or progress_every <= 0:
        raise ValueError("progress_every 必须是正整数。")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame。")
    missing = sorted({"date", "instrument"} - set(data.columns))
    if missing:
        raise ValueError(f"gan_gru_score 输入缺少字段：{missing}。")

    targets = _normalize_dates(target_dates)
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date).normalize()
        if (targets > cutoff).any():
            raise ValueError("target_dates 不得晚于 as_of_date。")
    if domain_data is None:
        raise TypeError(
            "gan_gru_score 必须接收 loader 返回的 FactorDataBundle。"
        )
    data_bundle = domain_data

    normalized_features, sequence_length = _resolve_runtime_mode(
        feature_spec,
        fixed_model_bundle,
        model_state_provider,
        training_config,
    )
    feature_names = [item["feature_name"] for item in normalized_features]
    feature_hash = _json_hash(normalized_features)
    feature_panel = _calculate_feature_panel(
        data_bundle,
        normalized_features,
        targets,
        show_progress=show_progress,
    )

    results = []
    rolling_config = (
        None
        if fixed_model_bundle is not None
        else _normalize_training_config(training_config)
    )
    for position, target_date in enumerate(targets, start=1):
        reference_feature = normalized_features[0]["feature_name"]
        sequence_dates = data_bundle.get_dependency_target_dates(
            reference_feature,
            [target_date],
        )
        if len(sequence_dates) != sequence_length:
            raise ValueError(
                f"{target_date:%Y-%m-%d} 的推理序列应有 {sequence_length} 日，"
                f"实际为 {len(sequence_dates)} 日。"
            )
        instruments, sequences = _build_prediction_sequences(
            feature_panel,
            feature_names,
            sequence_dates,
        )
        if not instruments:
            continue

        if fixed_model_bundle is not None:
            model_bundle = fixed_model_bundle
        else:
            provider = model_state_provider
            if provider is None:
                raise RuntimeError("滚动模式缺少 model_state_provider。")
            model_bundle = provider(
                target_date=target_date,
                feature_spec=normalized_features,
                instruments=instruments,
                training_config=rolling_config,
                show_progress=show_progress,
            )
        validated = _validate_model_bundle(model_bundle)
        if validated["feature_schema_hash"] != feature_hash:
            raise ValueError(
                "当前模型状态的特征组合或顺序与本次推理 feature_spec 不一致。"
            )
        if validated["model_config"]["sequence_length"] != sequence_length:
            raise ValueError("模型序列长度与本次依赖数据窗口不一致。")

        scores = _predict_with_bundle(validated, sequences)
        results.append(
            pd.DataFrame(
                {
                    "date": target_date,
                    "instrument": instruments,
                    "gan_gru_score": scores,
                }
            )
        )
        if show_progress and (
            position % progress_every == 0 or position == len(targets)
        ):
            print(
                "\r[gan_gru_score] "
                f"推理 {position}/{len(targets)}，"
                f"当前 {target_date:%Y-%m-%d}，有效股票 {len(instruments):,}"
                .ljust(150),
                end="",
                flush=True,
            )
    if show_progress:
        print()
    if not results:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(results, ignore_index=True).sort_values(
        ["date", "instrument"], kind="mergesort"
    ).reset_index(drop=True)


def _calendar_for_training(anchor_date, required_days):
    from factor_lib.common.data_adapters.bigquant_adapters.loader import (
        load_trading_dates,
    )

    anchor = pd.Timestamp(anchor_date).normalize()
    span_days = max(180, int(required_days) * 2 + 60)
    for _ in range(6):
        start = anchor - pd.Timedelta(days=span_days)
        calendar = load_trading_dates(start, anchor)
        if len(calendar) >= required_days:
            return calendar
        span_days *= 2
    raise ValueError(
        f"截至 {anchor:%Y-%m-%d} 无法取得 {required_days} 个交易日。"
    )


def _load_training_features(
    feature_spec,
    feature_target_dates,
    calendar,
    instruments,
    show_progress,
):
    from factor_lib.common.data_adapters.bigquant_adapters.loader import (
        get_factor_data_requirements,
        load_factor_raw_data,
    )
    from factor_lib.factor_hub.get_factor import get_factor

    date_positions = {date: index for index, date in enumerate(calendar)}
    earliest_target_position = date_positions[feature_target_dates.min()]
    final_target_position = date_positions[feature_target_dates.max()]
    pieces = []
    started_at = time.perf_counter()
    total = len(feature_spec)
    try:
        for index, item in enumerate(feature_spec, start=1):
            requirements = get_factor_data_requirements(
                item["factor_name"],
                item["params"],
            )
            if requirements["dependencies"] is not None:
                raise NotImplementedError(
                    "GAN-GRU 当前训练特征只能使用无下级依赖的基础因子。"
                )
            lookback = requirements["data_window"][
                "lookback_trading_days"
            ]
            raw_start = earliest_target_position - lookback
            if raw_start < 0:
                raise ValueError(
                    f"训练特征 {item['feature_name']!r} 缺少预热数据。"
                )
            raw_dates = calendar[raw_start: final_target_position + 1]
            if show_progress:
                _print_feature_progress(
                    "GAN-GRU训练特征",
                    index,
                    total,
                    item["feature_name"],
                    "读取训练期原始数据",
                    started_at,
                    completed_features=index - 1,
                    detail=(
                        f"独立预热 {lookback} 日，"
                        f"原始日期 {len(raw_dates)} 个"
                    ),
                )
            raw_bundle = _run_feature_stage_with_heartbeat(
                lambda: load_factor_raw_data(
                    factor_name=item["factor_name"],
                    dates=raw_dates,
                    factor_params=requirements["resolved_factor_params"],
                    instruments=instruments,
                    show_progress=show_progress,
                ),
                scope="GAN-GRU训练特征",
                feature_index=index,
                feature_total=total,
                feature_name=item["feature_name"],
                stage="读取训练期原始数据",
                started_at=started_at,
                show_progress=show_progress,
                completed_features=index - 1,
                detail=(
                    f"独立预热 {lookback} 日，"
                    f"原始日期 {len(raw_dates)} 个"
                ),
            )
            values = _run_feature_stage_with_heartbeat(
                lambda: get_factor(
                    item["factor_name"],
                    raw_bundle,
                    target_dates=feature_target_dates,
                    as_of_date=feature_target_dates.max(),
                    show_progress=show_progress,
                    progress_every=1,
                    **item["params"],
                ),
                scope="GAN-GRU训练特征",
                feature_index=index,
                feature_total=total,
                feature_name=item["feature_name"],
                stage="计算训练期因子值",
                started_at=started_at,
                show_progress=show_progress,
                completed_features=index - 1,
                detail=f"{len(feature_target_dates)} 个目标日",
            )
            value_column = _factor_output_column(item["factor_name"])
            piece = values[["date", "instrument", value_column]].copy()
            piece.rename(
                columns={value_column: item["feature_name"]},
                inplace=True,
            )
            pieces.append(piece)

            if show_progress:
                _print_feature_progress(
                    "GAN-GRU训练特征",
                    index,
                    total,
                    item["feature_name"],
                    "训练期特征计算完成",
                    started_at,
                    completed_features=index,
                    detail=f"独立预热 {lookback} 日，{len(piece):,} 条记录",
                )

        panel = pieces[0]
        merge_total = max(1, len(pieces) - 1)
        for merge_index, piece in enumerate(pieces[1:], start=1):
            if show_progress:
                _print_feature_progress(
                    "GAN-GRU训练特征",
                    merge_index,
                    merge_total,
                    str(piece.columns[-1]),
                    "合并并校验训练特征面板",
                    started_at,
                    completed_features=total,
                    detail=f"合并 {merge_index}/{merge_total}",
                )
            panel = panel.merge(
                piece,
                on=["date", "instrument"],
                how="inner",
                validate="one_to_one",
            )
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        return panel.sort_values(
            ["instrument", "date"], kind="mergesort"
        ).reset_index(drop=True)
    finally:
        if show_progress:
            print()


def _load_forward_labels(
    sample_dates,
    calendar,
    instruments,
    horizon,
    show_progress=False,
):
    from factor_lib.common.data_adapters.bigquant_adapters.daily import (
        load_daily_raw_data,
    )

    started_at = time.perf_counter()
    try:
        positions = {date: index for index, date in enumerate(calendar)}
        pairs = []
        required_dates = set()
        for sample_date in sample_dates:
            future_date = calendar[positions[sample_date] + horizon]
            pairs.append((sample_date, future_date))
            required_dates.add(sample_date)
            required_dates.add(future_date)

        if show_progress:
            _print_training_progress(
                "读取未来收益标签收盘价",
                started_at,
                detail=(
                    f"样本日 {len(sample_dates):,} 个，查询日 {len(required_dates):,} 个，"
                    f"股票 {len(instruments):,} 只"
                ),
            )
        close_panel = load_daily_raw_data(
            standard_fields=["close"],
            dates=sorted(required_dates),
            instruments=instruments,
            show_progress=False,
        )
        if show_progress:
            _print_training_progress(
                "对齐未来收益标签",
                started_at,
                detail=f"收盘价记录 {len(close_panel):,} 行",
            )
        close_panel = close_panel[["date", "instrument", "close"]].copy()
        close_panel["date"] = pd.to_datetime(close_panel["date"]).dt.normalize()
        start_close = close_panel.rename(columns={"close": "start_close"})
        pair_frame = pd.DataFrame(
            pairs,
            columns=["date", "label_end_date"],
        )
        labels = start_close.merge(pair_frame, on="date", how="inner")
        end_close = close_panel.rename(
            columns={"date": "label_end_date", "close": "end_close"}
        )
        labels = labels.merge(
            end_close,
            on=["label_end_date", "instrument"],
            how="inner",
            validate="many_to_one",
        )
        labels["target"] = labels["end_close"] / labels["start_close"] - 1.0
        labels.replace([np.inf, -np.inf], np.nan, inplace=True)
        result = labels.dropna(subset=["target"])[
            ["date", "instrument", "target"]
        ]
        if show_progress:
            _print_training_progress(
                "未来收益标签构造完成",
                started_at,
                completed=len(result),
                total=len(result),
                detail=f"完整标签 {len(result):,} 条",
            )
        return result
    finally:
        if show_progress:
            print()


def _build_training_samples(
    feature_panel,
    labels,
    feature_names,
    feature_dates,
    sample_dates,
    sequence_length,
    show_progress=False,
    progress_every=20,
):
    feature_positions = {
        date: index for index, date in enumerate(feature_dates)
    }
    label_lookup = labels.set_index(["date", "instrument"])["target"]
    sample_set = set(sample_dates)
    x_values = []
    y_values = []
    sample_date_values = []

    if not isinstance(progress_every, int) or progress_every <= 0:
        raise ValueError("progress_every 必须是正整数。")
    started_at = time.perf_counter()
    groups = feature_panel.groupby("instrument", sort=False)
    total_instruments = int(feature_panel["instrument"].nunique())
    try:
        if show_progress:
            _print_training_progress(
                "构造时序训练样本",
                started_at,
                completed=0,
                total=total_instruments,
                detail=f"序列长度 {sequence_length} 日",
            )
        for position, (instrument, group) in enumerate(groups, start=1):
            matrix = (
                group.set_index("date")[feature_names]
                .reindex(feature_dates)
                .to_numpy(dtype=np.float32)
            )
            for sample_date in sample_dates:
                label_key = (sample_date, instrument)
                if label_key not in label_lookup.index:
                    continue
                end = feature_positions[sample_date] + 1
                start = end - sequence_length
                if start < 0:
                    continue
                sequence = matrix[start:end]
                if not np.isfinite(sequence).all():
                    continue
                target = float(label_lookup.loc[label_key])
                if not np.isfinite(target) or sample_date not in sample_set:
                    continue
                x_values.append(sequence)
                y_values.append(target)
                sample_date_values.append(sample_date)

            if show_progress and (
                position % progress_every == 0 or position == total_instruments
            ):
                _print_training_progress(
                    "构造时序训练样本",
                    started_at,
                    completed=position,
                    total=total_instruments,
                    detail=(
                        f"当前 {instrument}，有效样本 {len(x_values):,} 条"
                    ),
                )

        if not x_values:
            raise ValueError("未构造出有效的 GAN-GRU 训练样本。")
        result = (
            np.stack(x_values).astype(np.float32, copy=False),
            np.asarray(y_values, dtype=np.float32).reshape(-1, 1),
            pd.DatetimeIndex(sample_date_values),
        )
        if show_progress:
            _print_training_progress(
                "时序训练样本构造完成",
                started_at,
                completed=total_instruments,
                total=total_instruments,
                detail=f"有效样本 {len(x_values):,} 条",
            )
        return result
    finally:
        if show_progress:
            print()


def _rank_ic_by_date(
    dates,
    targets,
    predictions,
    minimum_stocks,
    show_progress=False,
    started_at=None,
    epoch=None,
    total_epochs=None,
):
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "target": np.asarray(targets).reshape(-1),
            "prediction": np.asarray(predictions).reshape(-1),
        }
    )
    values = []
    sections = list(frame.groupby("date", sort=True))
    section_total = len(sections)
    progress_started_at = started_at or time.perf_counter()
    for section_position, (date, group) in enumerate(sections, start=1):
        if show_progress:
            epoch_text = ""
            if epoch is not None and total_epochs is not None:
                epoch_text = f"GRU epoch {epoch}/{total_epochs} | "
            _print_training_progress(
                "验证 RankIC 截面计算",
                progress_started_at,
                completed=section_position - 1,
                total=section_total,
                detail=(
                    f"{epoch_text}截面 {section_position}/{section_total}，"
                    f"当前 {pd.Timestamp(date):%Y-%m-%d}"
                ),
            )
        group = group.dropna()
        if len(group) < minimum_stocks:
            continue
        target_rank = group["target"].rank(method="average").to_numpy()
        prediction_rank = group["prediction"].rank(
            method="average"
        ).to_numpy()
        if np.std(target_rank) == 0 or np.std(prediction_rank) == 0:
            continue
        value = float(np.corrcoef(target_rank, prediction_rank)[0, 1])
        if np.isfinite(value):
            values.append(value)
    if show_progress:
        _print_training_progress(
            "验证 RankIC 截面计算完成",
            progress_started_at,
            completed=section_total,
            total=section_total,
            detail=f"有效截面 {len(values)} 个",
        )
    if not values:
        return np.nan, 0
    return float(np.mean(values)), len(values)


def train_gan_gru_model(
    anchor_date,
    feature_spec=None,
    instruments=None,
    training_config=None,
    training_start_date=None,
    training_end_date=None,
    show_progress=False,
    progress_every=20,
):
    """训练一次 GAN-GRU 并返回可直接用于推理的完整模型包。

    未提供固定训练起止日时，样本窗口自动截至 anchor_date 之前已经完整
    实现标签的最后一个交易日。固定区间模式要求两端同时提供，且标签结束日
    仍不得晚于 anchor_date。
    """
    normalized_features = _normalize_feature_spec(feature_spec)
    config = _normalize_training_config(training_config)
    if not isinstance(progress_every, int) or progress_every <= 0:
        raise ValueError("progress_every 必须是正整数。")
    training_started_at = time.perf_counter()
    if instruments is None:
        raise ValueError("训练必须显式提供 instruments，不能隐式扩大股票池。")
    instruments = sorted({str(item) for item in instruments if str(item)})
    if not instruments:
        raise ValueError("训练股票列表不能为空。")

    uses_fixed_dates = (
        training_start_date is not None or training_end_date is not None
    )
    if uses_fixed_dates and (
        training_start_date is None or training_end_date is None
    ):
        raise ValueError(
            "固定训练区间必须同时提供 training_start_date 和 training_end_date。"
        )

    from factor_lib.common.data_adapters.bigquant_adapters.loader import (
        get_factor_data_requirements,
    )

    maximum_child_lookback = 0
    for item in normalized_features:
        requirements = get_factor_data_requirements(
            item["factor_name"], item["params"]
        )
        if requirements["dependencies"] is not None:
            raise NotImplementedError(
                "GAN-GRU 当前训练特征只能使用无下级依赖的基础因子。"
            )
        maximum_child_lookback = max(
            maximum_child_lookback,
            requirements["data_window"]["lookback_trading_days"],
        )

    required_days = (
        config["training_window_days"]
        + config["sequence_length"]
        + config["label_horizon_days"]
        + maximum_child_lookback
        + 10
    )
    if show_progress:
        _print_training_progress(
            "准备训练交易日历",
            training_started_at,
            detail=(
                f"锚点 {pd.Timestamp(anchor_date):%Y-%m-%d}，"
                f"特征 {len(normalized_features)} 个，股票 {len(instruments):,} 只"
            ),
        )
    if uses_fixed_dates:
        from factor_lib.common.data_adapters.bigquant_adapters.loader import (
            load_trading_dates,
        )

        requested_start = pd.Timestamp(training_start_date).normalize()
        anchor = pd.Timestamp(anchor_date).normalize()
        history_needed = (
            config["sequence_length"] - 1 + maximum_child_lookback
        )
        span_days = max(60, history_needed * 2 + 30)
        for _ in range(6):
            calendar = load_trading_dates(
                requested_start - pd.Timedelta(days=span_days),
                anchor,
            )
            if requested_start in calendar:
                start_position = int(calendar.get_loc(requested_start))
                if start_position >= history_needed:
                    break
            span_days *= 2
        else:
            raise ValueError("固定训练区间之前缺少足够的特征预热交易日。")
    else:
        calendar = _calendar_for_training(anchor_date, required_days)
    anchor_trading_date = calendar.max()
    horizon = config["label_horizon_days"]
    positions = {date: index for index, date in enumerate(calendar)}

    if uses_fixed_dates:
        start_sample = pd.Timestamp(training_start_date).normalize()
        end_sample = pd.Timestamp(training_end_date).normalize()
        if start_sample > end_sample:
            raise ValueError("training_start_date 不能晚于 training_end_date。")
        missing = pd.DatetimeIndex([start_sample, end_sample]).difference(
            calendar
        )
        if len(missing) > 0:
            raise ValueError("固定训练区间的起止日必须是交易日。")
        if positions[end_sample] + horizon >= len(calendar):
            raise ValueError(
                "training_end_date 的未来标签尚未在 anchor_date 前完整实现。"
            )
        sample_dates = calendar[
            positions[start_sample]: positions[end_sample] + 1
        ]
    else:
        end_position = len(calendar) - horizon - 1
        start_position = end_position - config["training_window_days"] + 1
        if start_position < 0:
            raise ValueError("交易日历不足以覆盖滚动训练窗口。")
        sample_dates = calendar[start_position: end_position + 1]

    sequence_start_position = (
        positions[sample_dates.min()] - config["sequence_length"] + 1
    )
    if sequence_start_position < 0:
        raise ValueError("训练序列缺少历史交易日。")
    feature_target_dates = calendar[
        sequence_start_position: positions[sample_dates.max()] + 1
    ]
    if show_progress:
        _print_training_progress(
            "训练窗口已确定",
            training_started_at,
            detail=(
                f"样本 {sample_dates.min():%Y-%m-%d} 至 "
                f"{sample_dates.max():%Y-%m-%d}（{len(sample_dates):,} 日），"
                f"特征序列 {len(feature_target_dates):,} 日"
            ),
        )

    feature_panel = _load_training_features(
        normalized_features,
        feature_target_dates,
        calendar,
        instruments,
        show_progress,
    )
    labels = _load_forward_labels(
        sample_dates,
        calendar,
        instruments,
        horizon,
        show_progress=show_progress,
    )
    feature_names = [item["feature_name"] for item in normalized_features]
    x_all, y_all, sample_date_values = _build_training_samples(
        feature_panel,
        labels,
        feature_names,
        feature_target_dates,
        sample_dates,
        config["sequence_length"],
        show_progress=show_progress,
        progress_every=progress_every,
    )
    if len(x_all) < config["minimum_training_samples"]:
        raise ValueError(
            f"有效训练样本仅 {len(x_all):,}，少于 minimum_training_samples="
            f"{config['minimum_training_samples']:,}。"
        )

    unique_dates = sample_date_values.unique().sort_values()
    validation_count = max(
        config["minimum_validation_dates"],
        int(np.ceil(len(unique_dates) * config["validation_ratio"])),
    )
    if validation_count >= len(unique_dates):
        raise ValueError("训练日期不足以划分时间顺序验证集。")
    validation_start_position = len(unique_dates) - validation_count
    train_end_position = (
        validation_start_position - config["purge_trading_days"] - 1
    )
    if train_end_position < 0:
        raise ValueError("purge_trading_days 过大，训练集为空。")
    train_end_date = unique_dates[train_end_position]
    validation_start_date = unique_dates[validation_start_position]
    train_mask = sample_date_values <= train_end_date
    validation_mask = sample_date_values >= validation_start_date
    if train_mask.sum() < config["minimum_training_samples"]:
        raise ValueError("时间切分后的训练样本数不足。")
    if validation_mask.sum() == 0:
        raise ValueError("时间切分后的验证集为空。")

    x_train = x_all[train_mask]
    y_train = y_all[train_mask]
    train_dates = sample_date_values[train_mask]
    x_validation = x_all[validation_mask]
    y_validation = y_all[validation_mask]
    validation_dates = sample_date_values[validation_mask]
    if show_progress:
        _print_training_progress(
            "完成时间顺序训练/验证切分",
            training_started_at,
            detail=(
                f"训练 {len(x_train):,} 条，验证 {len(x_validation):,} 条，"
                f"验证起始 {validation_start_date:%Y-%m-%d}"
            ),
        )

    mean = x_train.reshape(-1, len(feature_names)).mean(axis=0)
    std = x_train.reshape(-1, len(feature_names)).std(axis=0, ddof=0)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    mean = mean.astype(np.float32)
    x_train = ((x_train - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    x_validation = ((x_validation - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    if show_progress:
        _print_training_progress(
            "完成训练集标准化",
            training_started_at,
            detail=(
                f"特征维度 {len(feature_names)}，"
                f"序列长度 {config['sequence_length']}"
            ),
        )

    torch, nn, optim, DataLoader, TensorDataset = _require_torch()
    torch.set_num_threads(config["cpu_threads"])
    seed = config["random_seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Generator(nn.Module):
        def __init__(self, input_size, latent_dim):
            super().__init__()
            self.latent_dim = latent_dim
            self.fc_noise = nn.Linear(latent_dim, input_size)
            self.rnn = nn.GRU(
                input_size, input_size, num_layers=1, batch_first=True
            )

        def forward(self, values):
            batch_size, sequence_length, _ = values.size()
            noise = torch.randn(
                batch_size,
                sequence_length,
                self.latent_dim,
                device=values.device,
            )
            return self.rnn(values + self.fc_noise(noise))[0]

    class Discriminator(nn.Module):
        def __init__(self, sequence_length, input_size, hidden_size):
            super().__init__()
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers=1, batch_first=True
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * sequence_length, 1),
                nn.Sigmoid(),
            )

        def forward(self, values):
            output, _ = self.rnn(values)
            return self.fc(output.reshape(output.size(0), -1)).squeeze(-1)

    train_x_tensor = torch.tensor(x_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.float32)
    gan_loader = DataLoader(
        TensorDataset(train_x_tensor, train_y_tensor),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    generator = Generator(len(feature_names), config["latent_dim"]).to(device)
    discriminator = Discriminator(
        config["sequence_length"],
        len(feature_names),
        config["discriminator_hidden_size"],
    ).to(device)
    generator_optimizer = optim.Adam(
        generator.parameters(), lr=config["gan_learning_rate"]
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=config["gan_learning_rate"]
    )
    binary_loss = nn.BCELoss()

    started_at = time.perf_counter()
    try:
        if show_progress:
            _print_training_progress(
                "开始 GAN 训练",
                training_started_at,
                detail=(
                    f"{config['gan_epochs']} 个 epoch，"
                    f"批大小 {config['batch_size']}，设备 {device}，"
                    f"CPU线程 {config['cpu_threads']}"
                ),
            )
        gan_total_batches = len(gan_loader)
        gan_total_steps = config["gan_epochs"] * gan_total_batches * 3
        for epoch in range(1, config["gan_epochs"] + 1):
            generator.train()
            discriminator.train()
            for batch_position, (real_values, _) in enumerate(
                gan_loader,
                start=1,
            ):
                step_base = (
                    (epoch - 1) * gan_total_batches * 3
                    + (batch_position - 1) * 3
                )
                real_values = real_values.to(device)
                batch_size = real_values.size(0)
                real_labels = torch.ones(batch_size, device=device)
                fake_labels = torch.zeros(batch_size, device=device)

                if show_progress:
                    _print_training_progress(
                        "GAN 训练",
                        started_at,
                        completed=step_base,
                        total=gan_total_steps,
                        detail=(
                            f"epoch {epoch}/{config['gan_epochs']}，"
                            f"批次 {batch_position}/{gan_total_batches}，"
                            "批内 1/3：生成判别样本"
                        ),
                    )
                # 判别器阶段不训练生成器；避免无效构造生成器反向传播图。
                with torch.no_grad():
                    fake_values = generator(real_values)

                if show_progress:
                    _print_training_progress(
                        "GAN 训练",
                        started_at,
                        completed=step_base + 1,
                        total=gan_total_steps,
                        detail=(
                            f"epoch {epoch}/{config['gan_epochs']}，"
                            f"批次 {batch_position}/{gan_total_batches}，"
                            "批内 2/3：更新判别器"
                        ),
                    )
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_loss = binary_loss(
                    discriminator(real_values), real_labels
                ) + binary_loss(discriminator(fake_values), fake_labels)
                discriminator_loss.backward()
                discriminator_optimizer.step()

                if show_progress:
                    _print_training_progress(
                        "GAN 训练",
                        started_at,
                        completed=step_base + 2,
                        total=gan_total_steps,
                        detail=(
                            f"epoch {epoch}/{config['gan_epochs']}，"
                            f"批次 {batch_position}/{gan_total_batches}，"
                            "批内 3/3：更新生成器"
                        ),
                    )
                # 生成器阶段只需要经由判别器取得对输入的梯度，
                # 不需要计算或保留判别器参数梯度。
                discriminator_optimizer.zero_grad(set_to_none=True)
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                try:
                    generator_optimizer.zero_grad(set_to_none=True)
                    generated = generator(real_values)
                    generator_loss = binary_loss(
                        discriminator(generated), real_labels
                    ) + config["lambda_reconstruction"] * nn.functional.l1_loss(
                        generated, real_values
                    )
                    generator_loss.backward()
                    generator_optimizer.step()
                finally:
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(True)

                if show_progress:
                    _print_training_progress(
                        "GAN 训练",
                        started_at,
                        completed=step_base + 3,
                        total=gan_total_steps,
                        detail=(
                            f"epoch {epoch}/{config['gan_epochs']}，"
                            f"批次 {batch_position}/{gan_total_batches} 完成"
                        ),
                    )

        generator.eval()
        generated_parts = []
        generated_total_batches = int(
            np.ceil(len(train_x_tensor) / config["batch_size"])
        )
        with torch.no_grad():
            for batch_position, start in enumerate(
                range(0, len(train_x_tensor), config["batch_size"]),
                start=1,
            ):
                if show_progress:
                    _print_training_progress(
                        "生成 GAN 增强样本",
                        started_at,
                        completed=batch_position - 1,
                        total=generated_total_batches,
                        detail=(
                            f"批次 {batch_position}/{generated_total_batches}："
                            "生成中"
                        ),
                    )
                generated_parts.append(
                    generator(
                        train_x_tensor[
                            start: start + config["batch_size"]
                        ].to(device)
                    ).cpu()
                )
                if show_progress:
                    _print_training_progress(
                        "生成 GAN 增强样本",
                        started_at,
                        completed=batch_position,
                        total=generated_total_batches,
                        detail=(
                            f"批次 {batch_position}/{generated_total_batches}："
                            "完成"
                        ),
                    )
        generated_x = torch.cat(generated_parts, dim=0)
        if show_progress:
            _print_training_progress(
                "GAN 增强样本生成完成",
                training_started_at,
                detail=f"训练增强样本 {len(generated_x):,} 条",
            )

        model_config = {
            "input_size": len(feature_names),
            "sequence_length": config["sequence_length"],
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"],
        }
        gru_model, _ = _build_gru_model(model_config)
        gru_model.to(device)
        gru_optimizer = optim.Adam(
            gru_model.parameters(), lr=config["gru_learning_rate"]
        )
        mean_squared_error = nn.MSELoss()
        train_date_groups = {}
        for sample_index, date in enumerate(train_dates):
            train_date_groups.setdefault(date, []).append(sample_index)
        train_date_keys = list(train_date_groups)

        validation_x_tensor = torch.tensor(
            x_validation, dtype=torch.float32
        )
        validation_y_tensor = torch.tensor(
            y_validation, dtype=torch.float32
        )
        best_rank_ic = -np.inf
        best_state = None
        best_epoch = 0
        no_improvement = 0
        if show_progress:
            _print_training_progress(
                "开始 GRU 训练",
                training_started_at,
                detail=(
                    f"最多 {config['gru_max_epochs']} 个 epoch，"
                    f"早停耐心值 {config['early_stop_patience']}"
                ),
            )
        gru_batches_per_epoch = len(train_date_keys)
        gru_total_steps = config["gru_max_epochs"] * gru_batches_per_epoch * 3
        for epoch in range(1, config["gru_max_epochs"] + 1):
            gru_model.train()
            date_order = np.random.permutation(gru_batches_per_epoch)
            for batch_position, date_position in enumerate(
                date_order,
                start=1,
            ):
                step_base = (
                    (epoch - 1) * gru_batches_per_epoch * 3
                    + (batch_position - 1) * 3
                )
                date = train_date_keys[int(date_position)]
                indices = torch.tensor(
                    train_date_groups[date], dtype=torch.long
                )
                original = train_x_tensor.index_select(0, indices)
                synthetic = generated_x.index_select(0, indices)
                targets = train_y_tensor.index_select(0, indices)
                batch_x = torch.cat([original, synthetic], dim=0).to(device)
                batch_y = torch.cat([targets, targets], dim=0).to(device)

                if show_progress:
                    _print_training_progress(
                        "GRU 训练",
                        started_at,
                        completed=step_base,
                        total=gru_total_steps,
                        detail=(
                            f"epoch {epoch}/{config['gru_max_epochs']}，"
                            f"截面批次 {batch_position}/{gru_batches_per_epoch}，"
                            f"当前 {pd.Timestamp(date):%Y-%m-%d}，"
                            "批内 1/3：前向计算"
                        ),
                    )
                gru_optimizer.zero_grad(set_to_none=True)
                loss = mean_squared_error(gru_model(batch_x), batch_y)

                if show_progress:
                    _print_training_progress(
                        "GRU 训练",
                        started_at,
                        completed=step_base + 1,
                        total=gru_total_steps,
                        detail=(
                            f"epoch {epoch}/{config['gru_max_epochs']}，"
                            f"截面批次 {batch_position}/{gru_batches_per_epoch}，"
                            "批内 2/3：反向传播"
                        ),
                    )
                loss.backward()

                if show_progress:
                    _print_training_progress(
                        "GRU 训练",
                        started_at,
                        completed=step_base + 2,
                        total=gru_total_steps,
                        detail=(
                            f"epoch {epoch}/{config['gru_max_epochs']}，"
                            f"截面批次 {batch_position}/{gru_batches_per_epoch}，"
                            "批内 3/3：更新参数"
                        ),
                    )
                gru_optimizer.step()
                if show_progress:
                    _print_training_progress(
                        "GRU 训练",
                        started_at,
                        completed=step_base + 3,
                        total=gru_total_steps,
                        detail=(
                            f"epoch {epoch}/{config['gru_max_epochs']}，"
                            f"截面批次 {batch_position}/{gru_batches_per_epoch} 完成"
                        ),
                    )

            gru_model.eval()
            predictions = []
            validation_total_batches = int(
                np.ceil(len(validation_x_tensor) / config["batch_size"])
            )
            with torch.no_grad():
                for batch_position, start in enumerate(
                    range(0, len(validation_x_tensor), config["batch_size"]),
                    start=1,
                ):
                    if show_progress:
                        _print_training_progress(
                            "GRU 验证前向计算",
                            started_at,
                            completed=batch_position - 1,
                            total=validation_total_batches,
                            detail=(
                                f"epoch {epoch}/{config['gru_max_epochs']}，"
                                f"批次 {batch_position}/{validation_total_batches}："
                                "计算中"
                            ),
                        )
                    predictions.append(
                        gru_model(
                            validation_x_tensor[
                                start: start + config["batch_size"]
                            ].to(device)
                        ).cpu().numpy()
                    )
                    if show_progress:
                        _print_training_progress(
                            "GRU 验证前向计算",
                            started_at,
                            completed=batch_position,
                            total=validation_total_batches,
                            detail=(
                                f"epoch {epoch}/{config['gru_max_epochs']}，"
                                f"批次 {batch_position}/{validation_total_batches}："
                                "完成"
                            ),
                        )
            predictions = np.concatenate(predictions).reshape(-1)
            rank_ic, valid_sections = _rank_ic_by_date(
                validation_dates,
                validation_y_tensor.numpy(),
                predictions,
                config["minimum_rankic_stocks"],
                show_progress=show_progress,
                started_at=started_at,
                epoch=epoch,
                total_epochs=config["gru_max_epochs"],
            )
            improved = np.isfinite(rank_ic) and (
                rank_ic > best_rank_ic + config["rankic_min_delta"]
            )
            if improved:
                best_rank_ic = float(rank_ic)
                best_epoch = epoch
                no_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in gru_model.state_dict().items()
                }
            else:
                no_improvement += 1

            if show_progress:
                epoch_elapsed = time.perf_counter() - started_at
                epoch_remaining = (
                    epoch_elapsed
                    * (config["gru_max_epochs"] - epoch)
                    / epoch
                )
                print(
                    "\r[GAN-GRU训练] "
                    f"GRU epoch {epoch}/{config['gru_max_epochs']}，"
                    f"验证 RankIC={rank_ic:.6f}，有效截面={valid_sections}，"
                    f"最佳 epoch={best_epoch}，耗时 "
                    f"{epoch_elapsed:.1f}s，预计剩余 "
                    f"{epoch_remaining:.1f}s".ljust(190),
                    end="",
                    flush=True,
                )
            if no_improvement >= config["early_stop_patience"]:
                break

        if best_state is None:
            raise ValueError(
                "验证集未产生有效 RankIC，模型状态不会被保存或返回。"
            )
        bundle = {
            "model_state_dict": best_state,
            "scaler": {"mean": mean, "std": std},
            "model_config": model_config,
            "feature_spec": normalized_features,
            "feature_schema_hash": _json_hash(normalized_features),
            "training_metadata": {
                "anchor_date": anchor_trading_date.strftime("%Y-%m-%d"),
                "sample_start_date": sample_dates.min().strftime("%Y-%m-%d"),
                "sample_end_date": sample_dates.max().strftime("%Y-%m-%d"),
                "label_horizon_days": horizon,
                "training_sample_count": int(train_mask.sum()),
                "validation_sample_count": int(validation_mask.sum()),
                "validation_start_date": validation_start_date.strftime(
                    "%Y-%m-%d"
                ),
                "best_epoch": best_epoch,
                "best_validation_rank_ic": best_rank_ic,
                "training_config": config,
                "training_config_hash": _json_hash(config),
            },
        }
        if show_progress:
            _print_training_progress(
                "训练完成",
                training_started_at,
                detail=(
                    f"最优 epoch={best_epoch}，验证 RankIC={best_rank_ic:.6f}，"
                    f"总样本 {len(x_all):,} 条"
                ),
            )
        return _validate_model_bundle(bundle)
    finally:
        if show_progress:
            print()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _persist_model_bundle(run_directory, model_bundle, update_count):
    bundle = _validate_model_bundle(model_bundle)
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=False, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_model = run_directory / f"model.pth.tmp.{token}"
    temporary_scaler = run_directory / f"scaler.pkl.tmp.{token}"
    temporary_state = run_directory / f"model_state.json.tmp.{token}"
    final_model = run_directory / "model.pth"
    final_scaler = run_directory / "scaler.pkl"
    final_state = run_directory / "model_state.json"
    backup_paths = {
        final_model: run_directory / f"model.pth.bak.{token}",
        final_scaler: run_directory / f"scaler.pkl.bak.{token}",
        final_state: run_directory / f"model_state.json.bak.{token}",
    }
    torch, _, _, _, _ = _require_torch()

    try:
        torch.save(bundle["model_state_dict"], temporary_model)
        with open(temporary_scaler, "wb") as handle:
            pickle.dump(bundle["scaler"], handle, protocol=pickle.HIGHEST_PROTOCOL)
        state_payload = {
            "model_config": bundle["model_config"],
            "feature_spec": bundle["feature_spec"],
            "feature_schema_hash": bundle["feature_schema_hash"],
            "training_metadata": bundle["training_metadata"],
            "update_count": int(update_count),
            "model_sha256": _file_sha256(temporary_model),
            "scaler_sha256": _file_sha256(temporary_scaler),
        }
        with open(temporary_state, "w", encoding="utf-8") as handle:
            json.dump(
                state_payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        torch.load(temporary_model, map_location="cpu")
        with open(temporary_scaler, "rb") as handle:
            pickle.load(handle)
        with open(temporary_state, "r", encoding="utf-8") as handle:
            json.load(handle)

        for final_path, backup_path in backup_paths.items():
            if final_path.exists():
                os.replace(final_path, backup_path)
        os.replace(temporary_model, final_model)
        os.replace(temporary_scaler, final_scaler)
        os.replace(temporary_state, final_state)
    except Exception:
        for final_path in (final_model, final_scaler, final_state):
            if final_path.exists():
                final_path.unlink()
        for final_path, backup_path in backup_paths.items():
            if backup_path.exists():
                os.replace(backup_path, final_path)
        raise
    finally:
        for path in (
            temporary_model,
            temporary_scaler,
            temporary_state,
            *backup_paths.values(),
        ):
            if path.exists():
                path.unlink()


def _load_persisted_model_bundle(run_directory):
    run_directory = Path(run_directory)
    model_path = run_directory / "model.pth"
    scaler_path = run_directory / "scaler.pkl"
    state_path = run_directory / "model_state.json"
    missing = [
        path.name
        for path in (model_path, scaler_path, state_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"运行实例目录 {run_directory} 缺少模型文件：{missing}。"
        )

    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    if _file_sha256(model_path) != state.get("model_sha256"):
        raise ValueError("model.pth 文件哈希与 model_state.json 不一致。")
    if _file_sha256(scaler_path) != state.get("scaler_sha256"):
        raise ValueError("scaler.pkl 文件哈希与 model_state.json 不一致。")

    torch, _, _, _, _ = _require_torch()
    model_state_dict = torch.load(model_path, map_location="cpu")
    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)
    bundle = {
        "model_state_dict": model_state_dict,
        "scaler": scaler,
        "model_config": state["model_config"],
        "feature_spec": state["feature_spec"],
        "feature_schema_hash": state["feature_schema_hash"],
        "training_metadata": state.get("training_metadata", {}),
    }
    return _validate_model_bundle(bundle), int(state.get("update_count", 0))


def _trading_days_since(last_anchor, target_date):
    from factor_lib.common.data_adapters.bigquant_adapters.loader import (
        load_trading_dates,
    )

    start = pd.Timestamp(last_anchor).normalize()
    end = pd.Timestamp(target_date).normalize()
    if end < start:
        raise ValueError("target_date 不能早于当前模型的训练锚点。")
    if end == start:
        return 0
    dates = load_trading_dates(start, end)
    return int((dates > start).sum())


def build_model_state_provider(
    persistence_mode="memory",
    model_root="/home/aiuser/work/userlib/factor_models",
    runtime_state: MutableMapping | None = None,
    run_label="gan_gru_score",
):
    """建立滚动模型状态提供器。

    ``memory`` 仅在当前 Python 进程保存模型；``simulation`` 使用
    runtime_state 记录运行实例 ID，并在该实例目录中覆盖三份当前模型文件。
    """
    if persistence_mode not in {"memory", "simulation"}:
        raise ValueError("persistence_mode 只能是 'memory' 或 'simulation'。")
    if not isinstance(run_label, str) or not run_label.strip():
        raise ValueError("run_label 必须是非空字符串。")
    safe_label = re.sub(r"[^0-9A-Za-z_-]+", "_", run_label.strip()).strip("_")
    if not safe_label:
        raise ValueError("run_label 清理后为空。")

    root = Path(model_root)
    state_key = "gan_gru_score_run_id"
    simulation_state: MutableMapping | None = None
    if persistence_mode == "simulation":
        if not isinstance(runtime_state, MutableMapping):
            raise TypeError(
                "simulation 模式必须传入可持久化的 MutableMapping "
                "作为 runtime_state。"
            )
        if not root.is_dir():
            raise FileNotFoundError(
                f"模型根目录不存在：{root}。请先创建该目录。"
            )
        simulation_state = runtime_state

    current_bundle = None
    update_count = 0
    run_directory = None

    def provider(
        target_date,
        feature_spec,
        instruments,
        training_config=None,
        show_progress=False,
    ):
        nonlocal current_bundle, update_count, run_directory
        normalized_features = _normalize_feature_spec(feature_spec)
        config = _normalize_training_config(training_config)
        target = pd.Timestamp(target_date).normalize()
        state_started_at = time.perf_counter()

        if persistence_mode == "simulation" and current_bundle is None:
            state = simulation_state
            if state is None:
                raise RuntimeError("simulation 模式缺少 runtime_state。")
            existing_run_id = state.get(state_key)
            if existing_run_id is not None:
                if not isinstance(existing_run_id, str) or not existing_run_id:
                    raise ValueError("runtime_state 中的运行实例 ID 无效。")
                run_directory = root / existing_run_id
                if show_progress:
                    _print_training_progress(
                        "加载持久化模型状态",
                        state_started_at,
                        detail=f"运行实例 {existing_run_id}",
                    )
                current_bundle, update_count = _load_persisted_model_bundle(
                    run_directory
                )
                if show_progress:
                    _print_training_progress(
                        "持久化模型状态加载完成",
                        state_started_at,
                        detail=f"已训练更新 {update_count} 次",
                    )

        needs_training = current_bundle is None
        elapsed_trading_days = None
        if current_bundle is not None:
            if current_bundle["feature_schema_hash"] != _json_hash(
                normalized_features
            ):
                raise ValueError(
                    "滚动运行期间 feature_spec 不得改变；"
                    "如需更换特征，请启动新的运行实例。"
                )
            training_metadata = current_bundle.get("training_metadata", {})
            expected_training_hash = _json_hash(config)
            if training_metadata.get("training_config_hash") != (
                expected_training_hash
            ):
                raise ValueError(
                    "滚动运行期间 training_config 不得改变；"
                    "如需更换训练参数，请启动新的运行实例。"
                )
            last_anchor = training_metadata.get("anchor_date")
            if last_anchor is None:
                raise ValueError("当前模型状态缺少训练锚点。")
            elapsed_trading_days = _trading_days_since(last_anchor, target)
            needs_training = (
                elapsed_trading_days >= config["retrain_interval_days"]
            )

        if not needs_training:
            if show_progress:
                remaining_days = max(
                    config["retrain_interval_days"] - int(elapsed_trading_days or 0),
                    0,
                )
                _print_training_progress(
                    "使用当前模型推理",
                    state_started_at,
                    detail=(
                        f"当前 {target:%Y-%m-%d}，已距上次训练 "
                        f"{elapsed_trading_days or 0} 个交易日，"
                        f"距下次更新 {remaining_days} 日"
                    ),
                )
            return current_bundle

        if show_progress:
            reason = "首次训练" if current_bundle is None else "达到模型更新周期"
            _print_training_progress(
                "准备训练模型",
                state_started_at,
                detail=f"{reason}，当前锚点 {target:%Y-%m-%d}",
            )

        new_bundle = train_gan_gru_model(
            anchor_date=target,
            feature_spec=normalized_features,
            instruments=instruments,
            training_config=config,
            show_progress=show_progress,
        )
        new_update_count = update_count + 1

        if persistence_mode == "simulation":
            state = simulation_state
            if state is None:
                raise RuntimeError("simulation 模式缺少 runtime_state。")
            created_new_directory = False
            if run_directory is None:
                run_id = (
                    f"{safe_label}_"
                    f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_"
                    f"{uuid.uuid4().hex[:8]}"
                )
                run_directory = root / run_id
                run_directory.mkdir(parents=False, exist_ok=False)
                created_new_directory = True
            try:
                if show_progress:
                    _print_training_progress(
                        "保存运行实例模型状态",
                        state_started_at,
                        detail=(
                            f"更新次数 {new_update_count}，目录 "
                            f"{run_directory.name}"
                        ),
                    )
                _persist_model_bundle(
                    run_directory,
                    new_bundle,
                    new_update_count,
                )
            except Exception:
                if created_new_directory and run_directory.exists():
                    try:
                        run_directory.rmdir()
                    except OSError:
                        pass
                if created_new_directory:
                    run_directory = None
                raise
            active_run_directory = run_directory
            if active_run_directory is None:
                raise RuntimeError("simulation 运行实例目录未建立。")
            state[state_key] = active_run_directory.name

        current_bundle = new_bundle
        update_count = new_update_count
        if show_progress:
            _print_training_progress(
                "当前模型已就绪",
                state_started_at,
                detail=(
                    f"锚点 {target:%Y-%m-%d}，累计更新 {update_count} 次"
                ),
            )
        return current_bundle

    return provider


FACTOR = {
    "name": "gan_gru_score",
    "func": calc_gan_gru_score,
    "input_schema": {
        "required": {"date": {}, "instrument": {}},
        "conditional": {},
    },
    "parameters": {
        # 需要输出因子值的最终截面日期。策略和评价函数负责传入；
        # loader 会从这些日期向前展开推理序列及每个依赖因子自己的预热期。
        "target_dates": {"default": None},

        # 本次计算能够使用信息的最晚日期。target_dates 中任何日期晚于该值
        # 都会直接报错；它是防未来信息边界，不能代替 target_dates。
        "as_of_date": {"default": None},

        # 滚动训练模式的特征定义。格式为：
        # [{"factor_name": 因子名, "params": 因子参数,
        #   "feature_name": 模型特征列名}, ...]
        # 列表顺序就是模型输入维度顺序。同一运行实例启动后不得改变。
        # None 表示滚动模式使用 DEFAULT_FEATURE_SPEC；固定模型模式必须为 None，
        # 因为固定模型的特征定义只能来自 fixed_model_bundle。
        "feature_spec": {"default": None},

        # 固定模型模式的完整模型包，必须同时包含：model_state_dict、scaler、
        # model_config、feature_spec、feature_schema_hash。
        # 使用该参数时，model_state_provider 必须为 None，且不能另外传入
        # feature_spec 或 training_config。
        "fixed_model_bundle": {"default": None},

        # 滚动训练模式的模型状态提供器，应由 build_model_state_provider()
        # 创建。它根据当前目标日判断是否需要重新训练，并返回当前有效模型包。
        # 使用该参数时，fixed_model_bundle 必须为 None。
        "model_state_provider": {"default": None},

        # 滚动训练参数字典。None 使用 DEFAULT_TRAINING_CONFIG。
        # 可配置项仅限该常量中已经登记的键；未知参数会报错。
        # sequence_length 同时决定 loader 预存多少个特征截面；
        # retrain_interval_days 决定模型按多少个交易日更新一次。
        # 固定模型模式不接受该参数。
        "training_config": {"default": None},

        # 是否显示依赖因子计算、GAN 训练、GRU 训练和推理进度。
        "show_progress": {"default": False},

        # 推理阶段每处理 progress_every 个目标截面更新一次；
        # 训练样本构造阶段按该默认频率更新股票处理进度。
        # GAN、GRU 和验证阶段始终逐批报告，不受本参数限制。
        "progress_every": {"default": 20},
    },
    "dependencies": {"resolver": _resolve_gan_gru_dependencies},
    "data_window": {
        "lookback_trading_days": 0,
        "requires_target_date_data": True,
        "minimum_history_observations": 0,
        "preheating_required": False,
        "insufficient_window_behavior": (
            "外层只接收最终目标日；各依赖因子的序列期和预热期由 loader "
            "分别准备。"
        ),
    },
    "output_schema": {
        "date": {},
        "instrument": {},
        "gan_gru_score": {"dtype": "float64"},
    },
}


FACTOR_INFO = """# gan_gru_score

## 因子含义

GAN-GRU 机器学习评分因子。模型读取每只股票连续多个交易日的基础因子序列，GAN 只在训练阶段生成增强样本，最终由 GRU 输出股票分数。输出列为 `date`、`instrument`、`gan_gru_score`；分数越高，表示模型预测的未来收益越高。

本因子有且只有两种运行模式：固定模型模式和滚动训练模式。两种模式不能同时启用。

## 特征定义 feature_spec

滚动模式通过 `feature_spec` 指定依赖因子，格式如下：

```python
feature_spec = [
    {
        "factor_name": "return_nd",
        "params": {"window": 5},
        "feature_name": "return_5d",
    },
    {
        "factor_name": "return_std_nd",
        "params": {"window": 20},
        "feature_name": "return_std_20d",
    },
]
```

- `factor_name`：因子库中已经登记的基础因子名称。
- `params`：该基础因子的内部参数，不得传入 `data`、`target_dates`、`as_of_date` 或进度参数。
- `feature_name`：该特征进入模型后的唯一列名。
- 列表顺序就是模型输入维度顺序，不只是展示顺序。

滚动模式不传 `feature_spec` 时使用 `DEFAULT_FEATURE_SPEC`，即模型4对应的20项默认特征。一次滚动运行首次训练后，特征数量、组合、参数和顺序全部固定；如需更改，必须启动新的运行实例。

## 模式一：固定模型

固定模型适合固定权重回测、模型对照研究和冻结模型推理。必须传入一个完整的 `fixed_model_bundle`，不能只传权重。

模型包至少包含：

- `model_state_dict`：GRU 权重。
- `scaler`：训练期得到的 `mean` 和 `std`。
- `model_config`：`input_size`、`sequence_length`、`hidden_size`、`num_layers` 和 `dropout`。
- `feature_spec`：训练该权重时实际使用的完整特征定义。
- `feature_schema_hash`：特征定义的校验值。

可以先通过本脚本的训练函数生成模型包：

```python
from importlib import import_module

gan_gru_module = import_module(
    "factor_lib.Factor Repository.gan_gru_score"
)

fixed_model_bundle = gan_gru_module.train_gan_gru_model(
    anchor_date="2024-05-10",
    feature_spec=None,  # 使用默认20项特征
    instruments=training_instruments,
    training_start_date="2022-01-04",
    training_end_date="2024-04-30",
    show_progress=True,
)

factor_params = {
    "fixed_model_bundle": fixed_model_bundle,
}
```

`anchor_date` 是训练时的信息截止日。固定训练期结束后的未来收益标签必须已经在该日期前完整实现，否则训练函数会报错。固定模式下不要再传 `feature_spec`、`model_state_provider` 或 `training_config`，模型特征只能以模型包中的记录为准。

## 模式二：滚动训练

滚动模式适合滚动回测和持续运行。因子在每个目标截面调用模型状态提供器；提供器会先判断当前模型是否仍在有效更新周期内，只有达到更新间隔时才重新训练。

回测和研究评价默认使用内存状态：

```python
from importlib import import_module

gan_gru_module = import_module(
    "factor_lib.Factor Repository.gan_gru_score"
)

model_state_provider = gan_gru_module.build_model_state_provider(
    persistence_mode="memory",
)

factor_params = {
    "feature_spec": None,  # None 表示 DEFAULT_FEATURE_SPEC
    "model_state_provider": model_state_provider,
    "training_config": {
        "sequence_length": 40,
        "training_window_days": 336,
        "retrain_interval_days": 84,
        "label_horizon_days": 5,
    },
}
```

模拟运行需要把运行实例编号保存在平台提供的持久化小状态中，并把模型文件写入预先建立的 `factor_models` 根目录：

```python
model_state_provider = gan_gru_module.build_model_state_provider(
    persistence_mode="simulation",
    model_root="/home/aiuser/work/userlib/factor_models",
    runtime_state=simulation_runtime_state,
    run_label="gan_gru_score",
)
```

这里的 `simulation_runtime_state` 必须是模拟运行包装层传入的、可跨重启保存的可写映射，不能用临时空字典冒充持久化状态。每个运行实例只维护当前有效的 `model.pth`、`scaler.pkl` 和 `model_state.json`；更新成功后覆盖旧版本，更新失败则保留旧模型文件。

## training_config 参数

`training_config=None` 时使用以下默认值：

- `sequence_length=40`：每个样本包含的连续特征交易日数，同时决定推理预存窗口。
- `training_window_days=336`：每次滚动训练使用的样本日期数量。
- `retrain_interval_days=84`：距离上次训练锚点达到多少个交易日后更新。
- `label_horizon_days=5`：监督标签为未来5个交易日累计收益。
- `validation_ratio=0.20`：时间顺序验证集所占比例。
- `purge_trading_days=5`：训练集和验证集之间剔除的交易日数。
- `minimum_validation_dates=20`：验证集至少包含的截面日期数。
- `minimum_rankic_stocks=30`：计算单日验证 RankIC 所需的最少股票数。
- `minimum_training_samples=1000`：时间切分后允许开始训练的最少样本数。
- `latent_dim=10`：GAN 随机噪声维度。
- `discriminator_hidden_size=32`：判别器 GRU 隐藏层维度。
- `lambda_reconstruction=0.5`：生成器重构损失权重。
- `gan_epochs=4`：GAN 最大训练轮数。
- `gru_max_epochs=20`：GRU 最大训练轮数。
- `early_stop_patience=4`：验证 RankIC 连续多少轮未改善后早停。
- `rankic_min_delta=0.0001`：判定 RankIC 改善所需的最小增量。
- `batch_size=1024`：GAN训练、增强生成及验证推理批量大小。
- `cpu_threads=4`：训练时显式分配给 PyTorch 的 CPU 线程数；沿用研究稿默认值。该参数影响运行速度，不改变样本、标签或模型结构。
- `hidden_size=64`：预测 GRU 隐藏层维度。
- `num_layers=2`：预测 GRU 层数。
- `dropout=0.2`：多层 GRU 的 dropout 比例。
- `gan_learning_rate=0.0002`：生成器和判别器学习率。
- `gru_learning_rate=0.001`：预测 GRU 学习率。
- `random_seed=42`：NumPy 与 PyTorch 随机种子。

未知参数不会被静默忽略，而是直接报错。

## 数据与时间边界

策略或评价函数只需传入最终目标日期。loader 会把目标日期展开为模型所需的连续序列日期，再分别读取每个基础因子自己的预热窗口，不会用最长因子的窗口统一覆盖所有依赖。

滚动训练锚点为当前目标日。若标签周期为 H 日，则训练样本的最后特征日期最多只能到锚点前 H 个交易日，确保未来收益标签在训练当时已经完整实现。训练和推理共用同一份 `feature_spec`、特征顺序和标准化口径。

任何股票只要在序列期内存在缺失特征或非有限值，就不会在该目标日输出模型分数；函数不会用0自动填充缺失特征。

## 常见参数错误

- 同时传入 `fixed_model_bundle` 和 `model_state_provider`。
- 两种模式都未提供。
- 固定模型模式另外传入 `feature_spec` 或 `training_config`。
- 固定模型包中的特征数量、顺序、哈希或标准化器维度不一致。
- 滚动运行中途改变 `feature_spec` 或 `training_config`。
- 模拟模式没有传入真正可持久化的 `runtime_state`。
- 训练股票池为空，或有效训练/验证样本不足。
"""
