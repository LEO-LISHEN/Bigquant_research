# -*- coding: utf-8 -*-
"""GAN-GRU 机器学习因子。

因子计算只消费已经由 loader 准备好的依赖因子数据。滚动训练由本文件中的
模型状态提供器负责，并通过 loader 按各依赖因子自己的预热窗口读取数据。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import re
import threading
import time
import uuid
from collections.abc import Mapping, MutableMapping, Sequence
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
    # strict：任何序列位置存在 NaN 时不构造样本，保持旧版严格口径。
    # impute_with_mask：仅用训练期统计量填补 NaN，并将缺失掩码作为额外输入。
    "missing_feature_mode": "strict",
    # 放宽模式下，一条样本至少有该比例的原始特征值真实可得；防止用几乎全空的
    # 序列强行产生分数。严格模式下此参数恒等于 1.0。
    "minimum_observed_value_ratio": 0.50,
}


# 特征筛选只服务于 GAN-GRU，不能替代正式的样本外回测。默认值刻意保守：
# 初筛只训练一次低预算模型，只有少量最终候选集合使用正式训练参数复训确认。
DEFAULT_FEATURE_SELECTION_CONFIG = {
    "train_ratio": 0.70,
    "additional_purge_trading_days": 0,
    "min_coverage": 0.80,
    "min_rankic_stocks": 30,
    "min_positive_rankic_ratio": 0.50,
    "rankic_lcb_z": 1.645,
    "hac_lags": None,
    "top_quantile": 0.10,
    "permutation_repeats": 1,
    # None 表示不因数量截断置换重要性排序；最终集合规模仅由
    # final_feature_counts 明确指定。
    "max_selected_features": None,
    "final_feature_counts": (4, 6, 8),
    "screening_gan_epochs": 1,
    "screening_gru_max_epochs": 3,
    "screening_early_stop_patience": 1,
    "max_interaction_pairs": 0,
    # 先以严格的完整序列口径审计候选池。若全体候选交集低于门槛，按“移除后
    # 交集覆盖率提升最大”的规则逐个移除，直到达到门槛；不预设保留数量。
    "candidate_pool_min_coverage": 0.80,
    "candidate_pool_min_feature_count": 1,
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

    # `_normalize_training_config` 会在返回结果中附加一个供模型包使用的
    # 派生字段 `missing_data_config`。特征筛选流程会基于该规范化结果再生成
    # 一份“低预算筛选训练配置”，因此该派生字段可能再次传回本函数。
    #
    # 这里显式接受并校验它，使规范化函数具有幂等性；同时仍拒绝真正未知的
    # 用户参数，避免拼写错误被静默吞掉。
    supplied_missing_data_config = supplied.pop("missing_data_config", None)

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
    resolved_missing_data_config = _normalize_missing_data_config(
        {
            "mode": config["missing_feature_mode"],
            "minimum_observed_value_ratio": config[
                "minimum_observed_value_ratio"
            ],
        }
    )
    if supplied_missing_data_config is not None:
        supplied_missing_data_config = _normalize_missing_data_config(
            supplied_missing_data_config
        )
        if supplied_missing_data_config != resolved_missing_data_config:
            raise ValueError(
                "training_config 中的 missing_data_config 与 "
                "missing_feature_mode / minimum_observed_value_ratio 不一致。"
            )
    config["missing_data_config"] = resolved_missing_data_config
    config["missing_feature_mode"] = config["missing_data_config"]["mode"]
    config["minimum_observed_value_ratio"] = config["missing_data_config"][
        "minimum_observed_value_ratio"
    ]
    return config


def _normalize_missing_data_config(value):
    """校验并标准化模型的缺失特征处理协议。

    模型包必须持久化该协议；否则固定模型推理时无法判断输入维度，也不能保证
    训练和推理采取同一填补规则。旧模型包未登记时按 strict 兼容。
    """
    if value is None:
        supplied = {}
    elif isinstance(value, Mapping):
        supplied = dict(value)
    else:
        raise TypeError("missing_data_config 必须是字典或 None。")
    allowed = {"mode", "minimum_observed_value_ratio"}
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"missing_data_config 包含未知参数：{unknown}。")
    mode = str(supplied.get("mode", "strict")).strip().lower()
    if mode not in {"strict", "impute_with_mask"}:
        raise ValueError(
            "missing_feature_mode 只能为 'strict' 或 'impute_with_mask'。"
        )
    ratio = float(supplied.get("minimum_observed_value_ratio", 1.0))
    if not 0.0 < ratio <= 1.0:
        raise ValueError("minimum_observed_value_ratio 必须位于 (0, 1]。")
    if mode == "strict":
        ratio = 1.0
    return {"mode": mode, "minimum_observed_value_ratio": ratio}


def _model_input_size(feature_count, missing_data_config):
    config = _normalize_missing_data_config(missing_data_config)
    return feature_count * (2 if config["mode"] == "impute_with_mask" else 1)


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


def _normalize_model_config(model_config, feature_count, missing_data_config=None):
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
    expected_input_size = _model_input_size(
        feature_count,
        missing_data_config,
    )
    if normalized["input_size"] != expected_input_size:
        raise ValueError(
            "model_config.input_size 与 feature_spec 和缺失值处理模式不一致。"
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
    missing_data_config = _normalize_missing_data_config(
        model_bundle.get("missing_data_config")
    )
    model_config = _normalize_model_config(
        model_bundle["model_config"],
        len(feature_spec),
        missing_data_config,
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
        "missing_data_config": missing_data_config,
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
                # 不能在合并阶段丢弃任一因子缺失的股票；strict/放宽模式分别在
                # 序列构造阶段决定是否接受该样本。
                how="outer",
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
    missing_data_config=None,
):
    missing_config = _normalize_missing_data_config(missing_data_config)
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
        observed_ratio = float(np.isfinite(values).mean())
        if (
            missing_config["mode"] == "strict"
            and not np.isfinite(values).all()
        ):
            continue
        if observed_ratio < missing_config["minimum_observed_value_ratio"]:
            continue
        instruments.append(str(instrument))
        sequences.append(values)
    if not sequences:
        return [], np.empty(
            (0, len(sequence_dates), len(feature_names)),
            dtype=np.float32,
        )
    return instruments, np.stack(sequences).astype(np.float32, copy=False)


def _transform_sequences_for_model(sequences, bundle):
    """用模型包中训练期统计量完成推理前变换，绝不使用推理期数据拟合。"""
    feature_count = len(bundle["feature_spec"])
    expected_raw_shape = (
        bundle["model_config"]["sequence_length"],
        feature_count,
    )
    if sequences.ndim != 3 or tuple(sequences.shape[1:]) != expected_raw_shape:
        raise ValueError(
            f"原始推理序列形状应为 (样本数, {expected_raw_shape[0]}, "
            f"{expected_raw_shape[1]})，实际为 {sequences.shape}。"
        )
    values = np.asarray(sequences, dtype=np.float32)
    observed = np.isfinite(values)
    mode = bundle["missing_data_config"]["mode"]
    if mode == "strict" and not observed.all():
        raise ValueError("strict 模式不接受包含缺失特征值的推理序列。")
    observed_ratio = observed.mean(axis=(1, 2))
    threshold = bundle["missing_data_config"]["minimum_observed_value_ratio"]
    if (observed_ratio < threshold).any():
        raise ValueError("推理序列的真实特征值占比低于模型登记的最低要求。")
    mean = bundle["scaler"]["mean"].reshape(1, 1, -1)
    std = bundle["scaler"]["std"].reshape(1, 1, -1)
    filled = np.where(observed, values, mean)
    standardized = ((filled - mean) / std).astype(np.float32, copy=False)
    if mode == "strict":
        return standardized
    # 掩码 1 表示该位置由训练期均值填补，模型可学习“缺失本身”的信息。
    missing_mask = (~observed).astype(np.float32, copy=False)
    return np.concatenate([standardized, missing_mask], axis=2)


def _predict_with_bundle(model_bundle, sequences):
    bundle = _validate_model_bundle(model_bundle)
    model, torch = _build_gru_model(bundle["model_config"])
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    model.eval()
    standardized = _transform_sequences_for_model(sequences, bundle)
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
    runtime_missing_config = (
        _validate_model_bundle(fixed_model_bundle)["missing_data_config"]
        if fixed_model_bundle is not None
        else _normalize_training_config(training_config)["missing_data_config"]
    )
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
            runtime_missing_config,
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
                # 保留单个因子的 NaN，交由样本构造阶段按模型缺失模式处理。
                how="outer",
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
    missing_data_config=None,
):
    missing_config = _normalize_missing_data_config(missing_data_config)
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
                observed_ratio = float(np.isfinite(sequence).mean())
                if (
                    missing_config["mode"] == "strict"
                    and not np.isfinite(sequence).all()
                ):
                    continue
                if observed_ratio < missing_config["minimum_observed_value_ratio"]:
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
    universe_panel=None,
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
    if universe_panel is not None:
        if not isinstance(universe_panel, pd.DataFrame):
            raise TypeError("universe_panel 必须是 pandas.DataFrame 或 None。")
        required_universe_columns = {"date", "instrument"}
        missing_universe_columns = required_universe_columns - set(
            universe_panel.columns
        )
        if missing_universe_columns:
            raise ValueError(
                "universe_panel 缺少字段："
                f"{sorted(missing_universe_columns)}。"
            )
        eligible_universe = universe_panel.loc[:, ["date", "instrument"]].copy()
        eligible_universe["date"] = pd.to_datetime(
            eligible_universe["date"], errors="coerce"
        ).dt.normalize()
        if eligible_universe.isna().any().any():
            raise ValueError("universe_panel 包含无效 date 或 instrument。")
        if eligible_universe.duplicated(["date", "instrument"]).any():
            raise ValueError("universe_panel 存在重复 date + instrument。")
        labels = labels.merge(
            eligible_universe,
            on=["date", "instrument"],
            how="inner",
            validate="one_to_one",
        )
        if labels.empty:
            raise ValueError("按动态股票池过滤后训练标签为空。")
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
        missing_data_config=config["missing_data_config"],
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

    # 仅用训练子集拟合填补值与标准化参数，验证/推理期绝不参与。
    mean = np.nanmean(x_train.reshape(-1, len(feature_names)), axis=0)
    std = np.nanstd(x_train.reshape(-1, len(feature_names)), axis=0, ddof=0)
    if not np.isfinite(mean).all():
        missing_features = [
            feature_names[index]
            for index, value in enumerate(mean)
            if not np.isfinite(value)
        ]
        raise ValueError(
            "训练期存在完全没有有效值的特征，不能进行缺失值放宽训练："
            f"{missing_features}。"
        )
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    mean = mean.astype(np.float32)

    def prepare_inputs(raw_sequences):
        observed = np.isfinite(raw_sequences)
        filled = np.where(observed, raw_sequences, mean.reshape(1, 1, -1))
        standardized = (
            (filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        ).astype(np.float32)
        if config["missing_data_config"]["mode"] == "strict":
            return standardized
        return np.concatenate(
            [standardized, (~observed).astype(np.float32)], axis=2
        )

    x_train = prepare_inputs(x_train)
    x_validation = prepare_inputs(x_validation)
    model_input_size = _model_input_size(
        len(feature_names), config["missing_data_config"]
    )
    if show_progress:
        _print_training_progress(
            "完成训练集标准化",
            training_started_at,
            detail=(
                f"原始特征维度 {len(feature_names)}，模型输入维度 {model_input_size}，"
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
    generator = Generator(model_input_size, config["latent_dim"]).to(device)
    discriminator = Discriminator(
        config["sequence_length"],
        model_input_size,
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
            "input_size": model_input_size,
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
            "missing_data_config": config["missing_data_config"],
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
                "missing_data_config": config["missing_data_config"],
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


def _selection_progress(stage, started_at, completed=None, total=None, detail=""):
    """GAN-GRU 特征筛选的统一单行进度输出。"""
    elapsed = time.perf_counter() - started_at
    parts = [f"[GAN-GRU特征筛选] {stage}"]
    if completed is not None and total is not None and total > 0:
        ratio = min(max(float(completed) / float(total), 0.0), 1.0)
        parts.append(f"{completed}/{total} ({ratio:.1%})")
        if 0 < completed < total:
            parts.append(f"预计剩余 {elapsed * (1.0 - ratio) / ratio:.1f}s")
    if detail:
        parts.append(str(detail))
    parts.append(f"已耗时 {elapsed:.1f}s")
    print("\r" + " | ".join(parts).ljust(220), end="", flush=True)


def _selection_positive_integer(value, parameter_name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{parameter_name} 必须为{'非负' if allow_zero else '正'}整数。")
    value = int(value)
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{parameter_name} 必须为{'非负' if allow_zero else '正'}整数。")
    return value


def _normalize_feature_selection_config(selection_config):
    if selection_config is None:
        supplied = {}
    elif isinstance(selection_config, Mapping):
        supplied = dict(selection_config)
    else:
        raise TypeError("selection_config 必须是字典或 None。")
    unknown = sorted(set(supplied) - set(DEFAULT_FEATURE_SELECTION_CONFIG))
    if unknown:
        raise ValueError(f"selection_config 包含未知参数：{unknown}。")
    config = dict(DEFAULT_FEATURE_SELECTION_CONFIG)
    config.update(supplied)

    ratio = float(config["train_ratio"])
    if not 0.5 <= ratio < 1.0:
        raise ValueError("selection_config['train_ratio'] 必须位于 [0.5, 1) 内。")
    config["train_ratio"] = ratio
    config["additional_purge_trading_days"] = _selection_positive_integer(
        config["additional_purge_trading_days"],
        "selection_config['additional_purge_trading_days']",
        allow_zero=True,
    )
    config["min_rankic_stocks"] = _selection_positive_integer(
        config["min_rankic_stocks"],
        "selection_config['min_rankic_stocks']",
    )
    for name in (
        "permutation_repeats",
        "screening_gan_epochs",
        "screening_gru_max_epochs",
        "screening_early_stop_patience",
    ):
        config[name] = _selection_positive_integer(
            config[name], f"selection_config[{name!r}]"
        )
    if config["max_selected_features"] is not None:
        config["max_selected_features"] = _selection_positive_integer(
            config["max_selected_features"],
            "selection_config['max_selected_features']",
        )
    config["max_interaction_pairs"] = _selection_positive_integer(
        config["max_interaction_pairs"],
        "selection_config['max_interaction_pairs']",
        allow_zero=True,
    )
    for name in (
        "min_coverage",
        "candidate_pool_min_coverage",
        "min_positive_rankic_ratio",
        "top_quantile",
    ):
        value = float(config[name])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"selection_config[{name!r}] 必须位于 (0, 1]。")
        config[name] = value
    config["candidate_pool_min_feature_count"] = _selection_positive_integer(
        config["candidate_pool_min_feature_count"],
        "selection_config['candidate_pool_min_feature_count']",
    )
    z_value = float(config["rankic_lcb_z"])
    if not np.isfinite(z_value) or z_value <= 0:
        raise ValueError("selection_config['rankic_lcb_z'] 必须为正数。")
    config["rankic_lcb_z"] = z_value
    if config["hac_lags"] is not None:
        config["hac_lags"] = _selection_positive_integer(
            config["hac_lags"],
            "selection_config['hac_lags']",
            allow_zero=True,
        )

    counts = config["final_feature_counts"]
    if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)):
        raise TypeError("selection_config['final_feature_counts'] 必须是整数序列。")
    config["final_feature_counts"] = tuple(
        sorted(
            {
                _selection_positive_integer(
                    value, "selection_config['final_feature_counts']"
                )
                for value in counts
            }
        )
    )
    if not config["final_feature_counts"]:
        raise ValueError("selection_config['final_feature_counts'] 不能为空。")
    return config


def _normalize_selection_candidate_params(params, factor_name):
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError(f"候选因子 {factor_name!r} 的参数必须是字典。")
    result = dict(params)
    reserved = sorted(set(result) & _RESERVED_CHILD_PARAMS)
    if reserved:
        raise ValueError(
            f"候选因子 {factor_name!r} 不得覆盖系统参数：{reserved}。"
        )
    return result


def _candidate_instances_from_metadata(factor_name, metadata):
    """将 FACTOR 中登记的常用实例展开为稳定的候选定义。"""
    raw_instances = metadata.get("candidate_instances")
    if raw_instances is None:
        return [("default", {})]
    if isinstance(raw_instances, Mapping):
        source = [
            {"id": instance_id, "params": params}
            for instance_id, params in raw_instances.items()
        ]
    elif isinstance(raw_instances, Sequence) and not isinstance(
        raw_instances, (str, bytes)
    ):
        source = list(raw_instances)
    else:
        raise TypeError(
            f"因子 {factor_name!r} 的 candidate_instances 必须是列表或字典。"
        )
    result = []
    seen = set()
    for position, item in enumerate(source, start=1):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"因子 {factor_name!r} 的 candidate_instances 第 {position} 项必须是字典。"
            )
        instance_id = str(item.get("id", "")).strip()
        if not instance_id:
            raise ValueError(
                f"因子 {factor_name!r} 的 candidate_instances 第 {position} 项缺少 id。"
            )
        if instance_id in seen:
            raise ValueError(
                f"因子 {factor_name!r} 的 candidate_instances id 重复：{instance_id!r}。"
            )
        seen.add(instance_id)
        result.append(
            (
                instance_id,
                _normalize_selection_candidate_params(
                    item.get("params", {}), factor_name
                ),
            )
        )
    if not result:
        raise ValueError(f"因子 {factor_name!r} 的 candidate_instances 不能为空。")
    return result


def _normalize_candidate_source(candidate_source):
    if candidate_source is None:
        return ()
    if isinstance(candidate_source, str):
        candidate_source = [candidate_source]
    if not isinstance(candidate_source, Sequence):
        raise TypeError("candidate_source 必须是字符串、字符串序列或 None。")
    aliases = {
        "base": "base",
        "base_factors": "base",
        "composite": "composite",
        "composite_factors": "composite",
    }
    result = []
    for item in candidate_source:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("candidate_source 含有无效类别名称。")
        key = item.strip().lower()
        if key not in aliases:
            raise ValueError(
                "candidate_source 仅支持 'base_factors'、'composite_factors'，"
                "或对应的 base、composite 别名。"
            )
        if aliases[key] not in result:
            result.append(aliases[key])
    return tuple(result)


def _candidate_feature_name(factor_name, instance_id):
    value = re.sub(r"[^0-9A-Za-z_]+", "_", f"{factor_name}_{instance_id}")
    value = value.strip("_").lower()
    if not value:
        raise ValueError(f"无法为 {factor_name!r} 生成合法 feature_name。")
    return value


def _expand_feature_selection_candidates(candidate_spec, candidate_source):
    """支持手工候选字典与按 FACTOR['factor_type'] 的整类发现。"""
    from factor_lib.common.data_adapters.bigquant_adapters.loader import (
        get_factor_metadata,
    )
    from factor_lib.factor_hub.discover_factors import discover_factors

    if candidate_spec is not None and not isinstance(candidate_spec, Mapping):
        raise TypeError("candidate_spec 必须是字典或 None。")
    source_types = _normalize_candidate_source(candidate_source)
    if candidate_spec is None and not source_types:
        raise ValueError("必须传入 candidate_spec 或 candidate_source。")

    raw = []
    if candidate_spec is not None:
        for factor_name, requested_instances in candidate_spec.items():
            if not isinstance(factor_name, str) or not factor_name.strip():
                raise ValueError("candidate_spec 存在无效因子名称。")
            name = factor_name.strip()
            metadata = get_factor_metadata(name)
            if metadata.get("factor_type") == "machine_learning" or name == "gan_gru_score":
                raise ValueError("特征筛选候选池不能包含机器学习因子或 gan_gru_score 自身。")
            if requested_instances is None:
                instances = _candidate_instances_from_metadata(name, metadata)
            elif isinstance(requested_instances, Mapping):
                instances = [("custom", _normalize_selection_candidate_params(requested_instances, name))]
            elif isinstance(requested_instances, Sequence) and not isinstance(
                requested_instances, (str, bytes)
            ):
                instances = []
                for index, value in enumerate(requested_instances, start=1):
                    if not isinstance(value, Mapping):
                        raise TypeError(
                            f"candidate_spec[{name!r}] 第 {index} 项必须是参数字典。"
                        )
                    raw_id = str(value.get("id", "")).strip()
                    params = value.get("params", value)
                    if raw_id:
                        if not isinstance(params, Mapping):
                            raise TypeError(f"候选因子 {name!r} 的 params 必须是字典。")
                        params = dict(params)
                        params.pop("id", None)
                        params.pop("params", None)
                    else:
                        params = dict(value)
                    params = _normalize_selection_candidate_params(params, name)
                    instance_id = raw_id or _json_hash(params)[:10]
                    instances.append((instance_id, params))
            else:
                raise TypeError(
                    f"candidate_spec[{name!r}] 必须是 None、参数字典或参数字典列表。"
                )
            raw.extend((name, instance_id, params) for instance_id, params in instances)

    if source_types:
        discovered = discover_factors()
        for name, metadata in sorted(discovered.items()):
            factor_type = metadata.get("factor_type")
            if factor_type not in source_types:
                continue
            if factor_type == "machine_learning" or name == "gan_gru_score":
                continue
            raw.extend(
                (name, instance_id, params)
                for instance_id, params in _candidate_instances_from_metadata(
                    name, metadata
                )
            )

    candidates = []
    seen_ids = set()
    seen_feature_names = set()
    for factor_name, instance_id, params in raw:
        candidate_id = f"{factor_name}::{instance_id}"
        if candidate_id in seen_ids:
            continue
        feature_name = _candidate_feature_name(factor_name, instance_id)
        if feature_name in seen_feature_names:
            raise ValueError(f"候选特征列名重复：{feature_name!r}。")
        seen_ids.add(candidate_id)
        seen_feature_names.add(feature_name)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "factor_name": factor_name,
                "instance_id": str(instance_id),
                "params": dict(params),
                "feature_name": feature_name,
            }
        )
    if not candidates:
        raise ValueError("展开后没有可用于 GAN-GRU 的候选特征实例。")
    return candidates


def _normalize_selection_instruments(instruments):
    if isinstance(instruments, str):
        instruments = [instruments]
    if not isinstance(instruments, Sequence):
        raise TypeError("股票列表必须是非空代码序列。")
    result = []
    for value in instruments:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"无效股票代码：{value!r}")
        code = value.strip()
        if code not in result:
            result.append(code)
    if not result:
        raise ValueError("股票列表不能为空。")
    return result


def _selection_quote_sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _selection_index_universe(index_codes, target_dates):
    try:
        import dai
    except ImportError as exc:
        raise ImportError("未能导入 dai；请在 BigQuant 环境运行。") from exc
    date_sql = ", ".join(
        _selection_quote_sql_literal(date.strftime("%Y-%m-%d"))
        for date in target_dates
    )
    index_sql = ", ".join(_selection_quote_sql_literal(code) for code in index_codes)
    sql = f"""
    SELECT date, member_code AS instrument
    FROM cn_stock_index_component
    WHERE instrument IN ({index_sql})
      AND date IN ({date_sql})
    ORDER BY date, member_code
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
    required = {"date", "instrument"}
    if required - set(panel.columns):
        raise ValueError("指数成分股查询结果缺少 date 或 instrument。")
    panel = panel.loc[:, ["date", "instrument"]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["instrument"] = panel["instrument"].astype("string").str.strip()
    if panel.empty or panel.isna().any().any() or (panel["instrument"] == "").any():
        raise ValueError("未读取到完整有效的指数历史成分股。")
    panel = panel.drop_duplicates(["date", "instrument"])
    missing_dates = target_dates.difference(pd.DatetimeIndex(panel["date"].unique()))
    if not missing_dates.empty:
        raise ValueError("部分筛选日期未找到指数历史成分股。")
    return panel.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)


def _resolve_selection_universe(universe, target_dates, show_progress, started_at):
    """将三种用户股票池模式解析为点时 date + instrument 面板。"""
    from factor_lib.common.data_adapters.bigquant_adapters.daily import (
        load_daily_raw_data,
    )

    if universe is None or universe == "all_a":
        raise ValueError(
            "GAN-GRU 特征筛选必须显式指定市场市值组、动态指数或自定义股票列表；"
            "不支持无边界的 all_a 股票池。"
        )
    if isinstance(universe, Sequence) and not isinstance(universe, (str, bytes, Mapping)):
        instruments = _normalize_selection_instruments(universe)
        panel = pd.MultiIndex.from_product(
            [target_dates, instruments], names=["date", "instrument"]
        ).to_frame(index=False)
        return {"type": "custom", "instruments": instruments}, panel, instruments
    if not isinstance(universe, Mapping):
        raise TypeError("universe 必须是股票列表或股票池配置字典。")
    universe_type = str(universe.get("type", "")).strip().lower()
    if universe_type in {"custom", "custom_list"}:
        instruments = _normalize_selection_instruments(universe.get("instruments"))
        panel = pd.MultiIndex.from_product(
            [target_dates, instruments], names=["date", "instrument"]
        ).to_frame(index=False)
        return {"type": "custom", "instruments": instruments}, panel, instruments
    if universe_type == "index":
        index_codes = _normalize_selection_instruments(
            universe.get("index_codes", universe.get("code"))
        )
        if show_progress:
            _selection_progress("读取点时指数成分股", started_at, detail=f"{index_codes}")
        panel = _selection_index_universe(index_codes, target_dates)
        return (
            {"type": "index", "index_codes": index_codes},
            panel,
            sorted(panel["instrument"].unique().tolist()),
        )
    if universe_type in {"market_cap_groups", "market_cap"}:
        group_count = _selection_positive_integer(
            universe.get("group_count", 15), "universe['group_count']"
        )
        selected_groups = universe.get("selected_groups")
        if not isinstance(selected_groups, Sequence) or isinstance(selected_groups, (str, bytes)):
            raise TypeError("universe['selected_groups'] 必须是市值组编号序列。")
        selected_groups = sorted(
            {
                _selection_positive_integer(value, "universe['selected_groups']")
                for value in selected_groups
            }
        )
        if not selected_groups or selected_groups[-1] > group_count:
            raise ValueError("selected_groups 必须位于 1 到 group_count 之间。")
        if show_progress:
            _selection_progress(
                "读取点时市值并划分股票池",
                started_at,
                detail=f"{group_count} 组，选择 {selected_groups}",
            )
        cap_panel = load_daily_raw_data(
            standard_fields=["total_market_cap"],
            dates=target_dates,
            instruments=None,
            show_progress=False,
        )
        required = {"date", "instrument", "total_market_cap"}
        if required - set(cap_panel.columns):
            raise ValueError("市值股票池数据缺少 date、instrument 或 total_market_cap。")
        cap_panel = cap_panel.loc[:, ["date", "instrument", "total_market_cap"]].copy()
        cap_panel["date"] = pd.to_datetime(cap_panel["date"], errors="coerce").dt.normalize()
        cap_panel["total_market_cap"] = pd.to_numeric(
            cap_panel["total_market_cap"], errors="coerce"
        )
        cap_panel = cap_panel.loc[
            cap_panel["date"].notna()
            & cap_panel["instrument"].notna()
            & cap_panel["total_market_cap"].gt(0)
        ].copy()
        if cap_panel.empty:
            raise ValueError("市值股票池没有有效市值记录。")
        parts = []
        for date, group in cap_panel.groupby("date", sort=True):
            ordered = group.sort_values("total_market_cap", kind="mergesort").copy()
            ordered["market_cap_group"] = np.ceil(
                np.arange(1, len(ordered) + 1) * group_count / len(ordered)
            ).astype(int)
            parts.append(
                ordered.loc[
                    ordered["market_cap_group"].isin(selected_groups),
                    ["date", "instrument"],
                ]
            )
        panel = pd.concat(parts, ignore_index=True).drop_duplicates(
            ["date", "instrument"]
        )
        missing_dates = target_dates.difference(pd.DatetimeIndex(panel["date"].unique()))
        if not missing_dates.empty:
            raise ValueError("部分筛选日期的指定市值组为空。")
        return (
            {
                "type": "market_cap_groups",
                "group_count": group_count,
                "selected_groups": selected_groups,
            },
            panel.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True),
            sorted(panel["instrument"].unique().tolist()),
        )
    raise ValueError("universe['type'] 仅支持 market_cap_groups、index、custom。")


def _filter_to_selection_universe(panel, universe_panel, panel_name):
    if universe_panel is None:
        return panel.copy()
    required = {"date", "instrument"}
    if not isinstance(panel, pd.DataFrame) or required - set(panel.columns):
        raise ValueError(f"{panel_name} 缺少 date 或 instrument。")
    result = panel.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result[["date", "instrument"]].isna().any().any():
        raise ValueError(f"{panel_name} 包含无效 date 或 instrument。")
    if result.duplicated(["date", "instrument"]).any():
        raise ValueError(f"{panel_name} 存在重复 date + instrument。")
    return result.merge(
        universe_panel.loc[:, ["date", "instrument"]],
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )


def _load_selection_calendar(start_date, end_date, history_days, label_horizon_days):
    from factor_lib.common.data_adapters.bigquant_adapters.loader import (
        load_trading_dates,
    )

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("selection_start_date 不能晚于 selection_end_date。")
    span = max(365, int((history_days + label_horizon_days + 20) * 2))
    for _ in range(7):
        calendar = load_trading_dates(
            start - pd.Timedelta(days=span), end + pd.Timedelta(days=span)
        )
        calendar = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
        selected = calendar[(calendar >= start) & (calendar <= end)]
        if not selected.empty:
            positions = {date: index for index, date in enumerate(calendar)}
            first = positions[selected.min()]
            last = positions[selected.max()]
            if first >= history_days and last + label_horizon_days < len(calendar):
                return calendar, selected
        span *= 2
    raise ValueError("筛选区间前预热或区间后未来收益标签交易日不足。")


def _build_validation_samples(
    feature_panel,
    labels,
    feature_names,
    validation_dates,
    calendar,
    sequence_length,
    missing_data_config,
    show_progress,
    started_at,
):
    """按验证截面准备模型输入，保留 date/instrument 用于截面评价。"""
    positions = {date: index for index, date in enumerate(calendar)}
    label_lookup = labels.set_index(["date", "instrument"])["target"]
    samples = []
    targets = []
    dates = []
    instruments = []
    try:
        for position, target_date in enumerate(validation_dates, start=1):
            end_position = positions[target_date]
            sequence_dates = calendar[
                end_position - sequence_length + 1 : end_position + 1
            ]
            codes, sequences = _build_prediction_sequences(
                feature_panel,
                feature_names,
                sequence_dates,
                missing_data_config,
            )
            if codes:
                section = pd.DataFrame(
                    {
                        "date": target_date,
                        "instrument": codes,
                        "sequence_position": np.arange(len(codes)),
                    }
                )
                section["target"] = [
                    label_lookup.get((target_date, code), np.nan) for code in codes
                ]
                section = section.loc[np.isfinite(section["target"])].copy()
                if not section.empty:
                    order = section["sequence_position"].to_numpy(dtype=int)
                    samples.append(sequences[order])
                    targets.append(section["target"].to_numpy(dtype=np.float32))
                    dates.extend([target_date] * len(section))
                    instruments.extend(section["instrument"].astype(str).tolist())
            if show_progress:
                _selection_progress(
                    "准备验证期时序样本",
                    started_at,
                    position,
                    len(validation_dates),
                    detail=f"当前 {target_date:%Y-%m-%d}，累计 {len(instruments):,} 条",
                )
        if not samples:
            raise ValueError("验证期没有可用于 GAN-GRU 推理的完整时序样本。")
        return (
            np.concatenate(samples, axis=0).astype(np.float32, copy=False),
            np.concatenate(targets).astype(np.float32, copy=False),
            pd.DatetimeIndex(dates),
            np.asarray(instruments, dtype=object),
        )
    finally:
        if show_progress:
            print()


def _prune_candidate_pool_by_coverage(
    feature_spec,
    calendar,
    schedule_dates,
    universe_panel,
    instruments,
    sequence_length,
    minimum_coverage,
    minimum_feature_count,
    show_progress,
    started_at,
):
    """按完整序列交集覆盖率逐步压缩候选池。

    这是特征筛选前的质量门槛，不涉及模型训练。每一步仅移除“移除后能使
    全体交集覆盖率提升最多”的一个候选；同分时优先移除自身覆盖率更低者。
    因此规则确定、可审计，也不会凭经验静默删因子。
    """
    if not feature_spec:
        raise ValueError("候选池不能为空。")
    positions = {date: index for index, date in enumerate(calendar)}
    first_position = positions[schedule_dates.min()] - sequence_length + 1
    if first_position < 0:
        raise ValueError("候选池覆盖率审计缺少序列预热日期。")
    feature_dates = calendar[first_position : positions[schedule_dates.max()] + 1]
    target_panel = universe_panel.loc[
        universe_panel["date"].isin(schedule_dates), ["date", "instrument"]
    ].drop_duplicates()
    target_index = pd.MultiIndex.from_frame(target_panel)
    if target_index.empty:
        raise ValueError("候选池覆盖率审计的动态股票池为空。")

    if show_progress:
        _selection_progress(
            "审计候选池完整序列覆盖率",
            started_at,
            detail=f"候选 {len(feature_spec)} 个，序列 {sequence_length} 日",
        )
    feature_panel = _load_training_features(
        feature_spec,
        feature_dates,
        calendar,
        instruments,
        show_progress,
    )
    flags = []
    for position, item in enumerate(feature_spec, start=1):
        matrix = (
            feature_panel.pivot(
                index="date",
                columns="instrument",
                values=item["feature_name"],
            )
            .reindex(index=feature_dates, columns=instruments)
        )
        ready = (
            matrix.notna().astype(int)
            .rolling(sequence_length, min_periods=sequence_length)
            .sum()
            .eq(sequence_length)
            .reindex(schedule_dates)
            .stack()
        )
        flags.append(ready.reindex(target_index).fillna(False).to_numpy(dtype=bool))
        if show_progress:
            _selection_progress(
                "审计候选池完整序列覆盖率",
                started_at,
                position,
                len(feature_spec),
                detail=item["feature_name"],
            )

    flag_matrix = np.column_stack(flags)
    active = list(range(len(feature_spec)))

    def coverage(indices):
        return float(flag_matrix[:, indices].all(axis=1).mean())

    initial_coverage = coverage(active)
    audit_rows = []
    while (
        coverage(active) < minimum_coverage
        and len(active) > minimum_feature_count
    ):
        before = coverage(active)
        options = []
        for remove_index in active:
            remaining = [index for index in active if index != remove_index]
            after = coverage(remaining)
            own_coverage = float(flag_matrix[:, remove_index].mean())
            options.append((after, -own_coverage, feature_spec[remove_index]["feature_name"], remove_index))
        # after 越高越好；自身覆盖率越低越优先移除；名称保证最终平局稳定。
        after, _, _, remove_index = sorted(
            options, key=lambda item: (-item[0], -item[1], item[2])
        )[0]
        item = feature_spec[remove_index]
        audit_rows.append(
            {
                "step": len(audit_rows) + 1,
                "removed_feature_name": item["feature_name"],
                "removed_factor_name": item["factor_name"],
                "removed_params": dict(item["params"]),
                "coverage_before": before,
                "coverage_after": after,
                "coverage_gain": after - before,
                "removed_single_feature_coverage": float(
                    flag_matrix[:, remove_index].mean()
                ),
                "remaining_feature_count": len(active) - 1,
            }
        )
        active.remove(remove_index)
        if show_progress:
            _selection_progress(
                "按覆盖率移除候选特征",
                started_at,
                len(audit_rows),
                len(feature_spec) - minimum_feature_count,
                detail=(
                    f"移除 {item['feature_name']}，交集覆盖率 "
                    f"{before:.2%} → {after:.2%}"
                ),
            )

    final_coverage = coverage(active)
    if final_coverage < minimum_coverage:
        raise ValueError(
            "即使已按规则移除候选特征，完整序列交集覆盖率仍未达到门槛："
            f"{final_coverage:.2%} < {minimum_coverage:.2%}。"
        )
    selected = [feature_spec[index] for index in active]
    return selected, pd.DataFrame(audit_rows), {
        "initial_coverage": initial_coverage,
        "final_coverage": final_coverage,
        "minimum_coverage": minimum_coverage,
        "initial_feature_count": len(feature_spec),
        "remaining_feature_count": len(selected),
        "target_stock_date_count": len(target_index),
    }


def _make_inference_context(model_bundle, cpu_threads):
    bundle = _validate_model_bundle(model_bundle)
    torch, _, _, _, _ = _require_torch()
    torch.set_num_threads(int(cpu_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _build_gru_model(bundle["model_config"])
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return bundle, torch, model, device


def _predict_from_inference_context(
    context,
    sequences,
    batch_size,
    show_progress=False,
    started_at=None,
    stage="验证期推理",
    detail="",
):
    bundle, torch, model, device = context
    standardized = _transform_sequences_for_model(sequences, bundle)
    started = time.perf_counter() if started_at is None else started_at
    results = []
    total = int(math.ceil(len(standardized) / batch_size))
    with torch.no_grad():
        for position, start in enumerate(range(0, len(standardized), batch_size), start=1):
            values = torch.tensor(
                standardized[start : start + batch_size], dtype=torch.float32, device=device
            )
            results.append(model(values).detach().cpu().numpy().reshape(-1))
            if show_progress:
                _selection_progress(
                    stage,
                    started,
                    position,
                    total,
                    detail=detail,
                )
    if show_progress:
        print()
    return np.concatenate(results).astype(float, copy=False)


def _newey_west_mean_statistics(values, lags):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if count == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if count == 1:
        return mean, np.nan, np.nan
    centered = values - mean
    long_run_variance = float(np.mean(centered * centered))
    maximum_lag = min(int(lags), count - 1)
    for lag in range(1, maximum_lag + 1):
        weight = 1.0 - lag / (maximum_lag + 1.0)
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / count)
    t_value = mean / standard_error if standard_error > 0 else np.nan
    return mean, standard_error, t_value


def _validation_metrics(
    dates,
    instruments,
    targets,
    scores,
    expected_label_count,
    minimum_stocks,
    top_quantile,
    hac_lags,
    lcb_z,
):
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "instrument": instruments,
            "target": np.asarray(targets, dtype=float),
            "score": np.asarray(scores, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    rankic_rows = []
    diagnostics = []
    for date, group in frame.groupby("date", sort=True):
        if len(group) < minimum_stocks:
            continue
        rank_score = group["score"].rank(method="average").to_numpy()
        rank_target = group["target"].rank(method="average").to_numpy()
        if np.std(rank_score) == 0 or np.std(rank_target) == 0:
            continue
        rank_ic = float(np.corrcoef(rank_score, rank_target)[0, 1])
        selected_count = max(1, int(math.ceil(len(group) * top_quantile)))
        ordered = group.sort_values("score", ascending=False, kind="mergesort")
        top = ordered.iloc[:selected_count]
        bottom = ordered.iloc[-selected_count:]
        rankic_rows.append({"date": date, "rank_ic": rank_ic, "sample_count": len(group)})
        diagnostics.append(
            {
                "date": date,
                "sample_count": len(group),
                "rank_ic": rank_ic,
                "top_count": selected_count,
                "top_quantile_return": float(top["target"].mean()),
                "universe_mean_return": float(group["target"].mean()),
                "top_quantile_excess": float(top["target"].mean() - group["target"].mean()),
                "top_bottom_spread": float(top["target"].mean() - bottom["target"].mean()),
                "top_positive": bool(top["target"].mean() > 0),
            }
        )
    rankic_series = pd.DataFrame(rankic_rows)
    diagnostic_frame = pd.DataFrame(diagnostics)
    values = rankic_series["rank_ic"].to_numpy(dtype=float) if not rankic_series.empty else np.array([])
    mean, standard_error, hac_t = _newey_west_mean_statistics(values, hac_lags)
    return {
        "rankic_series": rankic_series,
        "quantile_diagnostics": diagnostic_frame,
        "rank_ic_mean": mean,
        "rank_ic_hac_se": standard_error,
        "rank_ic_hac_t": hac_t,
        "rank_ic_lcb": mean - lcb_z * standard_error if np.isfinite(standard_error) else np.nan,
        "positive_rankic_ratio": float(np.mean(values > 0)) if len(values) else np.nan,
        "valid_date_count": int(len(rankic_series)),
        "coverage": float(len(frame) / expected_label_count) if expected_label_count else 0.0,
        "top_quantile_return_mean": (
            float(diagnostic_frame["top_quantile_return"].mean()) if not diagnostic_frame.empty else np.nan
        ),
        "top_quantile_excess_mean": (
            float(diagnostic_frame["top_quantile_excess"].mean()) if not diagnostic_frame.empty else np.nan
        ),
        "top_bottom_spread_mean": (
            float(diagnostic_frame["top_bottom_spread"].mean()) if not diagnostic_frame.empty else np.nan
        ),
        "top_positive_date_ratio": (
            float(diagnostic_frame["top_positive"].mean()) if not diagnostic_frame.empty else np.nan
        ),
    }


def _selection_metrics_row(name, metrics):
    return {
        "feature_set_id": name,
        "feature_count": None,
        "rank_ic_mean": metrics["rank_ic_mean"],
        "rank_ic_hac_se": metrics["rank_ic_hac_se"],
        "rank_ic_hac_t": metrics["rank_ic_hac_t"],
        "rank_ic_lcb": metrics["rank_ic_lcb"],
        "positive_rankic_ratio": metrics["positive_rankic_ratio"],
        "valid_date_count": metrics["valid_date_count"],
        "coverage": metrics["coverage"],
        "top_quantile_excess_mean": metrics["top_quantile_excess_mean"],
        "top_bottom_spread_mean": metrics["top_bottom_spread_mean"],
    }


def select_gan_gru_features(
    *,
    selection_start_date,
    selection_end_date,
    candidate_spec=None,
    candidate_source=None,
    universe=None,
    evaluation_interval_days=5,
    label_horizon_days=None,
    training_config=None,
    selection_config=None,
    show_progress=True,
):
    """为 GAN-GRU 选择模型相关、低冗余的基础或复合因子特征。

    ``candidate_spec`` 用于手工指定初始池；例如
    ``{"return_nd": [{"window": 5}], "book_to_price": None}``。
    ``candidate_source`` 可传入 ``"base_factors"``、``"composite_factors"``
    或二者的列表，函数会读取各因子的 ``candidate_instances`` 展开实例。

    本函数只使用 selection 区间内的信号日训练和验证。后续回测起止日的
    选择由调用者负责；函数不把 selection 区间与未来回测期作任何隐式绑定。
    """
    started_at = time.perf_counter()
    try:
        start = pd.Timestamp(selection_start_date).normalize()
        end = pd.Timestamp(selection_end_date).normalize()
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("selection_start_date 和 selection_end_date 必须是递增有效日期。")
        interval = _selection_positive_integer(
            evaluation_interval_days, "evaluation_interval_days"
        )
        config = _normalize_feature_selection_config(selection_config)
        full_training_config = _normalize_training_config(training_config)
        horizon = (
            full_training_config["label_horizon_days"]
            if label_horizon_days is None
            else _selection_positive_integer(label_horizon_days, "label_horizon_days")
        )
        if horizon != full_training_config["label_horizon_days"]:
            full_training_config = dict(full_training_config)
            full_training_config["label_horizon_days"] = horizon
        screening_training_config = dict(full_training_config)
        screening_training_config.update(
            {
                "gan_epochs": min(
                    screening_training_config["gan_epochs"], config["screening_gan_epochs"]
                ),
                "gru_max_epochs": min(
                    screening_training_config["gru_max_epochs"], config["screening_gru_max_epochs"]
                ),
                "early_stop_patience": min(
                    screening_training_config["early_stop_patience"], config["screening_early_stop_patience"]
                ),
            }
        )

        if show_progress:
            _selection_progress("展开初始候选因子池", started_at)
        candidates = _expand_feature_selection_candidates(
            candidate_spec, candidate_source
        )
        full_feature_spec = _normalize_feature_spec(
            [
                {
                    "factor_name": item["factor_name"],
                    "params": item["params"],
                    "feature_name": item["feature_name"],
                }
                for item in candidates
            ]
        )

        from factor_lib.common.data_adapters.bigquant_adapters.loader import (
            get_factor_data_requirements,
        )

        maximum_history = full_training_config["sequence_length"] - 1
        for item in full_feature_spec:
            requirements = get_factor_data_requirements(
                item["factor_name"], item["params"]
            )
            maximum_history = max(
                maximum_history + 0,
                full_training_config["sequence_length"] - 1
                + int(requirements["data_window"]["lookback_trading_days"]),
            )
        if show_progress:
            _selection_progress("读取交易日历并生成筛选计划", started_at)
        calendar, available_dates = _load_selection_calendar(
            start, end, maximum_history, horizon
        )
        schedule_dates = available_dates[::interval]
        if len(schedule_dates) < 8:
            raise ValueError("筛选区间的有效信号日不足 8 个；请扩大日期区间或缩短频率。")
        positions = {date: index for index, date in enumerate(calendar)}

        if show_progress:
            _selection_progress(
                "读取并校验点时股票池",
                started_at,
                detail=f"{len(schedule_dates)} 个信号日",
            )
        universe_config, universe_panel, load_instruments = _resolve_selection_universe(
            universe, available_dates, show_progress, started_at
        )
        if load_instruments is not None and not load_instruments:
            raise ValueError("动态股票池没有可加载的股票代码。")

        # 严格覆盖率门槛只服务于“候选池压缩”：它不训练模型，也不改动基础
        # 因子定义。压缩完成后，正式模型仍可按 training_config 选择 strict
        # 或 impute_with_mask 模式。
        (
            full_feature_spec,
            candidate_pool_pruning_audit,
            candidate_pool_coverage,
        ) = _prune_candidate_pool_by_coverage(
            full_feature_spec,
            calendar,
            schedule_dates,
            universe_panel,
            load_instruments,
            full_training_config["sequence_length"],
            config["candidate_pool_min_coverage"],
            config["candidate_pool_min_feature_count"],
            show_progress,
            started_at,
        )
        kept_feature_names = {
            item["feature_name"] for item in full_feature_spec
        }
        candidates = [
            item for item in candidates
            if item["feature_name"] in kept_feature_names
        ]
        if show_progress:
            _selection_progress(
                "候选池覆盖率压缩完成",
                started_at,
                detail=(
                    f"{candidate_pool_coverage['initial_feature_count']} → "
                    f"{candidate_pool_coverage['remaining_feature_count']} 个特征，"
                    f"覆盖率 {candidate_pool_coverage['initial_coverage']:.2%} → "
                    f"{candidate_pool_coverage['final_coverage']:.2%}"
                ),
            )

        validation_count = max(2, int(math.ceil(len(schedule_dates) * (1.0 - config["train_ratio"]))))
        if validation_count >= len(schedule_dates):
            raise ValueError("筛选区间不足以划分训练期和验证期。")
        validation_dates = schedule_dates[-validation_count:]
        validation_start_position = positions[validation_dates.min()]
        additional_purge = config["additional_purge_trading_days"]
        train_dates = pd.DatetimeIndex(
            [
                date
                for date in schedule_dates
                if positions[date] + horizon + additional_purge < validation_start_position
            ]
        )
        if len(train_dates) < 4:
            raise ValueError("按标签期限净化后训练期不足；请扩大筛选区间。")

        train_end = train_dates.max()
        training_anchor = calendar[positions[train_end] + horizon]
        if show_progress:
            _selection_progress(
                "训练筛选用 GAN-GRU 模型",
                started_at,
                detail=(
                    f"候选特征 {len(full_feature_spec)} 个，训练 {len(train_dates)} 日，"
                    f"验证 {len(validation_dates)} 日"
                ),
            )
        screening_bundle = train_gan_gru_model(
            anchor_date=training_anchor,
            feature_spec=full_feature_spec,
            instruments=load_instruments,
            training_config=screening_training_config,
            training_start_date=train_dates.min(),
            training_end_date=train_end,
            universe_panel=universe_panel,
            show_progress=show_progress,
            progress_every=1,
        )

        feature_start_position = validation_start_position - full_training_config["sequence_length"] + 1
        if feature_start_position < 0:
            raise ValueError("验证期缺少 GAN-GRU 推理序列预热数据。")
        feature_dates = calendar[
            feature_start_position : positions[validation_dates.max()] + 1
        ]
        if show_progress:
            _selection_progress("准备验证期特征和完整未来收益标签", started_at)
        validation_feature_panel = _load_training_features(
            full_feature_spec,
            feature_dates,
            calendar,
            load_instruments,
            show_progress,
        )
        validation_labels = _load_forward_labels(
            validation_dates,
            calendar,
            load_instruments,
            horizon,
            show_progress=show_progress,
        )
        validation_labels = _filter_to_selection_universe(
            validation_labels, universe_panel, "validation_labels"
        )
        expected_label_count = len(validation_labels)
        if expected_label_count == 0:
            raise ValueError("按股票池过滤后验证期没有完整未来收益标签。")
        validation_x, validation_y, validation_sample_dates, validation_instruments = _build_validation_samples(
            validation_feature_panel,
            validation_labels,
            [item["feature_name"] for item in full_feature_spec],
            validation_dates,
            calendar,
            full_training_config["sequence_length"],
            full_training_config["missing_data_config"],
            show_progress,
            started_at,
        )
        if config["hac_lags"] is None:
            hac_lags = max(0, int(math.ceil(horizon / interval)) - 1)
        else:
            hac_lags = config["hac_lags"]
        inference_context = _make_inference_context(
            screening_bundle, full_training_config["cpu_threads"]
        )
        baseline_scores = _predict_from_inference_context(
            inference_context,
            validation_x,
            full_training_config["batch_size"],
            show_progress=show_progress,
            started_at=started_at,
            stage="筛选模型验证期基准推理",
        )
        baseline_metrics = _validation_metrics(
            validation_sample_dates,
            validation_instruments,
            validation_y,
            baseline_scores,
            expected_label_count,
            config["min_rankic_stocks"],
            config["top_quantile"],
            hac_lags,
            config["rankic_lcb_z"],
        )

        if show_progress:
            _selection_progress(
                "执行特征置换重要性检验",
                started_at,
                completed=0,
                total=len(full_feature_spec) * config["permutation_repeats"],
            )
        date_indices = [
            positions
            for _, positions in pd.Series(
                np.arange(len(validation_sample_dates)), index=validation_sample_dates
            ).groupby(level=0)
        ]
        rng = np.random.default_rng(full_training_config["random_seed"])
        importance_rows = []
        total_tests = len(full_feature_spec) * config["permutation_repeats"]
        test_position = 0
        for feature_index, item in enumerate(full_feature_spec):
            impacts = []
            for repeat in range(1, config["permutation_repeats"] + 1):
                test_position += 1
                permuted = validation_x.copy()
                for indices in date_indices:
                    if len(indices) > 1:
                        permuted[indices, :, feature_index] = permuted[
                            rng.permutation(indices), :, feature_index
                        ]
                scores = _predict_from_inference_context(
                    inference_context,
                    permuted,
                    full_training_config["batch_size"],
                    show_progress=show_progress,
                    started_at=started_at,
                    stage="置换重要性推理",
                    detail=f"{item['feature_name']}，重复 {repeat}/{config['permutation_repeats']}",
                )
                metrics = _validation_metrics(
                    validation_sample_dates,
                    validation_instruments,
                    validation_y,
                    scores,
                    expected_label_count,
                    config["min_rankic_stocks"],
                    config["top_quantile"],
                    hac_lags,
                    config["rankic_lcb_z"],
                )
                impact = baseline_metrics["rank_ic_lcb"] - metrics["rank_ic_lcb"]
                impacts.append(impact)
                if show_progress:
                    _selection_progress(
                        "执行特征置换重要性检验",
                        started_at,
                        test_position,
                        total_tests,
                        detail=f"当前 {item['feature_name']}，重复 {repeat}",
                    )
            candidate = candidates[feature_index]
            importance_rows.append(
                {
                    **candidate,
                    "importance_mean": float(np.nanmean(impacts)),
                    "importance_std": float(np.nanstd(impacts, ddof=0)),
                    "permutation_impacts": impacts,
                }
            )
        importance = pd.DataFrame(importance_rows).sort_values(
            ["importance_mean", "importance_std", "candidate_id"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        importance["importance_rank"] = np.arange(1, len(importance) + 1)

        max_features = (
            len(importance)
            if config["max_selected_features"] is None
            else min(config["max_selected_features"], len(importance))
        )
        ranked_feature_spec = [
            {
                "factor_name": row.factor_name,
                "params": dict(row.params),
                "feature_name": row.feature_name,
            }
            for row in importance.itertuples(index=False)
        ]
        final_counts = sorted(
            {
                count
                for count in config["final_feature_counts"]
                if count <= max_features
            }
        )
        if not final_counts:
            final_counts = [max_features]

        final_rows = []
        final_details = []
        for position, count in enumerate(final_counts, start=1):
            subset = ranked_feature_spec[:count]
            if show_progress:
                _selection_progress(
                    "复训确认候选特征集合",
                    started_at,
                    position - 1,
                    len(final_counts),
                    detail=f"前 {count} 个特征",
                )
            bundle = train_gan_gru_model(
                anchor_date=training_anchor,
                feature_spec=subset,
                instruments=load_instruments,
                training_config=full_training_config,
                training_start_date=train_dates.min(),
                training_end_date=train_end,
                universe_panel=universe_panel,
                show_progress=show_progress,
                progress_every=1,
            )
            names = [item["feature_name"] for item in subset]

            # 重要：不能从“全部候选因子共同完整”的 validation_x 中切列。
            # 否则任一未入选候选因子的缺失都会错误降低当前子集的覆盖率。
            # 正式确认必须仅按 subset 自己的字段重新构建验证期面板与序列。
            if show_progress:
                _selection_progress(
                    "重建复训集合验证样本",
                    started_at,
                    position - 1,
                    len(final_counts),
                    detail=f"前 {count} 个特征，独立计算验证期面板",
                )
            subset_validation_feature_panel = _load_training_features(
                subset,
                feature_dates,
                calendar,
                load_instruments,
                show_progress,
            )
            (
                subset_x,
                subset_y,
                subset_sample_dates,
                subset_instruments,
            ) = _build_validation_samples(
                subset_validation_feature_panel,
                validation_labels,
                names,
                validation_dates,
                calendar,
                full_training_config["sequence_length"],
                full_training_config["missing_data_config"],
                show_progress,
                started_at,
            )
            if len(subset_x) == 0:
                raise ValueError(
                    f"前 {count} 个特征在验证期没有任何完整模型序列。"
                )
            context = _make_inference_context(bundle, full_training_config["cpu_threads"])
            scores = _predict_from_inference_context(
                context,
                subset_x,
                full_training_config["batch_size"],
                show_progress=show_progress,
                started_at=started_at,
                stage="复训集合验证期推理",
                detail=f"前 {count} 个特征",
            )
            metrics = _validation_metrics(
                subset_sample_dates,
                subset_instruments,
                subset_y,
                scores,
                expected_label_count,
                config["min_rankic_stocks"],
                config["top_quantile"],
                hac_lags,
                config["rankic_lcb_z"],
            )
            row = _selection_metrics_row(f"top_{count}", metrics)
            row["feature_count"] = count
            row["feature_names"] = names
            row["validation_sample_count"] = len(subset_y)
            final_rows.append(row)
            final_details.append(
                {
                    "feature_spec": subset,
                    "metrics": metrics,
                    "model_bundle": bundle,
                    "validation_sample_count": len(subset_y),
                }
            )
            if show_progress:
                _selection_progress(
                    "复训确认候选特征集合",
                    started_at,
                    position,
                    len(final_counts),
                    detail=f"前 {count} 个特征，RankIC-LCB={metrics['rank_ic_lcb']:.6f}",
                )
        final_ranking = pd.DataFrame(final_rows).sort_values(
            ["rank_ic_lcb", "rank_ic_mean", "positive_rankic_ratio", "coverage"],
            ascending=[False, False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        best_id = final_ranking.iloc[0]["feature_set_id"]
        best_detail = next(
            item for item in final_details if f"top_{len(item['feature_spec'])}" == best_id
        )
        qualification = {
            "coverage_passed": bool(best_detail["metrics"]["coverage"] >= config["min_coverage"]),
            "positive_rankic_ratio_passed": bool(
                best_detail["metrics"]["positive_rankic_ratio"] >= config["min_positive_rankic_ratio"]
            ),
            "rank_ic_positive": bool(best_detail["metrics"]["rank_ic_mean"] > 0),
        }
        qualification["passed"] = bool(all(qualification.values()))

        interactions = pd.DataFrame(
            columns=["feature_a", "feature_b", "impact_a", "impact_b", "joint_impact", "interaction"]
        )
        if config["max_interaction_pairs"] > 0 and len(importance) > 1:
            interaction_features = importance.head(max_features).reset_index(drop=True)
            pairs = [
                (left, right)
                for left in range(len(interaction_features))
                for right in range(left + 1, len(interaction_features))
            ][: config["max_interaction_pairs"]]
            rows = []
            for position, (left, right) in enumerate(pairs, start=1):
                working = validation_x.copy()
                for indices in date_indices:
                    if len(indices) > 1:
                        for feature_index in (left, right):
                            original_index = importance.loc[feature_index, "importance_rank"] - 1
                            working[indices, :, original_index] = working[
                                rng.permutation(indices), :, original_index
                            ]
                scores = _predict_from_inference_context(
                    inference_context, working, full_training_config["batch_size"],
                    show_progress=show_progress, started_at=started_at,
                    stage="特征交互置换推理",
                    detail=f"{position}/{len(pairs)}",
                )
                metrics = _validation_metrics(
                    validation_sample_dates, validation_instruments, validation_y, scores,
                    expected_label_count, config["min_rankic_stocks"], config["top_quantile"],
                    hac_lags, config["rankic_lcb_z"],
                )
                joint_impact = baseline_metrics["rank_ic_lcb"] - metrics["rank_ic_lcb"]
                impact_left = interaction_features.loc[left, "importance_mean"]
                impact_right = interaction_features.loc[right, "importance_mean"]
                rows.append(
                    {
                        "feature_a": interaction_features.loc[left, "feature_name"],
                        "feature_b": interaction_features.loc[right, "feature_name"],
                        "impact_a": impact_left,
                        "impact_b": impact_right,
                        "joint_impact": joint_impact,
                        "interaction": joint_impact - impact_left - impact_right,
                    }
                )
            interactions = pd.DataFrame(rows)

        if show_progress:
            _selection_progress(
                "特征筛选完成",
                started_at,
                len(final_counts),
                len(final_counts),
                detail=(
                    f"选择 {len(best_detail['feature_spec'])} 个特征，"
                    f"RankIC-LCB={best_detail['metrics']['rank_ic_lcb']:.6f}"
                ),
            )
        return {
            "selected_feature_spec": copy.deepcopy(best_detail["feature_spec"]),
            "selection_passed": qualification["passed"],
            "qualification": qualification,
            "feature_importance": importance,
            "final_feature_set_ranking": final_ranking,
            "screening_candidate_metrics": baseline_metrics,
            "baseline_metrics": baseline_metrics,
            "selected_metrics": best_detail["metrics"],
            "selected_rankic_series": best_detail["metrics"]["rankic_series"],
            "selected_quantile_diagnostics": best_detail["metrics"]["quantile_diagnostics"],
            "interaction_results": interactions,
            "candidate_pool_coverage": candidate_pool_coverage,
            "candidate_pool_pruning_audit": candidate_pool_pruning_audit,
            "resolved_candidates": pd.DataFrame(candidates),
            "universe_config": universe_config,
            "universe_panel": universe_panel,
            "selection_schedule": pd.DataFrame({"date": schedule_dates}),
            "train_dates": train_dates,
            "validation_dates": validation_dates,
            "training_anchor_date": training_anchor,
            "selection_config": config,
            "training_config": full_training_config,
            "screening_training_config": screening_training_config,
            "label_horizon_days": horizon,
            "hac_lags": hac_lags,
        }
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
            "missing_data_config": bundle["missing_data_config"],
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
        "missing_data_config": state.get("missing_data_config"),
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
    "factor_type": "machine_learning",
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
        # missing_feature_mode='strict' 保持传统完整序列口径；
        # 'impute_with_mask' 使用训练期均值填补缺失，并把缺失掩码拼接入模型输入。
        # minimum_observed_value_ratio 限制单条序列最少真实观测比例。
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

## 训练前的特征筛选

`select_gan_gru_features()` 是本机器学习因子的配套研究函数，不属于因子计算接口。
它先以候选特征训练一套低预算筛选模型，再在验证期逐特征进行截面内置换推理，使用
RankIC-LCB 的下降幅度衡量模型对该特征的依赖；随后只对少量前 N 个特征集合按正式训练
参数复训确认。正式复训确认会仅根据该特征集合重新加载验证期面板、构造序列并计算覆盖率，
不会继承“全部候选特征共同完整样本”的覆盖率。返回的 `selected_feature_spec` 可直接作为
本因子滚动训练模式的 `feature_spec`。

在训练筛选模型之前，函数先按 `candidate_pool_min_coverage` 审计所有候选的严格完整
序列交集覆盖率。若不达标，会逐次移除“移除后交集覆盖率提升最多”的候选，直至达到门槛；
全过程返回 `candidate_pool_pruning_audit` 和 `candidate_pool_coverage`，不会静默删因子。
这一步仅处理候选池，不修改任何基础因子的经济定义。

## 缺失特征放宽模式

`training_config["missing_feature_mode"]` 可选：

- `"strict"`：默认值。任一特征在序列内缺失，整条股票序列不参与训练或推理；与旧版结果兼容。
- `"impute_with_mask"`：允许部分特征缺失。每个缺失值仅用训练子集对应特征的均值填补，
  同时向 GRU 追加同维度缺失掩码（1 表示填补、0 表示原始值）。模型因此不会把填补值误认为
  真实观测，也不会使用验证期、回测期或推理期数据重新拟合填补值。

放宽模式仍受 `minimum_observed_value_ratio` 约束，默认至少 50% 的“时间 × 特征”位置必须有
真实值；完全或几乎完全缺失的序列仍不产生分数。模型包会固化该模式与训练期 scaler，固定模型
推理及滚动更新均必须沿用同一协议。

候选池有两种来源，可同时使用并自动去重：

```python
# 手工指定：None 表示展开该因子的 FACTOR['candidate_instances']。
candidate_spec = {
    "return_nd": [{"window": 5}, {"window": 20}],
    "book_to_price": None,
}

# 由 FACTOR['factor_type'] 批量发现所有登记实例。
candidate_source = ["base_factors", "composite_factors"]
```

筛选必须明确传入 `universe`，支持：

- `{"type": "market_cap_groups", "group_count": 15, "selected_groups": [1, 2]}`；每天按当日市值重新分组。
- `{"type": "index", "index_codes": ["000300.SH"]}`；每天使用历史指数成分股。
- `{"type": "custom", "instruments": [...]}`；使用指定代码集合。

它不会接收未来回测起止日，也不会替调用者决定筛选期与回测期的边界；调用者必须在正式回测
开始前冻结 `selected_feature_spec`。筛选期间只使用已完成的历史未来收益标签。`top_quantile`
诊断仅按模型分数取每个验证截面最高分股票，并统计其已实现未来收益，不涉及交易、成本或策略回测。

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
