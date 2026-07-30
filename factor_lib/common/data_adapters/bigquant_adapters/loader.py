# -*- coding: utf-8 -*-
"""BigQuant 多数据域加载统筹器。

职责：
1. 动态发现因子并读取 FACTOR 元数据；
2. 合并因子默认参数与调用参数；
3. 解析条件字段和固定/动态数据窗口；
4. 根据字段声明的数据域调用 daily、financial 等适配器；
5. 按 date + instrument 合并日频点时适配器输出。

loader 不计算因子、不生成目标日期、不决定调仓日，也不执行策略。
策略或研究层负责生成需要查询的连续区间或离散日期列表。

注意：
当前 loader 合并的是能够输出 date + instrument 日频点时面板的
适配器。非日频原始财报不能直接进入该合并流程；未来应由独立适配器
保留 report_date、announcement_date/available_date 等时间字段，并
通过专门的点时数据准备环节与目标日期对齐。
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np
import pandas as pd

from factor_lib.common.data_adapters.bigquant_adapters.daily import (
    load_daily_raw_data,
)
from factor_lib.common.data_adapters.bigquant_adapters.financial import (
    load_financial_raw_data,
)
from factor_lib.factor_hub.discover_factors import discover_factors


ADAPTER_REGISTRY = {
    "daily": load_daily_raw_data,
    "financial": load_financial_raw_data,
}


def get_factor_metadata(factor_name):
    """动态发现并返回指定因子的 FACTOR 元数据。"""
    if not isinstance(factor_name, str) or not factor_name.strip():
        raise ValueError("factor_name 必须是非空字符串。")

    discovered = discover_factors()
    name = factor_name.strip()

    if isinstance(discovered, Mapping):
        metadata = discovered.get(name)
        if metadata is not None:
            return metadata
    else:
        for metadata in discovered:
            if (
                isinstance(metadata, Mapping)
                and metadata.get("name") == name
            ):
                return metadata

    available = (
        sorted(discovered)
        if isinstance(discovered, Mapping)
        else sorted(
            item.get("name", "<unknown>")
            for item in discovered
            if isinstance(item, Mapping)
        )
    )
    raise KeyError(
        f"未发现因子 {name!r}。可用因子：{available}"
    )


def _resolved_factor_parameters(metadata, factor_params):
    if factor_params is None:
        factor_params = {}
    elif not isinstance(factor_params, Mapping):
        raise TypeError("factor_params 必须是字典或 None。")
    else:
        factor_params = dict(factor_params)

    parameters = metadata.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("FACTOR['parameters'] 必须是字典。")

    unknown_parameters = sorted(
        set(factor_params) - set(parameters)
    )
    if unknown_parameters:
        raise ValueError(
            f"因子 {metadata.get('name')!r} 收到未登记参数："
            f"{unknown_parameters}。请先在 FACTOR['parameters'] "
            "中声明这些参数。"
        )

    resolved = {}
    for name, specification in parameters.items():
        if (
            isinstance(specification, Mapping)
            and "default" in specification
        ):
            resolved[name] = specification["default"]

    resolved.update(factor_params)
    return resolved


def _condition_is_met(required_when, resolved_params):
    if not required_when:
        return True
    if not isinstance(required_when, Mapping):
        raise ValueError(
            "input_schema.conditional.required_when 必须是字典。"
        )
    return all(
        resolved_params.get(parameter_name) == expected_value
        for parameter_name, expected_value in required_when.items()
    )


def _resolve_field_data_domain(field_name, specification):
    """读取新 data_domain 写法，并兼容既有 frequency 写法。"""
    if not isinstance(specification, Mapping):
        specification = {}

    data_domain = specification.get("data_domain")
    legacy_frequency = specification.get("frequency")

    if (
        data_domain is not None
        and legacy_frequency is not None
        and data_domain != legacy_frequency
    ):
        raise ValueError(
            f"字段 {field_name!r} 的 data_domain 与 frequency 冲突："
            f"{data_domain!r} != {legacy_frequency!r}。"
        )

    if data_domain is None:
        data_domain = legacy_frequency
    if data_domain is None:
        data_domain = "daily"

    if not isinstance(data_domain, str) or not data_domain.strip():
        raise ValueError(
            f"字段 {field_name!r} 的 data_domain/frequency "
            "必须是非空字符串。"
        )
    return data_domain.strip()


def _add_field_requirement(result, field_name, specification):
    if field_name in {"date", "instrument"}:
        return

    data_domain = _resolve_field_data_domain(
        field_name,
        specification,
    )
    result.setdefault(data_domain, [])
    if field_name not in result[data_domain]:
        result[data_domain].append(field_name)


def _normalize_nonnegative_integer(value, field_name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{field_name} 必须是非负整数，不能是 bool。"
        )
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} 必须是非负整数。")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} 不能为负数。")
    return value


def _resolve_factor_data_window(metadata, resolved_params):
    """把固定或动态 data_window 解析为本次调用的纯数值窗口。"""
    specification = metadata.get("data_window", {})
    if not isinstance(specification, Mapping):
        raise ValueError(
            f"因子 {metadata.get('name')!r} 的 data_window 必须是字典。"
        )

    resolver = specification.get("resolver")
    if resolver is None:
        resolved_window = dict(specification)
    else:
        if not callable(resolver):
            raise ValueError(
                f"因子 {metadata.get('name')!r} 的 "
                "data_window.resolver 必须可调用。"
            )

        default_window = specification.get("default", {})
        if not isinstance(default_window, Mapping):
            raise ValueError(
                "动态 data_window.default 必须是字典。"
            )

        dynamic_window = resolver(dict(resolved_params))
        if not isinstance(dynamic_window, Mapping):
            raise ValueError(
                f"因子 {metadata.get('name')!r} 的窗口解析函数"
                "必须返回字典。"
            )

        resolved_window = dict(default_window)
        resolved_window.update(dynamic_window)

    lookback = _normalize_nonnegative_integer(
        resolved_window.get("lookback_trading_days", 0),
        "data_window.lookback_trading_days",
    )
    minimum_history = _normalize_nonnegative_integer(
        resolved_window.get("minimum_history_observations", 0),
        "data_window.minimum_history_observations",
    )

    requires_target_date_data = resolved_window.get(
        "requires_target_date_data",
        True,
    )
    preheating_required = resolved_window.get(
        "preheating_required",
        lookback > 0,
    )
    if not isinstance(
        requires_target_date_data,
        (bool, np.bool_),
    ):
        raise ValueError(
            "data_window.requires_target_date_data 必须是 bool。"
        )
    if not isinstance(
        preheating_required,
        (bool, np.bool_),
    ):
        raise ValueError(
            "data_window.preheating_required 必须是 bool。"
        )

    resolved_window["lookback_trading_days"] = lookback
    resolved_window["minimum_history_observations"] = (
        minimum_history
    )
    resolved_window["requires_target_date_data"] = bool(
        requires_target_date_data
    )
    resolved_window["preheating_required"] = bool(
        preheating_required
    )
    return resolved_window


def get_factor_data_requirements(
    factor_name,
    factor_params=None,
):
    """返回本次参数组合所需字段、适配器、窗口和完整参数。"""
    metadata = get_factor_metadata(factor_name)
    schema = metadata.get("input_schema")
    if not isinstance(schema, Mapping):
        raise ValueError(
            f"因子 {factor_name!r} 缺少规范的 input_schema，"
            "无法由 loader 自动调度数据适配器。"
        )

    resolved_params = _resolved_factor_parameters(
        metadata,
        factor_params,
    )
    fields_by_domain = {}

    required_schema = schema.get("required", {})
    if not isinstance(required_schema, Mapping):
        raise ValueError(
            "input_schema.required 必须是字典。"
        )
    for field_name, specification in required_schema.items():
        _add_field_requirement(
            fields_by_domain,
            field_name,
            specification,
        )

    conditional_schema = schema.get("conditional", {})
    if not isinstance(conditional_schema, Mapping):
        raise ValueError(
            "input_schema.conditional 必须是字典。"
        )
    for field_name, specification in conditional_schema.items():
        if not isinstance(specification, Mapping):
            raise ValueError(
                f"因子 {factor_name!r} 的条件字段"
                f"{field_name!r} 规范无效。"
            )
        if _condition_is_met(
            specification.get("required_when"),
            resolved_params,
        ):
            _add_field_requirement(
                fields_by_domain,
                field_name,
                specification,
            )

    resolved_window = _resolve_factor_data_window(
        metadata,
        resolved_params,
    )

    # fields_by_frequency 保留给现有策略代码使用；
    # fields_by_domain 是含义更准确的新名称。
    return {
        "factor_name": metadata.get("name", factor_name),
        "fields_by_domain": {
            domain: list(fields)
            for domain, fields in fields_by_domain.items()
        },
        "fields_by_frequency": {
            domain: list(fields)
            for domain, fields in fields_by_domain.items()
        },
        "data_window": resolved_window,
        "resolved_factor_params": resolved_params,
    }


def _normalize_adapter_output(panel, data_domain):
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(
            f"{data_domain!r} 适配器必须返回 pandas.DataFrame。"
        )

    required_keys = {"date", "instrument"}
    missing = required_keys - set(panel.columns)
    if missing:
        raise ValueError(
            f"{data_domain!r} 适配器输出缺少键字段："
            f"{sorted(missing)}。"
        )

    result = panel.copy()
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    ).dt.normalize()
    if result["date"].isna().any():
        raise ValueError(
            f"{data_domain!r} 适配器输出包含无效日期。"
        )
    if result["instrument"].isna().any():
        raise ValueError(
            f"{data_domain!r} 适配器输出包含空 instrument。"
        )

    duplicated = result.duplicated(
        ["date", "instrument"],
        keep=False,
    )
    if duplicated.any():
        examples = (
            result.loc[
                duplicated,
                ["date", "instrument"],
            ]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            f"{data_domain!r} 适配器输出存在重复的 "
            f"date + instrument：{examples}"
        )
    return result


def _merge_adapter_panels(panels):
    """按 date + instrument 合并不同日频点时数据域面板。"""
    if not panels:
        return pd.DataFrame(columns=["date", "instrument"])
    if len(panels) == 1:
        return panels[0]

    merged = panels[0]
    for panel in panels[1:]:
        overlapping = (
            set(merged.columns)
            & set(panel.columns)
            - {"date", "instrument"}
        )
        if overlapping:
            raise ValueError(
                "不同适配器返回了重复的标准字段："
                f"{sorted(overlapping)}。请确保一个标准字段只由"
                "一个数据域负责。"
            )

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


def _render_loader_progress(
    completed,
    total,
    data_domain,
    started_at,
    finished=False,
):
    elapsed = time.perf_counter() - started_at
    percentage = 100.0 if total == 0 else completed / total * 100.0

    if completed > 0 and completed < total:
        remaining = elapsed / completed * (total - completed)
        eta_text = f"，预计剩余 {remaining:.1f}s"
    else:
        eta_text = ""

    if finished:
        stage = "原始数据准备完成"
    elif completed == 0:
        stage = "准备加载"
    else:
        stage = f"已完成 {data_domain}"

    print(
        "\r[BigQuant loader] "
        f"{completed}/{total}（{percentage:6.2f}%）"
        f"，{stage}，耗时 {elapsed:.1f}s{eta_text}",
        end="",
        flush=True,
    )


def load_factor_raw_data(
    factor_name,
    start_date=None,
    end_date=None,
    dates=None,
    factor_params=None,
    instruments=None,
    adapter_overrides=None,
    show_progress=False,
):
    """根据 FACTOR 元数据调度适配器并拉取原始数据。

    日期选择必须二选一：
    - ``start_date`` + ``end_date``：连续数据覆盖区间；
    - ``dates``：离散数据日期列表。

    loader 只接收日期，不根据策略生成日期列表。因子的动态预热窗口可由
    ``get_factor_data_requirements`` 解析，但策略层仍负责据此生成最终
    需要查询的日期或区间。

    嵌套调用时由 loader 统一显示进度，内部适配器保持静默。
    """
    requirements = get_factor_data_requirements(
        factor_name,
        factor_params,
    )

    adapters = dict(ADAPTER_REGISTRY)
    if adapter_overrides is not None:
        if not isinstance(adapter_overrides, Mapping):
            raise TypeError(
                "adapter_overrides 必须是字典或 None。"
            )
        adapters.update(adapter_overrides)

    items = list(
        requirements["fields_by_domain"].items()
    )
    total_data_domains = len(items)
    panels = []
    started_at = time.perf_counter()

    if show_progress:
        _render_loader_progress(
            completed=0,
            total=total_data_domains,
            data_domain="",
            started_at=started_at,
        )

    try:
        for index, (data_domain, fields) in enumerate(
            items,
            start=1,
        ):
            adapter = adapters.get(data_domain)
            if adapter is None:
                raise NotImplementedError(
                    f"因子 {factor_name!r} 需要 {data_domain!r} "
                    "数据适配器，但 ADAPTER_REGISTRY 尚未登记。"
                )

            panel = adapter(
                standard_fields=fields,
                start_date=start_date,
                end_date=end_date,
                dates=dates,
                instruments=instruments,
                show_progress=False,
            )
            panels.append(
                _normalize_adapter_output(
                    panel,
                    data_domain,
                )
            )

            if show_progress:
                _render_loader_progress(
                    completed=index,
                    total=total_data_domains,
                    data_domain=data_domain,
                    started_at=started_at,
                    finished=index == total_data_domains,
                )

        return _merge_adapter_panels(panels)
    finally:
        if show_progress:
            print()
