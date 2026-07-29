# -*- coding: utf-8 -*-
"""BigQuant 数据加载统筹器。

loader 只负责读取 FACTOR 元数据、按字段频率调度适配器、合并适配器返回的
标准字段面板。它不计算因子、不决定目标日期、不选股，也不执行策略。
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from factor_lib.common.data_adapters.bigquant_adapters.daily import (
    load_daily_raw_data,
)
from factor_lib.factor_hub.discover_factors import discover_factors


# 后续新增 financial.py、minute.py 等适配器时，只需在这里登记频率名称和函数。
ADAPTER_REGISTRY = {
    "daily": load_daily_raw_data,
}


def get_factor_metadata(factor_name):
    """通过因子中心动态发现并返回一个因子的 FACTOR 元数据。"""
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
            if isinstance(metadata, Mapping) and metadata.get("name") == name:
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
    raise KeyError(f"未发现因子 {name!r}。可用因子：{available}")


def _resolved_factor_parameters(metadata, factor_params):
    factor_params = {} if factor_params is None else dict(factor_params)
    parameters = metadata.get("parameters", {})
    resolved = {}

    for name, specification in parameters.items():
        if isinstance(specification, Mapping) and "default" in specification:
            resolved[name] = specification["default"]
    resolved.update(factor_params)
    return resolved


def _condition_is_met(required_when, resolved_params):
    if not required_when:
        return True
    if not isinstance(required_when, Mapping):
        raise ValueError("input_schema.conditional.required_when 必须是字典。")
    return all(
        resolved_params.get(parameter_name) == expected_value
        for parameter_name, expected_value in required_when.items()
    )


def _add_field_requirement(result, field_name, specification):
    if field_name in {"date", "instrument"}:
        return
    if not isinstance(specification, Mapping):
        specification = {}
    frequency = specification.get("frequency", "daily")
    result.setdefault(frequency, [])
    if field_name not in result[frequency]:
        result[frequency].append(field_name)


def get_factor_data_requirements(factor_name, factor_params=None):
    """解析 FACTOR 元数据，返回当前参数组合所需的字段和数据窗口。

    因子字段未标注 ``frequency`` 时暂按 ``daily`` 处理；新增非日频因子时，
    应在对应字段规范中显式填写 ``frequency``，例如 ``financial`` 或 ``minute``。
    """
    metadata = get_factor_metadata(factor_name)
    schema = metadata.get("input_schema")
    if not isinstance(schema, Mapping):
        raise ValueError(
            f"因子 {factor_name!r} 缺少规范的 input_schema，"
            "无法由 loader 自动调度数据适配器。"
        )

    resolved_params = _resolved_factor_parameters(metadata, factor_params)
    fields_by_frequency = {}

    for field_name, specification in schema.get("required", {}).items():
        _add_field_requirement(fields_by_frequency, field_name, specification)

    for field_name, specification in schema.get("conditional", {}).items():
        if not isinstance(specification, Mapping):
            raise ValueError(
                f"因子 {factor_name!r} 的条件字段 {field_name!r} 规范无效。"
            )
        if _condition_is_met(specification.get("required_when"), resolved_params):
            _add_field_requirement(fields_by_frequency, field_name, specification)

    return {
        "factor_name": metadata.get("name", factor_name),
        "fields_by_frequency": fields_by_frequency,
        "data_window": metadata.get("data_window", {}),
        "resolved_factor_params": resolved_params,
    }


def _merge_adapter_panels(panels):
    """按 date + instrument 合并不同频率适配器已标准化的输出。"""
    if not panels:
        return pd.DataFrame(columns=["date", "instrument"])
    if len(panels) == 1:
        return panels[0]

    merged = panels[0]
    for panel in panels[1:]:
        required_keys = {"date", "instrument"}
        missing_left = required_keys - set(merged.columns)
        missing_right = required_keys - set(panel.columns)
        if missing_left or missing_right:
            raise ValueError(
                "不同频率适配器的输出必须包含 date、instrument，"
                f"当前缺失：left={sorted(missing_left)}，right={sorted(missing_right)}。"
            )
        try:
            merged = merged.merge(
                panel,
                on=["date", "instrument"],
                how="outer",
                validate="one_to_one",
            )
        except pd.errors.MergeError as exc:
            raise ValueError(
                "适配器输出在 date + instrument 上不是一对一关系；"
                "请在对应频率适配器中明确其时间对齐和点时口径。"
            ) from exc
    return merged


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
    """根据 FACTOR 元数据统筹多个数据适配器，拉取一个原始数据时间区间。

    日期选择方式必须二选一：

    - ``start_date`` + ``end_date``：连续原始数据覆盖区间；
    - ``dates``：离散原始数据日期节点。

    固定 N 日频策略应在内部先根据交易日历、信号日和预热窗口算出
    ``dates``，再传给本函数。loader 不接收 target_dates，不计算因子值，
    也不产生信号。
    """
    requirements = get_factor_data_requirements(factor_name, factor_params)
    adapters = dict(ADAPTER_REGISTRY)
    if adapter_overrides is not None:
        adapters.update(adapter_overrides)

    panels = []
    total_frequencies = len(requirements["fields_by_frequency"])
    for index, (frequency, fields) in enumerate(
        requirements["fields_by_frequency"].items(),
        start=1,
    ):
        adapter = adapters.get(frequency)
        if adapter is None:
            raise NotImplementedError(
                f"因子 {factor_name!r} 需要 {frequency!r} 数据适配器，"
                "但 BigQuant adapter registry 尚未登记该适配器。"
            )
        if show_progress:
            print(
                f"[BigQuant loader] {index}/{total_frequencies}："
                f"拉取 {frequency} 原始字段 {fields}...",
                flush=True,
            )
        panels.append(
            adapter(
                standard_fields=fields,
                start_date=start_date,
                end_date=end_date,
                dates=dates,
                instruments=instruments,
                show_progress=show_progress,
            )
        )

    raw_data = _merge_adapter_panels(panels)
    if show_progress:
        print(
            f"[BigQuant loader] 原始数据准备完成：{len(raw_data):,} 行，"
            f"频率：{list(requirements['fields_by_frequency'])}。",
            flush=True,
        )
    return raw_data
