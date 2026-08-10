# -*- coding: utf-8 -*-
"""BigQuant 多适配器、分粒度原始数据加载统筹器。

职责：
1. 动态读取因子的 FACTOR 元数据与本次因子参数；
2. 根据适配器字段目录判断每个标准字段应由谁加载；
3. 调用日频、财务、市场指数等 BigQuant 适配器；
4. 只在主键粒度一致的数据之间合并；
5. 返回 FactorDataBundle，保留各数据域原本的粒度。

loader 不计算因子、不生成调仓日期、不计算收益率，也不把市场数据广播到
每只股票。日期列表或连续时间窗口仍由策略/研究层负责生成。
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from factor_lib.common.data_adapters.bigquant_adapters.daily import (
    ADAPTER_SPEC as DAILY_ADAPTER_SPEC,
    load_daily_raw_data,
)
from factor_lib.common.data_adapters.bigquant_adapters.factor_data_bundle import (
    FactorDataBundle,
)
from factor_lib.common.data_adapters.bigquant_adapters.financial import (
    ADAPTER_SPEC as FINANCIAL_ADAPTER_SPEC,
    load_financial_raw_data,
)
from factor_lib.common.data_adapters.bigquant_adapters.market_daily import (
    ADAPTER_SPEC as MARKET_DAILY_ADAPTER_SPEC,
    load_market_daily_raw_data,
)
from factor_lib.factor_hub.discover_factors import discover_factors


ADAPTER_REGISTRY = {
    "daily": {
        "loader": load_daily_raw_data,
        "spec": DAILY_ADAPTER_SPEC,
    },
    "financial": {
        "loader": load_financial_raw_data,
        "spec": FINANCIAL_ADAPTER_SPEC,
    },
    "market_daily": {
        "loader": load_market_daily_raw_data,
        "spec": MARKET_DAILY_ADAPTER_SPEC,
    },
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

    unknown = sorted(set(factor_params) - set(parameters))
    if unknown:
        raise ValueError(
            f"因子 {metadata.get('name')!r} 收到未登记参数：{unknown}。"
            "请先在 FACTOR['parameters'] 中声明这些参数。"
        )

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
        raise ValueError(
            "input_schema.conditional.required_when 必须是字典。"
        )
    return all(
        resolved_params.get(name) == expected
        for name, expected in required_when.items()
    )


def _normalize_nonnegative_integer(value, field_name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} 必须是非负整数，不能是 bool。")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} 必须是非负整数。")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} 不能是负数。")
    return value


def _resolve_factor_data_window(metadata, resolved_params):
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
            raise ValueError("data_window.resolver 必须可调用。")
        default_window = specification.get("default", {})
        if not isinstance(default_window, Mapping):
            raise ValueError("动态 data_window.default 必须是字典。")
        dynamic_window = resolver(dict(resolved_params))
        if not isinstance(dynamic_window, Mapping):
            raise ValueError("data_window.resolver 必须返回字典。")
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
    requires_target = resolved_window.get(
        "requires_target_date_data", True
    )
    preheating = resolved_window.get(
        "preheating_required", lookback > 0
    )
    if not isinstance(requires_target, (bool, np.bool_)):
        raise ValueError("data_window.requires_target_date_data 必须是 bool。")
    if not isinstance(preheating, (bool, np.bool_)):
        raise ValueError("data_window.preheating_required 必须是 bool。")

    resolved_window["lookback_trading_days"] = lookback
    resolved_window["minimum_history_observations"] = minimum_history
    resolved_window["requires_target_date_data"] = bool(requires_target)
    resolved_window["preheating_required"] = bool(preheating)
    return resolved_window


def _resolve_factor_dependencies(metadata, resolved_params):
    """解析当前因子明确声明的直接依赖，不执行递归计算。"""
    specification = metadata.get("dependencies")
    if specification is None:
        return None
    if not isinstance(specification, Mapping):
        raise ValueError("FACTOR['dependencies'] 必须是字典。")

    resolver = specification.get("resolver")
    if not callable(resolver):
        raise ValueError("dependencies.resolver 必须是可调用函数。")
    resolved = resolver(dict(resolved_params))
    if not isinstance(resolved, Mapping):
        raise ValueError("dependencies.resolver 必须返回字典。")

    sequence_length = _normalize_nonnegative_integer(
        resolved.get("sequence_length"),
        "dependencies.sequence_length",
    )
    if sequence_length < 1:
        raise ValueError("dependencies.sequence_length 必须至少为 1。")

    items = resolved.get("items")
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("dependencies.items 必须是非空列表。")

    normalized_items = []
    feature_names = set()
    for position, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"第 {position} 个依赖定义必须是字典。"
            )
        factor_name = item.get("factor_name")
        feature_name = item.get("feature_name")
        factor_params = item.get("factor_params", {})
        if not isinstance(factor_name, str) or not factor_name.strip():
            raise ValueError(
                f"第 {position} 个依赖缺少有效 factor_name。"
            )
        if not isinstance(feature_name, str) or not feature_name.strip():
            raise ValueError(
                f"第 {position} 个依赖缺少有效 feature_name。"
            )
        factor_name = factor_name.strip()
        feature_name = feature_name.strip()
        if feature_name in feature_names:
            raise ValueError(f"feature_name 重复：{feature_name!r}。")
        if not isinstance(factor_params, Mapping):
            raise TypeError(
                f"依赖 {feature_name!r} 的 factor_params 必须是字典。"
            )

        child_metadata = get_factor_metadata(factor_name)
        if child_metadata.get("dependencies") is not None:
            raise NotImplementedError(
                f"当前只支持一层直接依赖；{factor_name!r} 本身仍有依赖。"
            )
        child_params = _resolved_factor_parameters(
            child_metadata,
            dict(factor_params),
        )
        child_window = _resolve_factor_data_window(
            child_metadata,
            child_params,
        )
        if not child_window["requires_target_date_data"]:
            raise NotImplementedError(
                f"当前复合因子依赖要求目标日数据；{factor_name!r} "
                "声明 requires_target_date_data=False。"
            )
        normalized_items.append(
            {
                "factor_name": factor_name,
                "feature_name": feature_name,
                "factor_params": dict(factor_params),
                "resolved_factor_params": child_params,
                "data_window": child_window,
            }
        )
        feature_names.add(feature_name)

    return {
        "sequence_length": sequence_length,
        "items": normalized_items,
    }


def _normalize_adapter_registry(adapter_overrides=None):
    registry = {
        name: {"loader": item["loader"], "spec": dict(item["spec"])}
        for name, item in ADAPTER_REGISTRY.items()
    }
    if adapter_overrides is None:
        return registry
    if not isinstance(adapter_overrides, Mapping):
        raise TypeError("adapter_overrides 必须是字典或 None。")

    for name, override in adapter_overrides.items():
        if name not in registry:
            if not isinstance(override, Mapping):
                raise ValueError(
                    f"新增适配器 {name!r} 时必须同时提供 loader 与 spec。"
                )
            loader = override.get("loader")
            spec = override.get("spec")
            if not callable(loader) or not isinstance(spec, Mapping):
                raise ValueError(
                    f"新增适配器 {name!r} 必须提供可调用 loader 和字典 spec。"
                )
            registry[name] = {"loader": loader, "spec": dict(spec)}
        elif callable(override):
            registry[name]["loader"] = override
        elif isinstance(override, Mapping):
            loader = override.get("loader", registry[name]["loader"])
            spec = override.get("spec", registry[name]["spec"])
            if not callable(loader) or not isinstance(spec, Mapping):
                raise ValueError(f"适配器覆盖 {name!r} 的格式无效。")
            registry[name] = {"loader": loader, "spec": dict(spec)}
        else:
            raise TypeError(
                f"适配器覆盖 {name!r} 必须是 callable 或字典。"
            )
    return registry


def _build_field_catalog(registry):
    catalog = {}
    for adapter_name, item in registry.items():
        spec = item["spec"]
        if spec.get("name", adapter_name) != adapter_name:
            raise ValueError(
                f"适配器注册名 {adapter_name!r} 与 spec.name 不一致。"
            )
        supported_fields = spec.get("supported_fields", ())
        for field in supported_fields:
            previous = catalog.get(field)
            if previous is not None and previous != adapter_name:
                raise ValueError(
                    f"标准字段 {field!r} 同时被 {previous!r} 和 "
                    f"{adapter_name!r} 声明，无法自动路由。"
                )
            catalog[field] = adapter_name
    return catalog


def _legacy_declared_adapter(field_name, specification):
    if not isinstance(specification, Mapping):
        return None
    data_domain = specification.get("data_domain")
    frequency = specification.get("frequency")
    if (
        data_domain is not None
        and frequency is not None
        and data_domain != frequency
    ):
        raise ValueError(
            f"字段 {field_name!r} 的 data_domain 与 frequency 冲突："
            f"{data_domain!r} != {frequency!r}。"
        )
    declared = data_domain if data_domain is not None else frequency
    if declared is None:
        return None
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(
            f"字段 {field_name!r} 的 data_domain/frequency 必须是非空字符串。"
        )
    return declared.strip()


def _route_field(field_name, specification, catalog, registry):
    if field_name in {"date", "instrument", "market_index"}:
        return None

    catalog_adapter = catalog.get(field_name)
    legacy_adapter = _legacy_declared_adapter(field_name, specification)

    if catalog_adapter is not None:
        if (
            legacy_adapter is not None
            and legacy_adapter in registry
            and legacy_adapter != catalog_adapter
        ):
            raise ValueError(
                f"字段 {field_name!r} 的旧 FACTOR 声明指向 "
                f"{legacy_adapter!r}，但 BigQuant 字段目录指向 "
                f"{catalog_adapter!r}。请检查映射。"
            )
        return catalog_adapter

    # 兼容本轮尚未迁移的旧 FACTOR：字段目录找不到时仍尝试其旧声明，
    # 最终由对应适配器给出“不支持字段”的明确错误。
    if legacy_adapter is not None and legacy_adapter in registry:
        return legacy_adapter

    raise KeyError(
        f"标准字段 {field_name!r} 尚未在任何 BigQuant 适配器中登记。"
        "请先完善数据源字段映射。"
    )


def get_factor_data_requirements(
    factor_name,
    factor_params=None,
    adapter_overrides=None,
):
    """解析本次参数组合需要的字段、适配器、数据窗口与上下文。"""
    metadata = get_factor_metadata(factor_name)
    schema = metadata.get("input_schema")
    if not isinstance(schema, Mapping):
        raise ValueError(
            f"因子 {factor_name!r} 缺少规范的 input_schema。"
        )

    registry = _normalize_adapter_registry(adapter_overrides)
    catalog = _build_field_catalog(registry)
    resolved_params = _resolved_factor_parameters(metadata, factor_params)
    resolved_dependencies = _resolve_factor_dependencies(
        metadata,
        resolved_params,
    )
    fields_by_adapter = defaultdict(list)

    required_schema = schema.get("required", {})
    conditional_schema = schema.get("conditional", {})
    if not isinstance(required_schema, Mapping):
        raise ValueError("input_schema.required 必须是字典。")
    if not isinstance(conditional_schema, Mapping):
        raise ValueError("input_schema.conditional 必须是字典。")

    field_specs = list(required_schema.items())
    for field_name, specification in conditional_schema.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"条件字段 {field_name!r} 的规范无效。")
        if _condition_is_met(
            specification.get("required_when"), resolved_params
        ):
            field_specs.append((field_name, specification))

    for field_name, specification in field_specs:
        adapter_name = _route_field(
            field_name, specification, catalog, registry
        )
        if (
            adapter_name is not None
            and field_name not in fields_by_adapter[adapter_name]
        ):
            fields_by_adapter[adapter_name].append(field_name)

    if not fields_by_adapter and resolved_dependencies is None:
        raise ValueError(
            f"因子 {factor_name!r} 没有可加载的非主键原始字段。"
        )

    resolved_window = _resolve_factor_data_window(
        metadata, resolved_params
    )
    normalized_fields = {
        name: list(fields) for name, fields in fields_by_adapter.items()
    }
    return {
        "factor_name": metadata.get("name", factor_name),
        "fields_by_adapter": normalized_fields,
        # 以下两个旧名称暂时保留，避免现有策略/研究脚本立即失效。
        "fields_by_domain": dict(normalized_fields),
        "fields_by_frequency": dict(normalized_fields),
        "data_window": resolved_window,
        "resolved_factor_params": resolved_params,
        "dependencies": resolved_dependencies,
    }


def _normalize_adapter_output(panel, adapter_name, spec):
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(
            f"适配器 {adapter_name!r} 必须返回 pandas.DataFrame。"
        )
    key_columns = tuple(spec.get("key_columns", ()))
    if not key_columns:
        raise ValueError(f"适配器 {adapter_name!r} 未声明 key_columns。")
    missing = sorted(set(key_columns) - set(panel.columns))
    if missing:
        raise ValueError(
            f"适配器 {adapter_name!r} 输出缺少主键：{missing}。"
        )

    result = panel.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(
            result["date"], errors="coerce"
        ).dt.normalize()
        if result["date"].isna().any():
            raise ValueError(
                f"适配器 {adapter_name!r} 输出包含无效 date。"
            )
    for key in key_columns:
        if result[key].isna().any():
            raise ValueError(
                f"适配器 {adapter_name!r} 的主键 {key!r} 包含空值。"
            )
    duplicated = result.duplicated(list(key_columns), keep=False)
    if duplicated.any():
        examples = (
            result.loc[duplicated, list(key_columns)]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            f"适配器 {adapter_name!r} 输出在主键 "
            f"{list(key_columns)} 上重复：{examples}"
        )
    return result


def _merge_same_granularity_panels(panels, key_columns, output_group):
    if not panels:
        raise ValueError(f"输出组 {output_group!r} 没有数据面板。")
    if len(panels) == 1:
        result = panels[0]
    else:
        result = panels[0]
        for panel in panels[1:]:
            overlapping = (
                set(result.columns)
                & set(panel.columns)
                - set(key_columns)
            )
            if overlapping:
                raise ValueError(
                    f"输出组 {output_group!r} 出现重复字段："
                    f"{sorted(overlapping)}。"
                )
            result = result.merge(
                panel,
                on=list(key_columns),
                how="outer",
                validate="one_to_one",
            )
    return result.sort_values(
        list(key_columns), kind="mergesort"
    ).reset_index(drop=True)


def _render_loader_progress(
    completed,
    total,
    adapter_name,
    started_at,
    stage=None,
    detail="",
):
    elapsed = time.perf_counter() - started_at
    percentage = 100.0 if total == 0 else completed / total * 100.0
    if 0 < completed < total:
        remaining = elapsed / completed * (total - completed)
        eta = f"，预计剩余 {remaining:.1f}s"
    else:
        eta = ""
    if stage is None:
        stage = "准备加载" if completed == 0 else f"已完成 {adapter_name}"
    message = (
        "\r[BigQuant loader] "
        f"{completed}/{total}（{percentage:6.2f}%），"
        f"{stage}"
    )
    if adapter_name:
        message += f"，当前 {adapter_name}"
    if detail:
        message += f"，{detail}"
    message += f"，耗时 {elapsed:.1f}s{eta}"
    print(message.ljust(180), end="", flush=True)


def load_trading_dates(start_date, end_date):
    """读取闭区间内的 A 股交易日；仅返回标准化日期索引。"""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("start_date 和 end_date 必须是有效日期。")
    if start > end:
        raise ValueError("start_date 不能晚于 end_date。")

    import dai

    sql = f"""
    SELECT DISTINCT date
    FROM cn_stock_bar1d
    WHERE date BETWEEN '{start:%Y-%m-%d}' AND '{end:%Y-%m-%d}'
    ORDER BY date
    """
    result = dai.query(
        sql,
        filters={
            "date": [
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            ]
        },
    ).df()
    if result.empty:
        raise ValueError(
            f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d} 未查询到交易日。"
        )
    dates = pd.DatetimeIndex(
        pd.to_datetime(result["date"], errors="raise")
    ).normalize().unique().sort_values()
    return dates


def _normalize_final_target_dates(start_date, end_date, dates):
    uses_range = start_date is not None or end_date is not None
    uses_dates = dates is not None
    if uses_range == uses_dates:
        raise ValueError(
            "日期选择必须二选一：start_date + end_date，或 dates。"
        )
    if uses_range:
        if start_date is None or end_date is None:
            raise ValueError("连续区间必须同时提供 start_date 和 end_date。")
        return load_trading_dates(start_date, end_date)

    if isinstance(dates, (str, pd.Timestamp, np.datetime64)):
        dates = [dates]
    normalized = pd.DatetimeIndex(
        pd.to_datetime(list(dates), errors="raise")
    ).normalize().unique().sort_values()
    if len(normalized) == 0:
        raise ValueError("dates 不能为空。")
    return normalized


def _load_calendar_covering_lookback(final_dates, required_history_days):
    """取得覆盖最早最终目标日前足够交易日的轻量交易日历。"""
    required_history_days = _normalize_nonnegative_integer(
        required_history_days,
        "required_history_days",
    )
    end = final_dates.max()
    span_days = max(60, required_history_days * 2 + 30)

    for _ in range(6):
        start = final_dates.min() - pd.Timedelta(days=span_days)
        calendar = load_trading_dates(start, end)
        positions = {date: index for index, date in enumerate(calendar)}
        missing = final_dates.difference(calendar)
        if len(missing) > 0:
            raise ValueError(
                "最终目标日期包含非交易日："
                f"{[date.strftime('%Y-%m-%d') for date in missing]}。"
            )
        earliest_position = min(positions[date] for date in final_dates)
        if earliest_position >= required_history_days:
            return calendar, positions
        span_days *= 2

    raise ValueError(
        f"无法为最早目标日准备 {required_history_days} 个历史交易日。"
    )


def _load_dependent_factor_raw_data(
    factor_name,
    requirements,
    start_date,
    end_date,
    dates,
    instruments,
    adapter_overrides,
    show_progress,
):
    """按每个直接依赖自己的窗口加载原始数据，不做最大窗口广播。"""
    dependency_spec = requirements["dependencies"]
    final_dates = _normalize_final_target_dates(
        start_date,
        end_date,
        dates,
    )
    sequence_length = dependency_spec["sequence_length"]
    dependency_items = dependency_spec["items"]
    maximum_lookback = max(
        item["data_window"]["lookback_trading_days"]
        for item in dependency_items
    )
    required_history = sequence_length - 1 + maximum_lookback
    calendar, calendar_positions = _load_calendar_covering_lookback(
        final_dates,
        required_history,
    )

    dependencies = {}
    dependency_target_dates = {}
    dependency_raw_dates = {}
    shell_parts = []
    started_at = time.perf_counter()
    total = len(dependency_items)

    try:
        for index, item in enumerate(dependency_items, start=1):
            feature_name = item["feature_name"]
            child_factor = item["factor_name"]
            lookback = item["data_window"]["lookback_trading_days"]
            per_target_feature_dates = {}
            per_target_raw_dates = {}

            for final_date in final_dates:
                final_position = calendar_positions[final_date]
                feature_start = final_position - sequence_length + 1
                raw_start = feature_start - lookback
                if raw_start < 0:
                    raise ValueError(
                        f"依赖 {feature_name!r} 在 {final_date:%Y-%m-%d} "
                        "缺少足够历史交易日。"
                    )
                per_target_feature_dates[final_date] = calendar[
                    feature_start: final_position + 1
                ]
                per_target_raw_dates[final_date] = calendar[
                    raw_start: final_position + 1
                ]

            raw_dates = pd.DatetimeIndex(
                [
                    date
                    for values in per_target_raw_dates.values()
                    for date in values
                ]
            ).unique().sort_values()

            if show_progress:
                _render_loader_progress(
                    index - 1,
                    total,
                    feature_name,
                    started_at,
                    stage=f"正在加载依赖因子 {child_factor}",
                    detail=(
                        f"{len(raw_dates)} 个日期，独立预热 {lookback} 日"
                    ),
                )

            child_bundle = load_factor_raw_data(
                factor_name=child_factor,
                dates=raw_dates,
                factor_params=item["resolved_factor_params"],
                instruments=instruments,
                adapter_overrides=adapter_overrides,
                show_progress=False,
            )
            if child_bundle.dependency_names:
                raise NotImplementedError(
                    f"当前只支持一层依赖，{child_factor!r} 返回了嵌套依赖。"
                )
            if not child_bundle.has_domain("security_daily"):
                raise ValueError(
                    f"依赖因子 {child_factor!r} 没有 security_daily 数据域。"
                )

            dependencies[feature_name] = child_bundle
            dependency_target_dates[feature_name] = (
                per_target_feature_dates
            )
            dependency_raw_dates[feature_name] = per_target_raw_dates
            security_keys = child_bundle.get_security_daily().loc[
                lambda frame: frame["date"].isin(final_dates),
                ["date", "instrument"],
            ]
            shell_parts.append(security_keys)

            if show_progress:
                _render_loader_progress(
                    index,
                    total,
                    feature_name,
                    started_at,
                    stage="依赖数据加载完成",
                    detail=f"{sum(child_bundle.row_counts().values()):,} 行",
                )

        shell = (
            pd.concat(shell_parts, ignore_index=True)
            .drop_duplicates(["date", "instrument"])
            .sort_values(["date", "instrument"], kind="mergesort")
            .reset_index(drop=True)
        )
        if shell.empty:
            raise ValueError(
                f"因子 {factor_name!r} 的依赖数据未覆盖任何最终目标截面。"
            )
        return FactorDataBundle(
            {"security_daily": shell},
            key_columns={"security_daily": ("date", "instrument")},
            dependencies=dependencies,
            dependency_target_dates=dependency_target_dates,
            dependency_raw_dates=dependency_raw_dates,
        )
    finally:
        if show_progress:
            print()


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
    """根据 FACTOR 语义字段自动加载原始数据并返回分域容器。

    日期选择必须二选一：连续闭区间 ``start_date`` + ``end_date``，或
    离散日期列表 ``dates``。市场指数等适配器上下文从同一份已解析的
    ``factor_params`` 中取得，例如 ``market_index='csi_all_share'``。
    """
    registry = _normalize_adapter_registry(adapter_overrides)
    requirements = get_factor_data_requirements(
        factor_name,
        factor_params,
        adapter_overrides=adapter_overrides,
    )
    if requirements["dependencies"] is not None:
        return _load_dependent_factor_raw_data(
            factor_name=factor_name,
            requirements=requirements,
            start_date=start_date,
            end_date=end_date,
            dates=dates,
            instruments=instruments,
            adapter_overrides=adapter_overrides,
            show_progress=show_progress,
        )

    resolved_params = requirements["resolved_factor_params"]
    items = list(requirements["fields_by_adapter"].items())
    started_at = time.perf_counter()
    grouped_panels = defaultdict(list)
    group_keys = {}

    if show_progress:
        _render_loader_progress(0, len(items), "", started_at)

    try:
        for index, (adapter_name, fields) in enumerate(items, start=1):
            item = registry.get(adapter_name)
            if item is None:
                raise NotImplementedError(
                    f"未注册 BigQuant 适配器 {adapter_name!r}。"
                )
            adapter = item["loader"]
            spec = item["spec"]

            if show_progress:
                _render_loader_progress(
                    index - 1,
                    len(items),
                    adapter_name,
                    started_at,
                    stage="正在调用数据适配器",
                    detail=f"{len(fields)} 个标准字段",
                )

            call_kwargs = {
                "standard_fields": fields,
                "start_date": start_date,
                "end_date": end_date,
                "dates": dates,
                "instruments": instruments,
                # loader 负责把外层进度开关传给当前实际工作的适配器；
                # 各适配器只在自己的查询阶段占用单行输出。
                "show_progress": show_progress,
            }
            for parameter_name in spec.get("context_parameters", ()):
                value = resolved_params.get(parameter_name)
                if value is None:
                    raise ValueError(
                        f"适配器 {adapter_name!r} 需要因子参数 "
                        f"{parameter_name!r}，但本次未提供且无默认值。"
                    )
                call_kwargs[parameter_name] = value

            panel = _normalize_adapter_output(
                adapter(**call_kwargs), adapter_name, spec
            )
            output_group = spec.get("output_group")
            key_columns = tuple(spec.get("key_columns", ()))
            if not isinstance(output_group, str) or not output_group:
                raise ValueError(
                    f"适配器 {adapter_name!r} 未声明 output_group。"
                )
            previous_keys = group_keys.get(output_group)
            if previous_keys is not None and previous_keys != key_columns:
                raise ValueError(
                    f"输出组 {output_group!r} 内存在不同主键："
                    f"{previous_keys} 与 {key_columns}。"
                )
            group_keys[output_group] = key_columns
            grouped_panels[output_group].append(panel)

            if show_progress:
                _render_loader_progress(
                    index,
                    len(items),
                    adapter_name,
                    started_at,
                    stage="适配器数据加载完成",
                    detail=f"{len(panel):,} 行 -> {output_group}",
                )

        group_items = list(grouped_panels.items())
        domains = {}
        for index, (output_group, panels) in enumerate(
            group_items,
            start=1,
        ):
            if show_progress:
                _render_loader_progress(
                    index - 1,
                    len(group_items),
                    output_group,
                    started_at,
                    stage="正在合并同粒度数据域",
                    detail=f"{len(panels)} 个面板",
                )
            domains[output_group] = _merge_same_granularity_panels(
                panels,
                group_keys[output_group],
                output_group,
            )
            if show_progress:
                _render_loader_progress(
                    index,
                    len(group_items),
                    output_group,
                    started_at,
                    stage="数据域合并完成",
                    detail=f"{len(domains[output_group]):,} 行",
                )
        return FactorDataBundle(domains, key_columns=group_keys)
    finally:
        if show_progress:
            print()
