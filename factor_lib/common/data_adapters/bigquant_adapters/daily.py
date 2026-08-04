# -*- coding: utf-8 -*-
"""BigQuant 日频原始数据适配器。

本模块只做三件事：
1. 将因子库的标准字段名映射为 BigQuant 的日频数据表字段；
2. 按调用方给定的连续日期区间或离散日期列表拉取原始数据；
3. 为 BigQuant 分区表查询显式传入日期 filters。

它不计算因子、不构造标签、不填充缺失值、不复权、不去重，
也不判断信号日或调仓日。
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Callable, Optional

import pandas as pd


# 每个标准字段只描述数据源中的位置，不包含任何因子口径或加工逻辑。
# table: 内部表标识；column: BigQuant 原始字段名。
DAILY_FIELD_MAPPING = {
    # cn_stock_bar1d：日行情
    "open": {"table": "bar1d", "column": "open"},
    "high": {"table": "bar1d", "column": "high"},
    "low": {"table": "bar1d", "column": "low"},
    "close": {"table": "bar1d", "column": "close"},
    "pre_close": {"table": "bar1d", "column": "pre_close"},
    "volume": {"table": "bar1d", "column": "volume"},
    "amount": {"table": "bar1d", "column": "amount"},
    "turn": {"table": "bar1d", "column": "turn"},
    "adjust_factor": {"table": "bar1d", "column": "adjust_factor"},
    "upper_limit": {"table": "bar1d", "column": "upper_limit"},
    "lower_limit": {"table": "bar1d", "column": "lower_limit"},
    "change_ratio": {"table": "bar1d", "column": "change_ratio"},
    "deal_number": {"table": "bar1d", "column": "deal_number"},
    "name": {"table": "bar1d", "column": "name"},
    # cn_stock_valuation：日频估值字段
    "total_market_cap": {
        "table": "valuation",
        "column": "total_market_cap",
    },
    "float_market_cap": {
        "table": "valuation",
        "column": "float_market_cap",
    },
    "pb": {"table": "valuation", "column": "pb"},
    "pe_ttm": {"table": "valuation", "column": "pe_ttm"},
    "pe_trailing": {
        "table": "valuation",
        "column": "pe_trailing",
    },
    "pe_leading": {
        "table": "valuation",
        "column": "pe_leading",
    },
    "ps_ttm": {"table": "valuation", "column": "ps_ttm"},
    "ps_trailing": {
        "table": "valuation",
        "column": "ps_trailing",
    },
    "ps_leading": {
        "table": "valuation",
        "column": "ps_leading",
    },
    "pcf_op_ttm": {
        "table": "valuation",
        "column": "pcf_op_ttm",
    },
    "pcf_op_leading": {
        "table": "valuation",
        "column": "pcf_op_leading",
    },
    "pcf_net_ttm": {
        "table": "valuation",
        "column": "pcf_net_ttm",
    },
    "pcf_net_leading": {
        "table": "valuation",
        "column": "pcf_net_leading",
    },
    "dividend_yield_ratio": {
        "table": "valuation",
        "column": "dividend_yield_ratio",
    },
    # cn_stock_moneyflow：日频个股资金流原始字段
    "main_inflow_amount": {
        "table": "moneyflow",
        "column": "inflow_amount_main",
    },
    "main_outflow_amount": {
        "table": "moneyflow",
        "column": "outflow_amount_main",
    },
    # cn_stock_prefactors：日频证券属性 / 行业 /状态字段
    "industry": {"table": "prefactors", "column": "cs_level1"},
    "industry_level1": {
        "table": "prefactors",
        "column": "cs_level1",
    },
    "industry_level2": {
        "table": "prefactors",
        "column": "cs_level2",
    },
    "industry_level3": {
        "table": "prefactors",
        "column": "cs_level3",
    },
    "industry_name_level1": {
        "table": "prefactors",
        "column": "cs_level1_name",
    },
    "industry_name_level2": {
        "table": "prefactors",
        "column": "cs_level2_name",
    },
    "industry_name_level3": {
        "table": "prefactors",
        "column": "cs_level3_name",
    },
    "is_risk_warning": {
        "table": "prefactors",
        "column": "is_risk_warning",
    },
    "suspended": {
        "table": "prefactors",
        "column": "suspended",
    },
    "st_status": {"table": "prefactors", "column": "st_status"},
    "list_date": {"table": "prefactors", "column": "list_date"},
    "list_days": {"table": "prefactors", "column": "list_days"},
}


# 供 loader 自动发现字段归属与输出粒度。映射关系只存在于数据源适配层，
# 因子 FACTOR 无需记录 BigQuant 表名或数据域名称。
ADAPTER_SPEC = {
    "name": "daily",
    "output_group": "security_daily",
    "key_columns": ("date", "instrument"),
    "supported_fields": tuple(DAILY_FIELD_MAPPING),
    "context_parameters": (),
}


TABLE_SPECS = {
    "bar1d": {
        "name": "cn_stock_bar1d",
        "alias": "b",
        "base_priority": 30,
    },
    "valuation": {
        "name": "cn_stock_valuation",
        "alias": "v",
        "base_priority": 20,
    },
    "prefactors": {
        "name": "cn_stock_prefactors",
        "alias": "p",
        "base_priority": 10,
    },
    "moneyflow": {
        "name": "cn_stock_moneyflow",
        "alias": "mf",
        "base_priority": 5,
    },
}


def list_supported_daily_fields():
    """返回当前 BigQuant 日频适配器支持的标准字段名。"""
    return sorted(DAILY_FIELD_MAPPING)


def _normalize_fields(standard_fields):
    if isinstance(standard_fields, str):
        standard_fields = [standard_fields]
    elif not isinstance(standard_fields, Iterable):
        raise TypeError(
            "standard_fields 必须是字段名字符串或字段名序列。"
        )

    fields = []
    for field in standard_fields:
        if not isinstance(field, str) or not field.strip():
            raise ValueError(
                "standard_fields 中存在空字段名或非字符串字段名。"
            )
        field = field.strip()
        if field not in {"date", "instrument"} and field not in fields:
            fields.append(field)

    if not fields:
        raise ValueError(
            "至少需要指定一个非 date/instrument 的日频标准字段。"
        )

    unsupported = sorted(
        set(fields) - set(DAILY_FIELD_MAPPING)
    )
    if unsupported:
        raise ValueError(
            "BigQuant 日频适配器暂不支持字段："
            f"{unsupported}。可用字段："
            f"{list_supported_daily_fields()}"
        )
    return fields


def _normalize_date(value, parameter_name):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{parameter_name} 必须是可解析日期：{value!r}"
        ) from exc

    if pd.isna(timestamp):
        raise ValueError(f"{parameter_name} 不允许为空。")
    return timestamp.strftime("%Y-%m-%d")


def _normalize_instruments(instruments):
    if instruments is None:
        return None

    if isinstance(instruments, str):
        instruments = [instruments]
    elif not isinstance(instruments, Iterable):
        raise TypeError(
            "instruments 必须是证券代码字符串、代码序列或 None。"
        )

    normalized = []
    for instrument in instruments:
        if not isinstance(instrument, str) or not instrument.strip():
            raise ValueError(
                "instruments 中存在空代码或非字符串代码。"
            )
        instrument = instrument.strip()
        if instrument not in normalized:
            normalized.append(instrument)

    if not normalized:
        raise ValueError(
            "instruments 为空；请传入至少一个证券代码或使用 None。"
        )
    return normalized


def _quote_sql_literal(value):
    return "'" + value.replace("'", "''") + "'"


def _normalize_dates(dates):
    if isinstance(dates, (str, pd.Timestamp)):
        dates = [dates]
    elif not isinstance(dates, Iterable):
        raise TypeError(
            "dates 必须是日期字符串、日期序列或 None。"
        )

    normalized = []
    for position, value in enumerate(dates, start=1):
        normalized.append(
            _normalize_date(value, f"dates[{position}]")
        )

    if not normalized:
        raise ValueError("dates 不能为空。")
    return sorted(set(normalized))


def _resolve_date_selector(start_date, end_date, dates):
    """规范日期选择条件；区间与离散日期列表不能同时使用。"""
    has_dates = dates is not None
    has_start = start_date is not None
    has_end = end_date is not None

    if has_dates and (has_start or has_end):
        raise ValueError(
            "日期选择方式冲突：请传入 dates，或同时传入 "
            "start_date 与 end_date，不能混用。"
        )

    if has_dates:
        return {
            "mode": "dates",
            "dates": _normalize_dates(dates),
        }

    if has_start != has_end:
        raise ValueError(
            "连续区间必须同时提供 start_date 和 end_date。"
        )
    if not has_start:
        raise ValueError(
            "请传入 dates，或同时传入 start_date 和 end_date。"
        )

    normalized_start = _normalize_date(
        start_date,
        "start_date",
    )
    normalized_end = _normalize_date(
        end_date,
        "end_date",
    )
    if normalized_start > normalized_end:
        raise ValueError("start_date 不能晚于 end_date。")

    return {
        "mode": "range",
        "start_date": normalized_start,
        "end_date": normalized_end,
    }


def _build_partition_filters(date_selector):
    """生成 BigQuant dai.query 要求的日期分区范围。

    对离散日期请求，filters 使用最早和最晚日期限制底层分区扫描；
    SQL 中的 ``date IN (...)`` 继续负责只返回调用方真正请求的日期。
    """
    if date_selector["mode"] == "range":
        start_date = date_selector["start_date"]
        end_date = date_selector["end_date"]
    else:
        start_date = date_selector["dates"][0]
        end_date = date_selector["dates"][-1]

    return {"date": [start_date, end_date]}


def _build_daily_sql(fields, date_selector, instruments):
    required_tables = {
        DAILY_FIELD_MAPPING[field]["table"]
        for field in fields
    }
    base_table = max(
        required_tables,
        key=lambda table: TABLE_SPECS[table]["base_priority"],
    )
    base_spec = TABLE_SPECS[base_table]
    base_alias = base_spec["alias"]

    select_parts = [
        f"{base_alias}.date AS date",
        f"{base_alias}.instrument AS instrument",
    ]
    for field in fields:
        mapping = DAILY_FIELD_MAPPING[field]
        table_alias = TABLE_SPECS[mapping["table"]]["alias"]
        select_parts.append(
            f"{table_alias}.{mapping['column']} AS {field}"
        )

    join_parts = []
    join_tables = sorted(
        required_tables - {base_table},
        key=lambda table: TABLE_SPECS[table]["base_priority"],
        reverse=True,
    )
    for table in join_tables:
        spec = TABLE_SPECS[table]
        join_parts.append(
            f"LEFT JOIN {spec['name']} AS {spec['alias']}\n"
            f"    ON {base_alias}.date = {spec['alias']}.date\n"
            f"    AND {base_alias}.instrument = "
            f"{spec['alias']}.instrument"
        )

    if date_selector["mode"] == "range":
        where_parts = [
            f"{base_alias}.date BETWEEN "
            f"{_quote_sql_literal(date_selector['start_date'])} "
            f"AND {_quote_sql_literal(date_selector['end_date'])}"
        ]
    else:
        dates_sql = ", ".join(
            _quote_sql_literal(date)
            for date in date_selector["dates"]
        )
        where_parts = [
            f"{base_alias}.date IN ({dates_sql})"
        ]

    if instruments is not None:
        instrument_sql = ", ".join(
            _quote_sql_literal(item)
            for item in instruments
        )
        where_parts.append(
            f"{base_alias}.instrument IN ({instrument_sql})"
        )

    return "\n".join(
        [
            "SELECT",
            "    " + ",\n    ".join(select_parts),
            f"FROM {base_spec['name']} AS {base_alias}",
            *join_parts,
            "WHERE " + "\n  AND ".join(where_parts),
            "ORDER BY date, instrument",
        ]
    )


def _default_query(sql, filters):
    """执行 BigQuant 原生查询，并显式声明分区范围。"""
    try:
        import dai
    except ImportError as exc:
        raise ImportError(
            "未能导入 dai。请在 BigQuant 环境中运行，"
            "或通过 query_func 传入兼容的查询函数。"
        ) from exc

    return dai.query(sql, filters=filters)


def _render_progress(stage, started_at, completed=None, total=None, detail=""):
    elapsed = time.perf_counter() - started_at
    parts = [f"[BigQuant 日频适配器] {stage}"]
    if completed is not None and total:
        percentage = completed / total
        parts.append(f"{completed}/{total} ({percentage:.1%})")
        if 0 < completed < total:
            remaining = elapsed / completed * (total - completed)
            parts.append(f"预计剩余 {remaining:.1f}s")
    if detail:
        parts.append(str(detail))
    parts.append(f"已耗时 {elapsed:.1f}s")
    print("\r" + " | ".join(parts).ljust(180), end="", flush=True)


def load_daily_raw_data(
    standard_fields,
    start_date=None,
    end_date=None,
    dates=None,
    instruments=None,
    query_func: Optional[Callable] = None,
    show_progress=False,
):
    """按日期区间或离散日期拉取 BigQuant 日频原始数据。

    参数
    ----
    standard_fields : str 或 sequence[str]
        所需的因子库标准字段名。不必包含 date、instrument；
        返回值固定包含它们。
    start_date, end_date : str 或 datetime，可选
        连续原始数据覆盖区间（闭区间）。必须与 dates 二选一。
    dates : 日期或 sequence[日期]，可选
        需要拉取的离散日期节点。会在内存中去重和排序；
        必须与 start_date、end_date 二选一。固定 N 日频策略的
        预存模块应传入此参数。
    instruments : sequence[str] 或 None，默认 None
        可选的静态证券范围；None 表示不在 SQL 中限制证券代码。
    query_func : callable，可选
        用于本地测试或替换数据客户端。接收 SQL 字符串，返回
        dai.query 的结果对象或 pandas.DataFrame。传入自定义函数时，
        由该函数自行处理其数据客户端所需的分区参数。
    show_progress : bool，默认 False
        仅显示一次开始和一次完成信息，不改变任何返回数据。

    返回
    ----
    pandas.DataFrame
        原始字段面板，固定含 date、instrument 及请求字段。
        不会填充、去重、类型转换或进行任何因子相关加工。
    """
    fields = _normalize_fields(standard_fields)
    date_selector = _resolve_date_selector(
        start_date,
        end_date,
        dates,
    )
    instruments = _normalize_instruments(instruments)

    sql = _build_daily_sql(
        fields,
        date_selector,
        instruments,
    )
    partition_filters = _build_partition_filters(date_selector)
    started_at = time.perf_counter()

    if date_selector["mode"] == "range":
        date_summary = (
            f"{date_selector['start_date']} 至 "
            f"{date_selector['end_date']}"
        )
    else:
        date_summary = (
            f"{len(date_selector['dates'])} 个离散日期"
            f"（{date_selector['dates'][0]} 至 "
            f"{date_selector['dates'][-1]}）"
        )
    if show_progress:
        _render_progress(
            "[1/3] 提交日频查询",
            started_at,
            detail=f"{date_summary}，{len(fields)} 个字段",
        )

    if query_func is None:
        query_result = _default_query(
            sql,
            filters=partition_filters,
        )
    else:
        # 自定义查询函数主要用于本地测试，保持原有单参数接口兼容。
        query_result = query_func(sql)

    if show_progress:
        _render_progress(
            "[2/3] 将查询结果转换为 DataFrame",
            started_at,
            detail=date_summary,
        )
    result = (
        query_result.df()
        if hasattr(query_result, "df")
        else query_result
    )
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            "query_func 必须返回 pandas.DataFrame "
            "或具有 .df() 的查询结果。"
        )

    expected_columns = ["date", "instrument", *fields]
    missing_columns = sorted(
        set(expected_columns) - set(result.columns)
    )
    if missing_columns:
        raise ValueError(
            "BigQuant 查询结果缺少预期字段："
            f"{missing_columns}。请检查字段映射和数据表权限。"
        )

    # 按请求顺序返回；这只是字段投影，不改变原始字段值或行记录。
    result = result.loc[:, expected_columns]

    if show_progress:
        _render_progress(
            "[3/3] 查询结果校验完成",
            started_at,
            completed=1,
            total=1,
            detail=f"{len(result):,} 行",
        )
        print()

    return result
