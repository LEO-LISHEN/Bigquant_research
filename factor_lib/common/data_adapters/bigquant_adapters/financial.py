# -*- coding: utf-8 -*-
"""BigQuant 日频点时财务数据适配器。

本模块只做两件事：
1. 将因子库使用的语义标准字段映射为 BigQuant 字段；
2. 按连续日期区间或离散日期列表读取日频点时财务面板。

不在此处计算因子、前向填充原始财报、清洗数据或构造标签。因子
模块只依赖左侧的标准字段名；BigQuant 表名与字段名仅存在于本文件。
"""

from __future__ import annotations

import inspect
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Callable, Optional

import pandas as pd


# 这两张表都是 BigQuant 已按交易日组织的点时财务面板，能够用
# date + instrument 唯一定位。非日频原始报表不可登记到本适配器。
FINANCIAL_TABLE_CONFIG = {
    "cn_stock_factors_financial_indicators": {
        "date_column": "date",
        "instrument_column": "instrument",
        "description": "BigQuant 日频点时财务指标表",
    },
    "cn_stock_factors_financial_items": {
        "date_column": "date",
        "instrument_column": "instrument",
        "description": "BigQuant 日频点时财务科目表",
    },
}

_INDICATORS = "cn_stock_factors_financial_indicators"
_ITEMS = "cn_stock_factors_financial_items"


def _field(table, column, description):
    """构造一条字段映射，避免目录中出现不一致的键结构。"""
    return {
        "table": table,
        "column": column,
        "description": description,
    }


# 左侧为数据源无关的语义字段；右侧为 BigQuant 的表和字段映射。
# 新字段必须先确认：字段存在、MRQ/TTM/LF 口径正确、来源是日频点时表。
FINANCIAL_FIELD_MAPPING = {
    # ===== 成长：利润、收入、现金流 =====
    "quarterly_net_profit_yoy": _field(
        _INDICATORS, "net_profit_yoy_mrq", "净利润同比增长率（单季度）"
    ),
    "quarterly_net_profit_qoq": _field(
        _INDICATORS, "net_profit_qoq_mrq", "净利润环比增长率（单季度）"
    ),
    "ttm_net_profit_yoy": _field(
        _INDICATORS, "net_profit_yoy_ttm", "净利润同比增长率（TTM）"
    ),
    "latest_net_profit_yoy": _field(
        _INDICATORS, "net_profit_yoy_lf", "净利润同比增长率（最新一期）"
    ),
    "quarterly_parent_net_profit_yoy": _field(
        _INDICATORS,
        "net_profit_to_parent_yoy_mrq",
        "归母净利润同比增长率（单季度）",
    ),
    "ttm_parent_net_profit_yoy": _field(
        _INDICATORS,
        "net_profit_to_parent_yoy_ttm",
        "归母净利润同比增长率（TTM）",
    ),
    "latest_parent_net_profit_yoy": _field(
        _INDICATORS,
        "net_profit_to_parent_yoy_lf",
        "归母净利润同比增长率（最新一期）",
    ),
    "quarterly_operating_revenue_yoy": _field(
        _INDICATORS,
        "operating_revenue_yoy_mrq",
        "营业收入同比增长率（单季度）",
    ),
    "quarterly_operating_revenue_qoq": _field(
        _INDICATORS,
        "operating_revenue_qoq_mrq",
        "营业收入环比增长率（单季度）",
    ),
    "ttm_operating_revenue_yoy": _field(
        _INDICATORS,
        "operating_revenue_yoy_ttm",
        "营业收入同比增长率（TTM）",
    ),
    "latest_operating_revenue_yoy": _field(
        _INDICATORS,
        "operating_revenue_yoy_lf",
        "营业收入同比增长率（最新一期）",
    ),
    "quarterly_gross_profit_yoy": _field(
        _INDICATORS,
        "gross_profit_yoy_mrq",
        "毛利润同比增长率（单季度）",
    ),
    "quarterly_gross_profit_qoq": _field(
        _INDICATORS,
        "gross_profit_qoq_mrq",
        "毛利润环比增长率（单季度）",
    ),
    "ttm_gross_profit_yoy": _field(
        _INDICATORS, "gross_profit_yoy_ttm", "毛利润同比增长率（TTM）"
    ),
    "latest_gross_profit_yoy": _field(
        _INDICATORS,
        "gross_profit_yoy_lf",
        "毛利润同比增长率（最新一期）",
    ),
    "quarterly_operating_profit_yoy": _field(
        _INDICATORS,
        "operating_profit_yoy_mrq",
        "营业利润同比增长率（单季度）",
    ),
    "quarterly_operating_cashflow_yoy": _field(
        _INDICATORS,
        "net_cffoa_yoy_mrq",
        "经营活动现金流净额同比增长率（单季度）",
    ),
    "quarterly_roe_avg_yoy": _field(
        _INDICATORS,
        "roe_avg_mrq_yoy",
        "平均净资产收益率同比增长率（单季度）",
    ),
    "quarterly_operating_cost_yoy": _field(
        _INDICATORS,
        "operating_costs_yoy_mrq",
        "营业成本同比增长率（单季度）",
    ),
    "quarterly_selling_expense_yoy": _field(
        _INDICATORS,
        "selling_epense_yoy_mrq",
        "销售费用同比增长率（单季度；源字段沿用平台拼写）",
    ),
    "quarterly_finance_expense_yoy": _field(
        _INDICATORS,
        "finance_expense_yoy_mrq",
        "财务费用同比增长率（单季度）",
    ),
    "quarterly_research_expense_yoy": _field(
        _INDICATORS,
        "rad_expense_yoy_mrq",
        "研发费用同比增长率（单季度）",
    ),
    "quarterly_cash_paid_for_goods_yoy": _field(
        _INDICATORS,
        "cash_paid_for_goods_yoy_mrq",
        "购买商品、接受劳务支付的现金同比增长率（单季度）",
    ),
    "quarterly_cash_received_yoy": _field(
        _INDICATORS,
        "cash_received_yoy_mrq",
        "销售商品、提供劳务收到的现金同比增长率（单季度）",
    ),
    "quarterly_revenue_per_share_yoy": _field(
        _INDICATORS,
        "revenue_ps_mrq_yoy",
        "每股营业收入同比增长率（单季度）",
    ),
    "ttm_revenue_per_share_yoy": _field(
        _INDICATORS,
        "revenue_ps_ttm_yoy",
        "每股营业收入同比增长率（TTM）",
    ),
    "latest_revenue_per_share_yoy": _field(
        _INDICATORS,
        "revenue_ps_lf_yoy",
        "每股营业收入同比增长率（最新一期）",
    ),

    # ===== 第二批：日频点时财务科目 =====
    "quarterly_net_profit": _field(
        _ITEMS, "net_profit_mrq", "净利润（单季度）"
    ),
    "ttm_net_profit": _field(_ITEMS, "net_profit_ttm", "净利润（TTM）"),
    "latest_net_profit": _field(_ITEMS, "net_profit_lf", "净利润（最新一期）"),
    "quarterly_deducted_net_profit": _field(
        _INDICATORS, "net_profit_deducted_mrq", "扣非净利润（单季度）"
    ),
    "ttm_deducted_net_profit": _field(
        _INDICATORS, "net_profit_deducted_ttm", "扣非净利润（TTM）"
    ),
    "latest_deducted_net_profit": _field(
        _INDICATORS, "net_profit_deducted_lf", "扣非净利润（最新一期）"
    ),
    "quarterly_operating_cashflow": _field(
        _ITEMS, "net_cffoa_mrq", "经营活动产生的现金流量净额（单季度）"
    ),
    "ttm_operating_cashflow": _field(
        _ITEMS, "net_cffoa_ttm", "经营活动产生的现金流量净额（TTM）"
    ),
    "latest_operating_cashflow": _field(
        _ITEMS, "net_cffoa_lf", "经营活动产生的现金流量净额（最新一期）"
    ),

    # ===== 第二批：财务质量与营运能力 =====
    "quarterly_roe_avg": _field(
        _INDICATORS, "roe_avg_mrq", "平均净资产收益率（单季度）"
    ),
    "ttm_roe_avg": _field(
        _INDICATORS, "roe_avg_ttm", "平均净资产收益率（TTM）"
    ),
    "latest_roe_avg": _field(
        _INDICATORS, "roe_avg_lf", "平均净资产收益率（最新一期）"
    ),
    "quarterly_roe_avg_deducted": _field(
        _INDICATORS, "roe_avg_deduct_mrq", "扣非平均净资产收益率（单季度）"
    ),
    "ttm_roe_avg_deducted": _field(
        _INDICATORS, "roe_avg_deduct_ttm", "扣非平均净资产收益率（TTM）"
    ),
    "latest_roe_avg_deducted": _field(
        _INDICATORS, "roe_avg_deduct_lf", "扣非平均净资产收益率（最新一期）"
    ),
    "quarterly_roa_avg": _field(
        _INDICATORS, "roa_avg_mrq", "平均总资产净利率（单季度）"
    ),
    "ttm_roa_avg": _field(
        _INDICATORS, "roa_avg_ttm", "平均总资产净利率（TTM）"
    ),
    "latest_roa_avg": _field(
        _INDICATORS, "roa_avg_lf", "平均总资产净利率（最新一期）"
    ),
    "quarterly_gross_profit_margin": _field(
        _INDICATORS, "gross_profit_rate_mrq", "销售毛利率（单季度）"
    ),
    "ttm_gross_profit_margin": _field(
        _INDICATORS, "gross_profit_rate_ttm", "销售毛利率（TTM）"
    ),
    "latest_gross_profit_margin": _field(
        _INDICATORS, "gross_profit_rate_lf", "销售毛利率（最新一期）"
    ),
    "quarterly_net_profit_margin": _field(
        _INDICATORS, "net_profit_rate_mrq", "销售净利率（单季度）"
    ),
    "ttm_net_profit_margin": _field(
        _INDICATORS, "net_profit_rate_ttm", "销售净利率（TTM）"
    ),
    "latest_net_profit_margin": _field(
        _INDICATORS, "net_profit_rate_lf", "销售净利率（最新一期）"
    ),
    "quarterly_total_assets_turnover": _field(
        _INDICATORS, "total_assets_turnover_mrq", "总资产周转率（单季度）"
    ),
    "ttm_total_assets_turnover": _field(
        _INDICATORS, "total_assets_turnover_ttm", "总资产周转率（TTM）"
    ),
    "latest_total_assets_turnover": _field(
        _INDICATORS, "total_assets_turnover_lf", "总资产周转率（最新一期）"
    ),
    "quarterly_operating_cashflow_to_parent_net_profit": _field(
        _INDICATORS,
        "cffoa_to_net_profit_from_parent_mrq",
        "净利润现金含量（单季度；经营现金流/归母净利润）",
    ),
    "ttm_operating_cashflow_to_parent_net_profit": _field(
        _INDICATORS,
        "cffoa_to_net_profit_from_parent_ttm",
        "净利润现金含量（TTM；经营现金流/归母净利润）",
    ),
    "latest_operating_cashflow_to_parent_net_profit": _field(
        _INDICATORS,
        "cffoa_to_net_profit_from_parent_lf",
        "净利润现金含量（最新一期；经营现金流/归母净利润）",
    ),
}


ADAPTER_SPEC = {
    "name": "financial",
    "output_group": "security_daily",
    "key_columns": ("date", "instrument"),
    "supported_fields": tuple(FINANCIAL_FIELD_MAPPING),
    "context_parameters": (),
}


def list_supported_financial_fields(table=None):
    """返回该适配器支持的标准字段；可按 BigQuant 来源表过滤。"""
    if table is None:
        return sorted(FINANCIAL_FIELD_MAPPING)
    if not isinstance(table, str) or not table.strip():
        raise ValueError("table 必须是非空字符串或 None。")
    table = table.strip()
    if table not in FINANCIAL_TABLE_CONFIG:
        raise KeyError(
            f"未登记财务来源表 {table!r}；可用表："
            f"{sorted(FINANCIAL_TABLE_CONFIG)}"
        )
    return sorted(
        name
        for name, spec in FINANCIAL_FIELD_MAPPING.items()
        if spec["table"] == table
    )


def get_financial_field_info(field_name):
    """返回一个标准字段的映射信息副本，供审计或调试使用。"""
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name 必须是非空字符串。")
    field_name = field_name.strip()
    spec = FINANCIAL_FIELD_MAPPING.get(field_name)
    if spec is None:
        raise KeyError(
            f"未登记财务标准字段 {field_name!r}；可用字段："
            f"{list_supported_financial_fields()}"
        )
    return {"standard_field": field_name, **dict(spec)}


def _validate_mapping_catalog():
    source_pairs = {}
    for name, spec in FINANCIAL_FIELD_MAPPING.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(spec, Mapping):
            raise ValueError("FINANCIAL_FIELD_MAPPING 存在无效条目。")
        table = spec.get("table")
        column = spec.get("column")
        if table not in FINANCIAL_TABLE_CONFIG:
            raise ValueError(f"字段 {name!r} 使用了未登记来源表 {table!r}。")
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"字段 {name!r} 缺少有效 BigQuant 字段名。")
        source_pair = (table, column.strip())
        if source_pair in source_pairs:
            raise ValueError(
                f"标准字段 {source_pairs[source_pair]!r} 和 {name!r} "
                f"重复映射到 {table}.{column}。"
            )
        source_pairs[source_pair] = name


def _normalize_fields(standard_fields):
    if isinstance(standard_fields, str):
        standard_fields = [standard_fields]
    elif not isinstance(standard_fields, Iterable):
        raise TypeError("standard_fields 必须是字段名字符串或字段名序列。")

    fields = []
    for field in standard_fields:
        if not isinstance(field, str) or not field.strip():
            raise ValueError("standard_fields 含空字段名或非字符串字段名。")
        field = field.strip()
        if field not in {"date", "instrument"} and field not in fields:
            fields.append(field)
    if not fields:
        raise ValueError("至少需要一个非 date/instrument 的财务标准字段。")

    unsupported = sorted(set(fields) - set(FINANCIAL_FIELD_MAPPING))
    if unsupported:
        raise ValueError(
            f"BigQuant 财务适配器暂不支持字段：{unsupported}。"
        )
    return fields


def _normalize_date(value, parameter_name):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter_name} 必须是可解析日期：{value!r}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{parameter_name} 不允许为空。")
    return timestamp.strftime("%Y-%m-%d")


def _normalize_dates(dates):
    if isinstance(dates, (str, pd.Timestamp)):
        dates = [dates]
    elif not isinstance(dates, Iterable):
        raise TypeError("dates 必须是日期或日期序列。")
    normalized = [
        _normalize_date(value, f"dates[{index}]")
        for index, value in enumerate(dates, start=1)
    ]
    if not normalized:
        raise ValueError("dates 不能为空。")
    return sorted(set(normalized))


def _normalize_instruments(instruments):
    if instruments is None:
        return None
    if isinstance(instruments, str):
        instruments = [instruments]
    elif not isinstance(instruments, Iterable):
        raise TypeError("instruments 必须是证券代码、代码序列或 None。")
    result = []
    for instrument in instruments:
        if not isinstance(instrument, str) or not instrument.strip():
            raise ValueError("instruments 含空代码或非字符串代码。")
        instrument = instrument.strip()
        if instrument not in result:
            result.append(instrument)
    if not result:
        raise ValueError("instruments 不能为空；请传入至少一个代码或 None。")
    return result


def _resolve_date_selector(start_date, end_date, dates):
    if dates is not None:
        if start_date is not None or end_date is not None:
            raise ValueError("dates 与 start_date/end_date 两种日期选择方式互斥。")
        return {"mode": "dates", "dates": _normalize_dates(dates)}
    if (start_date is None) != (end_date is None):
        raise ValueError("连续区间必须同时提供 start_date 和 end_date。")
    if start_date is None:
        raise ValueError("请传入 dates，或同时传入 start_date 和 end_date。")
    start = _normalize_date(start_date, "start_date")
    end = _normalize_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date 不能晚于 end_date。")
    return {"mode": "range", "start_date": start, "end_date": end}


def _quote_sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _build_partition_filters(selector):
    if selector["mode"] == "range":
        return {"date": [selector["start_date"], selector["end_date"]]}
    return {"date": [selector["dates"][0], selector["dates"][-1]]}


def _group_fields_by_table(fields):
    grouped = defaultdict(list)
    for field in fields:
        grouped[FINANCIAL_FIELD_MAPPING[field]["table"]].append(field)
    return dict(grouped)


def _build_table_sql(table, fields, selector, instruments):
    config = FINANCIAL_TABLE_CONFIG[table]
    date_column = config["date_column"]
    instrument_column = config["instrument_column"]
    select_parts = [
        f"f.{date_column} AS date",
        f"f.{instrument_column} AS instrument",
        *[
            f"f.{FINANCIAL_FIELD_MAPPING[field]['column']} AS {field}"
            for field in fields
        ],
    ]
    if selector["mode"] == "range":
        where_parts = [
            f"f.{date_column} BETWEEN {_quote_sql_literal(selector['start_date'])} "
            f"AND {_quote_sql_literal(selector['end_date'])}"
        ]
    else:
        selected_dates = ", ".join(_quote_sql_literal(x) for x in selector["dates"])
        where_parts = [f"f.{date_column} IN ({selected_dates})"]
    if instruments is not None:
        selected_instruments = ", ".join(_quote_sql_literal(x) for x in instruments)
        where_parts.append(f"f.{instrument_column} IN ({selected_instruments})")
    return "\n".join([
        "SELECT",
        "    " + ",\n    ".join(select_parts),
        f"FROM {table} AS f",
        "WHERE " + "\n  AND ".join(where_parts),
        f"ORDER BY f.{date_column}, f.{instrument_column}",
    ])


def _default_query(sql, filters):
    try:
        import dai
    except ImportError as exc:
        raise ImportError(
            "未能导入 dai；请在 BigQuant 环境运行，或传入 query_func。"
        ) from exc
    return dai.query(sql, filters=filters)


def _call_query_func(query_func, sql, filters):
    try:
        signature = inspect.signature(query_func)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        return query_func(sql, filters=filters)
    parameters = signature.parameters
    accepts_filters = "filters" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    return query_func(sql, filters=filters) if accepts_filters else query_func(sql)


def _to_dataframe(query_result):
    result = query_result.df() if hasattr(query_result, "df") else query_result
    if not isinstance(result, pd.DataFrame):
        raise TypeError("query_func 必须返回 DataFrame 或具有 .df() 的查询结果。")
    return result


def _validate_table_result(result, table, fields):
    expected = ["date", "instrument", *fields]
    missing = sorted(set(expected) - set(result.columns))
    if missing:
        raise ValueError(f"BigQuant 表 {table!r} 的结果缺少字段：{missing}。")
    result = result.loc[:, expected].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any() or result["instrument"].isna().any():
        raise ValueError(f"BigQuant 表 {table!r} 返回无效 date 或空 instrument。")
    duplicated = result.duplicated(["date", "instrument"], keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, ["date", "instrument"]].head(5)
        raise ValueError(
            f"BigQuant 表 {table!r} 返回重复 date + instrument："
            f"{examples.astype(str).to_dict('records')}"
        )
    return result


def _merge_table_panels(panels):
    if not panels:
        return pd.DataFrame(columns=["date", "instrument"])
    merged = panels[0]
    for panel in panels[1:]:
        overlap = set(merged.columns) & set(panel.columns) - {"date", "instrument"}
        if overlap:
            raise ValueError(f"财务来源表出现重复标准字段：{sorted(overlap)}。")
        merged = merged.merge(
            panel, on=["date", "instrument"], how="outer", validate="one_to_one"
        )
    return merged


def _render_progress(completed, total, table, started_at, finished=False):
    elapsed = time.perf_counter() - started_at
    percentage = 100.0 if total == 0 else completed / total * 100.0
    eta = ""
    if 0 < completed < total:
        eta = f" | 预计剩余 {elapsed / completed * (total - completed):.1f}s"
    stage = "完成" if finished else (f"已完成 {table}" if completed else "准备查询")
    print(
        f"\r[BigQuant 财务适配器] {completed}/{total} ({percentage:.1f}%) "
        f"| {stage} | 已耗时 {elapsed:.1f}s{eta}",
        end="",
        flush=True,
    )


def load_financial_raw_data(
    standard_fields,
    start_date=None,
    end_date=None,
    dates=None,
    instruments=None,
    query_func: Optional[Callable] = None,
    show_progress=False,
):
    """读取请求字段的 BigQuant 日频点时财务面板。

    日期请求须二选一：连续区间 ``start_date + end_date``，或离散日期
    列表 ``dates``。返回列严格为 date、instrument 与请求的标准字段；
    不填充缺失值，也不对财务字段做任何计算或改写。
    """
    _validate_mapping_catalog()
    fields = _normalize_fields(standard_fields)
    selector = _resolve_date_selector(start_date, end_date, dates)
    instruments = _normalize_instruments(instruments)
    fields_by_table = _group_fields_by_table(fields)
    partition_filters = _build_partition_filters(selector)
    table_items = list(fields_by_table.items())
    started_at = time.perf_counter()
    panels = []
    if show_progress:
        _render_progress(0, len(table_items), "", started_at)
    try:
        for index, (table, table_fields) in enumerate(table_items, start=1):
            sql = _build_table_sql(table, table_fields, selector, instruments)
            query_result = (
                _default_query(sql, partition_filters)
                if query_func is None
                else _call_query_func(query_func, sql, partition_filters)
            )
            panels.append(_validate_table_result(_to_dataframe(query_result), table, table_fields))
            if show_progress:
                _render_progress(
                    index, len(table_items), table, started_at,
                    finished=index == len(table_items),
                )
        return (
            _merge_table_panels(panels)
            .reindex(columns=["date", "instrument", *fields])
            .sort_values(["date", "instrument"], kind="mergesort")
            .reset_index(drop=True)
        )
    finally:
        if show_progress:
            print()
