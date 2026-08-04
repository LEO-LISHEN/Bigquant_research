# -*- coding: utf-8 -*-
"""BigQuant 日频点时财务数据适配器。

本模块只负责：
1. 将因子库的语义标准字段映射到 BigQuant 日频点时财务表；
2. 按连续日期区间或离散交易日列表拉取数据；
3. 当字段来自多张日频点时财务表时，分别查询后合并；
4. 显式传递 BigQuant 日期分区 filters。

本模块不负责：
- 计算因子；
- 计算同比、环比或复合增长率；
- 填充缺失值、去极值、标准化或中性化；
- 决定信号日、调仓日或执行日；
- 将非日频原始财报向前填充到交易日。

非日频原始财报应由独立适配器读取，并保留报告期、公告日和
实际可用日等时间字段，不能在这里直接按 date + instrument 合并。
"""

from __future__ import annotations

import inspect
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Callable, Optional

import pandas as pd


# 这里只登记已经按交易日组织、能够使用 date + instrument 唯一定位的
# BigQuant 日频点时财务表。非日频原始财报不得登记到这里。
FINANCIAL_TABLE_CONFIG = {
    "cn_stock_factors_financial_indicators": {
        "date_column": "date",
        "instrument_column": "instrument",
        "description": "BigQuant 日频财务指标表",
    },
    "cn_stock_factors_financial_items": {
        "date_column": "date",
        "instrument_column": "instrument",
        "description": "BigQuant 日频财务科目因子表",
    },
}


# 左侧为因子库统一使用的数据源无关标准字段。
# table/column 为 BigQuant 平台映射；description 用于查询和审计。
#
# 新增字段前必须先在 BigQuant 官方字段文档或实际表结构中确认：
# 1. 来源表与字段真实存在；
# 2. 字段含义、MRQ/TTM/LF 口径正确；
# 3. 来源表已经是按交易日组织的点时面板。
FINANCIAL_FIELD_MAPPING = {
    # ===== 盈利质量：净资产收益率 =====
    "quarterly_average_roe": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "roe_avg_mrq",
        "description": "净资产收益率（平均，单季度）",
    },
    # ===== 成长：净利润 =====
    "quarterly_net_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "net_profit_yoy_mrq",
        "description": "净利润同比增长率（单季度）",
    },
    "quarterly_net_profit_qoq": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "net_profit_qoq_mrq",
        "description": "净利润环比增长率（单季度）",
    },
    "ttm_net_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "net_profit_yoy_ttm",
        "description": "净利润同比增长率（滚动十二期）",
    },
    "latest_net_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "net_profit_yoy_lf",
        "description": "净利润同比增长率（最新一期）",
    },
    "quarterly_parent_net_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "net_profit_to_parent_yoy_mrq",
        "description": "归母净利润同比增长率（单季度）",
    },
    "ttm_parent_net_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "net_profit_to_parent_yoy_ttm",
        "description": "归母净利润同比增长率（滚动十二期）",
    },
    "latest_parent_net_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "net_profit_to_parent_yoy_lf",
        "description": "归母净利润同比增长率（最新一期）",
    },
    # ===== 成长：营业收入 =====
    "quarterly_operating_revenue_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "operating_revenue_yoy_mrq",
        "description": "营业收入同比增长率（单季度）",
    },
    "quarterly_operating_revenue_qoq": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "operating_revenue_qoq_mrq",
        "description": "营业收入环比增长率（单季度）",
    },
    "ttm_operating_revenue_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "operating_revenue_yoy_ttm",
        "description": "营业收入同比增长率（滚动十二期）",
    },
    "latest_operating_revenue_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "operating_revenue_yoy_lf",
        "description": "营业收入同比增长率（最新一期）",
    },
    # ===== 成长：毛利润、营业利润和费用 =====
    "quarterly_gross_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "gross_profit_yoy_mrq",
        "description": "毛利润同比增长率（单季度）",
    },
    "quarterly_gross_profit_qoq": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "gross_profit_qoq_mrq",
        "description": "毛利润环比增长率（单季度）",
    },
    "ttm_gross_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "gross_profit_yoy_ttm",
        "description": "毛利润同比增长率（滚动十二期）",
    },
    "latest_gross_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "gross_profit_yoy_lf",
        "description": "毛利润同比增长率（最新一期）",
    },
    "quarterly_operating_profit_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "operating_profit_yoy_mrq",
        "description": "营业利润同比增长率（单季度）",
    },
    "quarterly_operating_cost_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "operating_costs_yoy_mrq",
        "description": "营业成本同比增长率（单季度）",
    },
    "quarterly_selling_expense_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "selling_epense_yoy_mrq",
        "description": "销售费用同比增长率（单季度）",
    },
    "quarterly_finance_expense_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "finance_expense_yoy_mrq",
        "description": "财务费用同比增长率（单季度）",
    },
    "quarterly_research_expense_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "rad_expense_yoy_mrq",
        "description": "研发费用同比增长率（单季度）",
    },
    # ===== 现金流成长 =====
    "quarterly_cash_paid_for_goods_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "cash_paid_for_goods_yoy_mrq",
        "description": "购买商品、接受劳务支付的现金同比增长率（单季度）",
    },
    "quarterly_cash_received_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "cash_received_yoy_mrq",
        "description": "销售商品、提供劳务收到的现金同比增长率（单季度）",
    },
    # ===== 每股营业收入成长 =====
    "quarterly_revenue_per_share_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "revenue_ps_mrq_yoy",
        "description": "每股营业收入同比增长率（单季度）",
    },
    "ttm_revenue_per_share_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "revenue_ps_ttm_yoy",
        "description": "每股营业收入同比增长率（滚动十二期）",
    },
    "latest_revenue_per_share_yoy": {
        "table": "cn_stock_factors_financial_indicators",
        "column": "revenue_ps_lf_yoy",
        "description": "每股营业收入同比增长率（最新一期）",
    },
}


# 财务适配器虽然独立查询，但其结果已经是日频点时股票面板，因而可以与
# daily.py 的输出在 security_daily 粒度内按 date + instrument 合并。
ADAPTER_SPEC = {
    "name": "financial",
    "output_group": "security_daily",
    "key_columns": ("date", "instrument"),
    "supported_fields": tuple(FINANCIAL_FIELD_MAPPING),
    "context_parameters": (),
}


def list_supported_financial_fields(table=None):
    """返回当前适配器支持的语义标准财务字段。

    参数
    ----
    table : str，可选
        只列出指定 BigQuant 来源表的标准字段。
    """
    if table is None:
        return sorted(FINANCIAL_FIELD_MAPPING)

    if not isinstance(table, str) or not table.strip():
        raise ValueError("table 必须是非空字符串或 None。")
    table = table.strip()
    if table not in FINANCIAL_TABLE_CONFIG:
        raise KeyError(
            f"未登记财务来源表 {table!r}。可用来源表："
            f"{sorted(FINANCIAL_TABLE_CONFIG)}"
        )

    return sorted(
        field_name
        for field_name, specification in FINANCIAL_FIELD_MAPPING.items()
        if specification["table"] == table
    )


def get_financial_field_info(field_name):
    """返回一个标准财务字段的 BigQuant 映射信息副本。"""
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name 必须是非空字符串。")
    field_name = field_name.strip()

    specification = FINANCIAL_FIELD_MAPPING.get(field_name)
    if specification is None:
        raise KeyError(
            f"未登记财务标准字段 {field_name!r}。可用字段："
            f"{list_supported_financial_fields()}"
        )

    return {
        "standard_field": field_name,
        **dict(specification),
    }


def _validate_mapping_catalog():
    """在查询前检查字段目录结构，避免错误配置生成错误 SQL。"""
    source_pairs = {}

    for field_name, specification in FINANCIAL_FIELD_MAPPING.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError("FINANCIAL_FIELD_MAPPING 含有无效标准字段名。")
        if not isinstance(specification, Mapping):
            raise ValueError(
                f"字段 {field_name!r} 的映射必须是字典。"
            )

        table = specification.get("table")
        column = specification.get("column")
        if table not in FINANCIAL_TABLE_CONFIG:
            raise ValueError(
                f"字段 {field_name!r} 引用了未登记来源表 {table!r}。"
            )
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                f"字段 {field_name!r} 缺少有效的 BigQuant column。"
            )

        source_pair = (table, column.strip())
        previous_field = source_pairs.get(source_pair)
        if previous_field is not None:
            raise ValueError(
                f"标准字段 {previous_field!r} 和 {field_name!r} "
                f"重复映射到 {table}.{column}。"
            )
        source_pairs[source_pair] = field_name


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
            "至少需要指定一个非 date/instrument 的财务标准字段。"
        )

    unsupported = sorted(
        set(fields) - set(FINANCIAL_FIELD_MAPPING)
    )
    if unsupported:
        raise ValueError(
            "BigQuant 财务适配器暂不支持字段："
            f"{unsupported}。可用字段："
            f"{list_supported_financial_fields()}"
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


def _normalize_dates(dates):
    if isinstance(dates, (str, pd.Timestamp)):
        dates = [dates]
    elif not isinstance(dates, Iterable):
        raise TypeError("dates 必须是日期或日期序列。")

    normalized = [
        _normalize_date(value, f"dates[{position}]")
        for position, value in enumerate(dates, start=1)
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
        raise TypeError(
            "instruments 必须是证券代码、代码序列或 None。"
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
            "instruments 为空；请传入至少一个代码或使用 None。"
        )
    return normalized


def _resolve_date_selector(start_date, end_date, dates):
    has_dates = dates is not None
    has_start = start_date is not None
    has_end = end_date is not None

    if has_dates and (has_start or has_end):
        raise ValueError(
            "日期选择方式冲突：dates 与 start_date/end_date 不能混用。"
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

    normalized_start = _normalize_date(start_date, "start_date")
    normalized_end = _normalize_date(end_date, "end_date")
    if normalized_start > normalized_end:
        raise ValueError("start_date 不能晚于 end_date。")

    return {
        "mode": "range",
        "start_date": normalized_start,
        "end_date": normalized_end,
    }


def _quote_sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _build_partition_filters(date_selector):
    if date_selector["mode"] == "range":
        start_date = date_selector["start_date"]
        end_date = date_selector["end_date"]
    else:
        start_date = date_selector["dates"][0]
        end_date = date_selector["dates"][-1]
    return {"date": [start_date, end_date]}


def _group_fields_by_table(fields):
    grouped = defaultdict(list)
    for field_name in fields:
        table = FINANCIAL_FIELD_MAPPING[field_name]["table"]
        grouped[table].append(field_name)
    return dict(grouped)


def _build_table_sql(table, fields, date_selector, instruments):
    table_config = FINANCIAL_TABLE_CONFIG[table]
    date_column = table_config["date_column"]
    instrument_column = table_config["instrument_column"]

    select_parts = [
        f"f.{date_column} AS date",
        f"f.{instrument_column} AS instrument",
    ]
    for field_name in fields:
        source_column = FINANCIAL_FIELD_MAPPING[field_name]["column"]
        select_parts.append(
            f"f.{source_column} AS {field_name}"
        )

    if date_selector["mode"] == "range":
        where_parts = [
            f"f.{date_column} BETWEEN "
            f"{_quote_sql_literal(date_selector['start_date'])} "
            f"AND {_quote_sql_literal(date_selector['end_date'])}"
        ]
    else:
        date_sql = ", ".join(
            _quote_sql_literal(date)
            for date in date_selector["dates"]
        )
        where_parts = [f"f.{date_column} IN ({date_sql})"]

    if instruments is not None:
        instrument_sql = ", ".join(
            _quote_sql_literal(instrument)
            for instrument in instruments
        )
        where_parts.append(
            f"f.{instrument_column} IN ({instrument_sql})"
        )

    return "\n".join(
        [
            "SELECT",
            "    " + ",\n    ".join(select_parts),
            f"FROM {table} AS f",
            "WHERE " + "\n  AND ".join(where_parts),
            f"ORDER BY f.{date_column}, f.{instrument_column}",
        ]
    )


def _default_query(sql, filters):
    try:
        import dai
    except ImportError as exc:
        raise ImportError(
            "未能导入 dai。请在 BigQuant 环境运行，"
            "或通过 query_func 注入兼容查询函数。"
        ) from exc
    return dai.query(sql, filters=filters)


def _call_query_func(query_func, sql, filters):
    """兼容 query_func(sql) 和 query_func(sql, filters=...) 两种测试接口。"""
    try:
        signature = inspect.signature(query_func)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        parameters = signature.parameters
        accepts_filters = (
            "filters" in parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        if accepts_filters:
            return query_func(sql, filters=filters)
        return query_func(sql)

    # 无法检查签名的可调用对象按推荐接口调用。
    return query_func(sql, filters=filters)


def _to_dataframe(query_result):
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
    return result


def _validate_table_result(result, table, fields):
    expected_columns = ["date", "instrument", *fields]
    missing_columns = sorted(
        set(expected_columns) - set(result.columns)
    )
    if missing_columns:
        raise ValueError(
            f"BigQuant 表 {table!r} 的查询结果缺少字段："
            f"{missing_columns}。请检查字段权限和映射。"
        )

    result = result.loc[:, expected_columns].copy()
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    ).dt.normalize()
    if result["date"].isna().any():
        raise ValueError(
            f"BigQuant 表 {table!r} 返回了无法解析的 date。"
        )
    if result["instrument"].isna().any():
        raise ValueError(
            f"BigQuant 表 {table!r} 返回了空 instrument。"
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
            f"BigQuant 表 {table!r} 返回重复的 date + instrument："
            f"{examples}"
        )

    return result


def _merge_table_panels(panels):
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
                "多个财务来源表返回了重复标准字段："
                f"{sorted(overlapping)}。"
            )

        merged = merged.merge(
            panel,
            on=["date", "instrument"],
            how="outer",
            validate="one_to_one",
        )

    return merged


def _render_progress(
    completed,
    total,
    table,
    started_at,
    finished=False,
    stage=None,
    detail="",
):
    elapsed = time.perf_counter() - started_at
    percentage = 100.0 if total == 0 else completed / total * 100.0

    if completed > 0 and completed < total:
        remaining = elapsed / completed * (total - completed)
        eta_text = f"，预计剩余 {remaining:.1f}s"
    else:
        eta_text = ""

    if stage is None:
        if finished:
            stage = "完成"
        elif completed == 0:
            stage = "准备查询"
        else:
            stage = f"已完成 {table}"

    message = (
        "\r[BigQuant 财务适配器] "
        f"{completed}/{total}（{percentage:6.2f}%）"
        f"，{stage}"
    )
    if detail:
        message += f"，{detail}"
    message += f"，耗时 {elapsed:.1f}s{eta_text}"
    print(message.ljust(180), end="", flush=True)


def load_financial_raw_data(
    standard_fields,
    start_date=None,
    end_date=None,
    dates=None,
    instruments=None,
    query_func: Optional[Callable] = None,
    show_progress=False,
):
    """读取 BigQuant 日频点时财务标准字段面板。

    日期选择必须二选一：
    - ``start_date`` + ``end_date``：连续闭区间；
    - ``dates``：离散目标交易日列表。

    字段可以来自多张已登记的日频点时财务表。本函数按来源表分别
    查询，再按 date + instrument 外连接，最终返回固定包含
    date、instrument 和请求标准字段的 DataFrame。

    本函数不填充、不去重、不计算财务指标，也不读取非日频原始财报。
    """
    _validate_mapping_catalog()
    fields = _normalize_fields(standard_fields)
    date_selector = _resolve_date_selector(
        start_date,
        end_date,
        dates,
    )
    instruments = _normalize_instruments(instruments)
    fields_by_table = _group_fields_by_table(fields)
    partition_filters = _build_partition_filters(date_selector)

    started_at = time.perf_counter()
    panels = []
    table_items = list(fields_by_table.items())
    total_tables = len(table_items)

    if show_progress:
        _render_progress(
            completed=0,
            total=total_tables,
            table="",
            started_at=started_at,
        )

    try:
        for index, (table, table_fields) in enumerate(
            table_items,
            start=1,
        ):
            if show_progress:
                _render_progress(
                    completed=index - 1,
                    total=total_tables,
                    table=table,
                    started_at=started_at,
                    stage="正在查询来源表",
                    detail=f"{len(table_fields)} 个字段",
                )
            sql = _build_table_sql(
                table=table,
                fields=table_fields,
                date_selector=date_selector,
                instruments=instruments,
            )

            if query_func is None:
                query_result = _default_query(
                    sql,
                    filters=partition_filters,
                )
            else:
                query_result = _call_query_func(
                    query_func,
                    sql,
                    partition_filters,
                )

            panel = _validate_table_result(
                _to_dataframe(query_result),
                table=table,
                fields=table_fields,
            )
            panels.append(panel)

            if show_progress:
                _render_progress(
                    completed=index,
                    total=total_tables,
                    table=table,
                    started_at=started_at,
                    finished=index == total_tables,
                    detail=f"返回 {len(panel):,} 行",
                )

        if show_progress:
            _render_progress(
                completed=total_tables,
                total=total_tables,
                table="",
                started_at=started_at,
                stage="正在合并财务来源表",
                detail=f"{total_tables} 张表",
            )
        result = _merge_table_panels(panels)
        expected_columns = ["date", "instrument", *fields]
        result = (
            result.reindex(columns=expected_columns)
            .sort_values(
                ["date", "instrument"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        if show_progress:
            _render_progress(
                completed=total_tables,
                total=total_tables,
                table="",
                started_at=started_at,
                finished=True,
                detail=f"合并后 {len(result):,} 行",
            )
        return result
    finally:
        if show_progress:
            print()
