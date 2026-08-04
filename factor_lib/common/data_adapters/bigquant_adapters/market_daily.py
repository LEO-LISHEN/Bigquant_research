# -*- coding: utf-8 -*-
"""BigQuant 市场指数日频原始数据适配器。

本模块只负责三件事：
1. 将因子库统一使用的市场指数名称映射为 BigQuant 指数代码；
2. 将市场日频语义字段映射到 BigQuant 指数日行情表；
3. 按连续日期区间或离散日期列表拉取原始数据。

本模块不计算指数收益率、不向股票面板广播市场数据，也不计算因子。
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Iterable
from typing import Callable, Optional

import pandas as pd


# 左侧是 factor_lib 跨数据源统一使用的名称；右侧只属于 BigQuant。
# 不在这里保存中文展示名，避免把展示信息混入查询映射。
MARKET_INDEX_CODE_MAPPING = {
    "csi_all_share": "000985.CSI",
    "csi_300": "000300.SH",
    "csi_500": "000905.SH",
    "csi_1000": "000852.SH",
    "sse_composite": "000001.SH",
    "szse_component": "399001.SZ",
    "chinext": "399006.SZ",
    "star_50": "000688.SH",
}


MARKET_DAILY_FIELD_MAPPING = {
    "market_open": "open",
    "market_high": "high",
    "market_low": "low",
    "market_close": "close",
    "market_pre_close": "pre_close",
    "market_volume": "volume",
    "market_amount": "amount",
    "market_turn": "turn",
    "market_change": "change",
    "market_change_ratio": "change_ratio",
}


ADAPTER_SPEC = {
    "name": "market_daily",
    "output_group": "market_daily",
    "key_columns": ("date", "market_index"),
    "supported_fields": tuple(MARKET_DAILY_FIELD_MAPPING),
    "context_parameters": ("market_index",),
}


def list_supported_market_indices():
    """返回可传给因子参数 ``market_index`` 的统一指数名称。"""
    return sorted(MARKET_INDEX_CODE_MAPPING)


def list_supported_market_daily_fields():
    """返回市场日频适配器支持的语义标准字段。"""
    return sorted(MARKET_DAILY_FIELD_MAPPING)


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
        if field not in {"date", "market_index"} and field not in fields:
            fields.append(field)

    if not fields:
        raise ValueError("至少需要一个市场日频标准字段。")

    unsupported = sorted(
        set(fields) - set(MARKET_DAILY_FIELD_MAPPING)
    )
    if unsupported:
        raise KeyError(
            f"市场日频适配器不支持字段：{unsupported}；"
            f"可用字段：{list_supported_market_daily_fields()}"
        )
    return fields


def _normalize_market_indices(market_index):
    if isinstance(market_index, str):
        values = [market_index]
    elif isinstance(market_index, Iterable):
        values = list(market_index)
    else:
        raise TypeError(
            "market_index 必须是统一指数名称字符串或字符串序列。"
        )

    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("market_index 中存在空值或非字符串值。")
        value = value.strip()
        if value not in MARKET_INDEX_CODE_MAPPING:
            raise KeyError(
                f"未知市场指数统一名称：{value!r}；"
                f"可用名称：{list_supported_market_indices()}"
            )
        if value not in normalized:
            normalized.append(value)

    if not normalized:
        raise ValueError("market_index 不能为空。")
    return normalized


def _normalize_date(value, parameter_name):
    try:
        result = pd.Timestamp(value).normalize()
    except Exception as exc:
        raise ValueError(f"{parameter_name} 不是有效日期：{value!r}") from exc
    if pd.isna(result):
        raise ValueError(f"{parameter_name} 不是有效日期：{value!r}")
    return result.strftime("%Y-%m-%d")


def _normalize_dates(dates):
    if isinstance(dates, (str, pd.Timestamp)):
        dates = [dates]
    elif not isinstance(dates, Iterable):
        raise TypeError("dates 必须是日期或日期序列。")

    normalized = sorted({_normalize_date(item, "dates") for item in dates})
    if not normalized:
        raise ValueError("dates 不能为空。")
    return normalized


def _resolve_date_selector(start_date, end_date, dates):
    uses_range = start_date is not None or end_date is not None
    uses_dates = dates is not None

    if uses_range and uses_dates:
        raise ValueError(
            "日期选择方式必须二选一：start_date/end_date 或 dates。"
        )
    if not uses_range and not uses_dates:
        raise ValueError(
            "必须提供 start_date/end_date 或 dates。"
        )

    if uses_dates:
        return {"mode": "dates", "dates": _normalize_dates(dates)}

    if start_date is None or end_date is None:
        raise ValueError("连续区间必须同时提供 start_date 和 end_date。")
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


def _build_market_daily_sql(fields, market_indices, date_selector):
    code_to_name = {
        MARKET_INDEX_CODE_MAPPING[name]: name
        for name in market_indices
    }
    codes_sql = ", ".join(
        _quote_sql_literal(code) for code in code_to_name
    )

    select_parts = [
        "date AS date",
        "instrument AS source_market_code",
    ]
    select_parts.extend(
        f"{MARKET_DAILY_FIELD_MAPPING[field]} AS {field}"
        for field in fields
    )

    where_parts = [f"instrument IN ({codes_sql})"]
    if date_selector["mode"] == "range":
        where_parts.append(
            "date BETWEEN "
            f"{_quote_sql_literal(date_selector['start_date'])} AND "
            f"{_quote_sql_literal(date_selector['end_date'])}"
        )
    else:
        dates_sql = ", ".join(
            _quote_sql_literal(date) for date in date_selector["dates"]
        )
        where_parts.append(f"date IN ({dates_sql})")

    sql = "\n".join(
        [
            "SELECT",
            "    " + ",\n    ".join(select_parts),
            "FROM cn_stock_index_bar1d",
            "WHERE " + "\n  AND ".join(where_parts),
            "ORDER BY date, source_market_code",
        ]
    )
    return sql, code_to_name


def _default_query(sql, filters):
    try:
        import dai
    except ImportError as exc:
        raise ImportError(
            "未能导入 dai。请在 BigQuant 环境运行，"
            "或通过 query_func 传入兼容查询函数。"
        ) from exc
    return dai.query(sql, filters=filters)


def _call_query_func(query_func, sql, filters):
    try:
        signature = inspect.signature(query_func)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        accepts_filters = (
            "filters" in signature.parameters
            or any(
                parameter.kind == parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )
        if accepts_filters:
            return query_func(sql, filters=filters)
    return query_func(sql)


def _to_dataframe(query_result):
    if isinstance(query_result, pd.DataFrame):
        return query_result
    if hasattr(query_result, "df"):
        result = query_result.df()
        if isinstance(result, pd.DataFrame):
            return result
    raise TypeError(
        "query_func 必须返回 pandas.DataFrame 或具有 .df() 的查询结果。"
    )


def load_market_daily_raw_data(
    standard_fields,
    market_index,
    start_date=None,
    end_date=None,
    dates=None,
    instruments=None,
    query_func: Optional[Callable] = None,
    show_progress=False,
):
    """按日期拉取一个或多个市场指数的日频原始数据。

    ``market_index`` 使用 factor_lib 的统一名称，例如 ``csi_all_share``；
    返回值固定包含 ``date``、``market_index`` 和请求字段。参数
    ``instruments`` 仅为与 loader 的统一适配器接口兼容，在本适配器中忽略。
    """
    del instruments

    fields = _normalize_fields(standard_fields)
    market_indices = _normalize_market_indices(market_index)
    date_selector = _resolve_date_selector(start_date, end_date, dates)
    sql, code_to_name = _build_market_daily_sql(
        fields,
        market_indices,
        date_selector,
    )
    filters = _build_partition_filters(date_selector)
    started_at = time.perf_counter()

    if show_progress:
        print(
            "\r[BigQuant 市场日频适配器] 开始拉取原始数据...",
            end="",
            flush=True,
        )

    try:
        if query_func is None:
            query_result = _default_query(sql, filters)
        else:
            query_result = _call_query_func(query_func, sql, filters)
        result = _to_dataframe(query_result).copy()

        expected_source_columns = {
            "date",
            "source_market_code",
            *fields,
        }
        missing = sorted(expected_source_columns - set(result.columns))
        if missing:
            raise ValueError(
                f"市场日频查询结果缺少字段：{missing}。"
            )

        result["date"] = pd.to_datetime(
            result["date"], errors="coerce"
        ).dt.normalize()
        if result["date"].isna().any():
            raise ValueError("市场日频查询结果包含无效 date。")

        result["market_index"] = result[
            "source_market_code"
        ].map(code_to_name)
        if result["market_index"].isna().any():
            unknown_codes = sorted(
                result.loc[
                    result["market_index"].isna(),
                    "source_market_code",
                ]
                .dropna()
                .astype(str)
                .unique()
            )
            raise ValueError(
                f"查询返回了未请求或未映射的指数代码：{unknown_codes}。"
            )

        duplicated = result.duplicated(
            ["date", "market_index"], keep=False
        )
        if duplicated.any():
            examples = (
                result.loc[duplicated, ["date", "market_index"]]
                .head(5)
                .astype(str)
                .to_dict("records")
            )
            raise ValueError(
                "市场日频查询结果在 date + market_index 上重复："
                f"{examples}"
            )

        return (
            result[["date", "market_index", *fields]]
            .sort_values(["date", "market_index"], kind="mergesort")
            .reset_index(drop=True)
        )
    finally:
        if show_progress:
            elapsed = time.perf_counter() - started_at
            print(
                "\r[BigQuant 市场日频适配器] "
                f"完成，耗时 {elapsed:.1f}s。"
            )
