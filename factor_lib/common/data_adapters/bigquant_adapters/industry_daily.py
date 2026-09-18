# -*- coding: utf-8 -*-
"""BigQuant 申万一级行业指数日频原始数据适配器。

本模块只负责将 ``cn_stock_industry_sw_bar1d`` 的原始日行情字段
映射为因子库使用的语义字段，并按日期与行业指数范围取数。

不计算价量同比、不把行业数据广播至个股面板，也不构造交易信号。
``instrument`` 始终保留为申万一级行业指数代码（如 ``801010.SWI``）。
"""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Iterable
from typing import Callable, Optional

import pandas as pd


# 左侧为因子库跨数据源使用的语义字段；右侧为 BigQuant 原始字段。
# 使用 industry_ 前缀，避免与个股日频的 close、volume 等字段混淆。
INDUSTRY_DAILY_FIELD_MAPPING = {
    "industry_open": "open",
    "industry_high": "high",
    "industry_low": "low",
    "industry_close": "close",
    "industry_pre_close": "pre_close",
    "industry_volume": "volume",
    "industry_amount": "amount",
    "industry_turn": "turn",
    "industry_change_ratio": "change_ratio",
}


ADAPTER_SPEC = {
    "name": "industry_daily",
    "output_group": "industry_daily",
    "key_columns": ("date", "instrument"),
    "supported_fields": tuple(INDUSTRY_DAILY_FIELD_MAPPING),
    # ``industry_indices`` 是行业指数代码列表或 None；None 表示全部可得
    # 的申万一级行业指数。它不能由股票池 instruments 替代。
    "context_parameters": ("industry_indices",),
}


def list_supported_industry_daily_fields():
    """返回当前行业指数日频适配器支持的语义字段名。"""
    return sorted(INDUSTRY_DAILY_FIELD_MAPPING)


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
        raise ValueError("至少需要一个非主键行业日频字段。")

    unsupported = sorted(
        set(fields) - set(INDUSTRY_DAILY_FIELD_MAPPING)
    )
    if unsupported:
        raise KeyError(
            f"行业日频适配器不支持字段：{unsupported}；可用字段："
            f"{list_supported_industry_daily_fields()}"
        )
    return fields


def _normalize_industry_indices(industry_indices):
    """规范行业指数代码；None 表示不限制行业指数。"""
    if industry_indices is None:
        return None
    if isinstance(industry_indices, str):
        values = [industry_indices]
    elif isinstance(industry_indices, Iterable):
        values = list(industry_indices)
    else:
        raise TypeError(
            "industry_indices 必须是行业指数代码字符串、代码序列或 None。"
        )

    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "industry_indices 中存在空代码或非字符串代码。"
            )
        value = value.strip()
        if value not in normalized:
            normalized.append(value)

    if not normalized:
        raise ValueError(
            "industry_indices 为空；请传入至少一个行业指数代码或使用 None。"
        )
    return normalized


def _normalize_date(value, parameter_name):
    try:
        timestamp = pd.Timestamp(value).normalize()
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

    normalized = []
    for position, value in enumerate(dates, start=1):
        normalized.append(_normalize_date(value, f"dates[{position}]"))
    if not normalized:
        raise ValueError("dates 不能为空。")
    return sorted(set(normalized))


def _resolve_date_selector(start_date, end_date, dates):
    uses_range = start_date is not None or end_date is not None
    uses_dates = dates is not None

    if uses_range and uses_dates:
        raise ValueError(
            "日期选择方式必须二选一：start_date/end_date 或 dates。"
        )
    if not uses_range and not uses_dates:
        raise ValueError("必须提供 start_date/end_date 或 dates。")
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


def _build_industry_daily_sql(fields, industry_indices, date_selector):
    select_parts = ["date AS date", "instrument AS instrument"]
    select_parts.extend(
        f"{INDUSTRY_DAILY_FIELD_MAPPING[field]} AS {field}"
        for field in fields
    )

    where_parts = []
    if industry_indices is not None:
        instruments_sql = ", ".join(
            _quote_sql_literal(item) for item in industry_indices
        )
        where_parts.append(f"instrument IN ({instruments_sql})")

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

    return "\n".join(
        [
            "SELECT",
            "    " + ",\n    ".join(select_parts),
            "FROM cn_stock_industry_sw_bar1d",
            "WHERE " + "\n  AND ".join(where_parts),
            "ORDER BY date, instrument",
        ]
    )


def _default_query(sql, filters):
    try:
        import dai
    except ImportError as exc:
        raise ImportError(
            "未能导入 dai。请在 BigQuant 环境中运行，"
            "或通过 query_func 传入兼容的查询函数。"
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


def _render_progress(stage, started_at, detail=""):
    elapsed = time.perf_counter() - started_at
    message = f"\r[BigQuant 行业日频适配器] {stage}"
    if detail:
        message += f" | {detail}"
    message += f" | 已耗时 {elapsed:.1f}s"
    print(message.ljust(180), end="", flush=True)


def _run_with_query_heartbeat(
    action,
    stage,
    started_at,
    detail,
    show_progress,
    interval_seconds=2.0,
):
    """查询阻塞期间显示存活状态，不伪造平台查询进度。"""
    if not show_progress:
        return action()

    stop_event = threading.Event()

    def heartbeat():
        while not stop_event.wait(interval_seconds):
            _render_progress(
                stage,
                started_at,
                detail=f"{detail}，查询仍在运行",
            )

    worker = threading.Thread(
        target=heartbeat,
        name="bigquant-industry-daily-query-progress",
        daemon=True,
    )
    worker.start()
    try:
        return action()
    finally:
        stop_event.set()
        worker.join(timeout=max(interval_seconds, 0.1))


def load_industry_daily_raw_data(
    standard_fields,
    industry_indices=None,
    start_date=None,
    end_date=None,
    dates=None,
    instruments=None,
    query_func: Optional[Callable] = None,
    show_progress=False,
):
    """加载申万一级行业指数日频原始数据。

    Parameters
    ----------
    standard_fields : str 或 sequence[str]
        请求的行业语义字段，如 ``industry_close``、``industry_volume``。
        返回值固定包含 ``date`` 与 ``instrument``。
    industry_indices : str、sequence[str] 或 None，默认 None
        要读取的申万一级行业指数代码，例如 ``"801010.SWI"``。
        None 表示不限制行业指数，返回数据源在所选日期内的全部指数。
    start_date, end_date : str 或 datetime，可选
        连续闭区间，必须与 dates 二选一。
    dates : 日期或 sequence[日期]，可选
        离散日期列表，必须与 start_date/end_date 二选一。
    instruments : 忽略
        为与统一 loader 接口兼容而保留。这里的行业指数筛选必须使用
        ``industry_indices``，避免误把股票池代码应用到行业指数数据源。
    query_func : callable，可选
        本地测试或替代数据客户端。可返回 pandas.DataFrame，或具有
        ``.df()`` 方法的 BigQuant 查询结果。
    show_progress : bool，默认 False
        是否显示查询阶段的单行状态。

    Returns
    -------
    pandas.DataFrame
        ``date``、``instrument`` 与所请求的语义字段；其中 instrument 是
        申万一级行业指数代码，而不是个股代码。
    """
    del instruments

    if not isinstance(show_progress, bool):
        raise TypeError("show_progress 必须是 bool。")

    fields = _normalize_fields(standard_fields)
    selected_indices = _normalize_industry_indices(industry_indices)
    date_selector = _resolve_date_selector(start_date, end_date, dates)
    sql = _build_industry_daily_sql(
        fields,
        selected_indices,
        date_selector,
    )
    filters = _build_partition_filters(date_selector)
    started_at = time.perf_counter()

    if show_progress:
        index_detail = (
            "全部行业" if selected_indices is None
            else f"{len(selected_indices)} 个行业指数"
        )
        _render_progress(
            "[1/3] 提交行业指数查询",
            started_at,
            detail=f"{index_detail}，{len(fields)} 个字段",
        )

    try:
        def execute_and_convert():
            if query_func is None:
                query_result = _default_query(sql, filters)
            else:
                query_result = _call_query_func(
                    query_func,
                    sql,
                    filters,
                )
            return _to_dataframe(query_result).copy()

        query_detail = (
            "全部行业" if selected_indices is None
            else f"{len(selected_indices)} 个行业指数"
        )
        result = _run_with_query_heartbeat(
            execute_and_convert,
            "[2/3] 执行并接收行业指数查询",
            started_at,
            query_detail,
            show_progress,
        )

        expected_columns = {"date", "instrument", *fields}
        missing = sorted(expected_columns - set(result.columns))
        if missing:
            raise ValueError(f"行业日频查询结果缺少字段：{missing}。")

        result["date"] = pd.to_datetime(
            result["date"], errors="coerce"
        ).dt.normalize()
        if result["date"].isna().any():
            raise ValueError("行业日频查询结果包含无效 date。")
        if result["instrument"].isna().any():
            raise ValueError("行业日频查询结果包含空行业指数代码。")

        duplicated = result.duplicated(["date", "instrument"], keep=False)
        if duplicated.any():
            examples = (
                result.loc[duplicated, ["date", "instrument"]]
                .head(5)
                .astype(str)
                .to_dict("records")
            )
            raise ValueError(
                "行业日频查询结果在 date + instrument 上重复："
                f"{examples}"
            )

        result = (
            result[["date", "instrument", *fields]]
            .sort_values(["date", "instrument"], kind="mergesort")
            .reset_index(drop=True)
        )
        if show_progress:
            _render_progress(
                "[3/3] 行业日频数据准备完成",
                started_at,
                detail=f"{len(result):,} 行",
            )
        return result
    finally:
        if show_progress:
            print()
