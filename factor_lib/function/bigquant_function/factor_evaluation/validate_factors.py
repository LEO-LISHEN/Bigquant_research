# -*- coding: utf-8 -*-
"""BigQuant 因子批量运行验证器。

本模块用于验证“因子发现 -> 数据适配 -> 因子计算 -> 输出结构”的完整调用链。
它不构造未来收益标签、不计算 IC，也不运行策略回测。
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Mapping, Sequence

import pandas as pd

from factor_lib.common.data_adapters.bigquant_adapters.loader import (
    get_factor_data_requirements,
    load_factor_raw_data,
)
from factor_lib.factor_hub.discover_factors import discover_factors
from factor_lib.factor_hub.get_factor import get_factor


_SYSTEM_FACTOR_PARAMS = {
    "data",
    "target_dates",
    "as_of_date",
    "show_progress",
    "progress_every",
}


def _query_recent_trading_dates(end_date, required_count, max_history_days):
    """只查询一次交易日历，返回足够覆盖预热期的最近目标日期。"""
    import dai

    end_timestamp = pd.Timestamp(end_date).normalize()
    if pd.isna(end_timestamp):
        raise ValueError("end_date 必须是可解析日期。")
    if required_count <= 0:
        raise ValueError("target_date_count 必须是正整数。")

    # A 股一年约 250 个交易日；四倍日历长度可覆盖长假与较长预热窗口。
    calendar_days = max(180, int((required_count + max_history_days) * 4))
    begin_timestamp = end_timestamp - pd.Timedelta(days=calendar_days)
    sql = f"""
    SELECT DISTINCT date
    FROM cn_stock_bar1d
    WHERE date BETWEEN '{begin_timestamp:%Y-%m-%d}'
                   AND '{end_timestamp:%Y-%m-%d}'
    ORDER BY date
    """
    calendar = dai.query(
        sql,
        filters={
            "date": [
                begin_timestamp.strftime("%Y-%m-%d"),
                end_timestamp.strftime("%Y-%m-%d"),
            ],
        },
    ).df()
    if calendar.empty or "date" not in calendar.columns:
        raise ValueError("未读取到有效的 A 股交易日历。")

    dates = pd.DatetimeIndex(
        pd.to_datetime(calendar["date"], errors="coerce")
    ).dropna().normalize().unique().sort_values()
    if len(dates) < required_count + max_history_days:
        raise ValueError(
            "交易日历不足以同时覆盖测试日期和最长因子预热窗口："
            f"需要至少 {required_count + max_history_days} 个交易日，"
            f"实际只有 {len(dates)} 个。"
        )
    return dates[-required_count:], dates


def _build_raw_dates(calendar, target_dates, history_days):
    """为全部目标日一次性构造历史预热日期集合。"""
    positions = {date: position for position, date in enumerate(calendar)}
    raw_dates = set()
    for target_date in target_dates:
        position = positions.get(target_date)
        if position is None or position < history_days:
            raise ValueError(
                f"目标日期 {target_date:%Y-%m-%d} 的预热数据不足 {history_days} 个交易日。"
            )
        raw_dates.update(calendar[position - history_days : position + 1])
    return sorted(raw_dates)


def _output_factor_column(metadata):
    """从统一 output_schema 中确定唯一的因子值列。"""
    schema = metadata.get("output_schema", {})
    if not isinstance(schema, Mapping):
        raise ValueError("FACTOR 缺少规范的 output_schema。")
    candidates = [
        column
        for column in schema
        if column not in {"date", "instrument"}
    ]
    if len(candidates) != 1:
        raise ValueError(
            "output_schema 必须声明且只声明一个因子值列，"
            f"实际为：{candidates}"
        )
    return candidates[0]


def _security_daily_row_count(raw_data):
    """兼容普通 DataFrame 与 BigQuant 分域数据容器。"""
    if isinstance(raw_data, pd.DataFrame):
        return len(raw_data)
    if hasattr(raw_data, "get_security_daily"):
        return len(raw_data.get_security_daily())
    return None


def _normalize_factor_names(factors, factor_names):
    if factor_names is None:
        return sorted(factors)
    if isinstance(factor_names, str):
        factor_names = [factor_names]
    names = list(factor_names)
    missing = sorted(set(names) - set(factors))
    if missing:
        raise ValueError(f"未发现因子：{missing}")
    return names


def _factor_params_for(factor_name, factor_params_by_name):
    if factor_params_by_name is None:
        return {}
    if not isinstance(factor_params_by_name, Mapping):
        raise TypeError("factor_params_by_name 必须是字典或 None。")
    params = factor_params_by_name.get(factor_name, {})
    if not isinstance(params, Mapping):
        raise TypeError(f"{factor_name} 的参数必须是字典。")
    return dict(params)


def validate_factors(
    end_date,
    factor_names=None,
    factor_params_by_name=None,
    instruments=None,
    target_date_count=30,
    adapter_options_by_name=None,
    keep_factor_results=False,
    show_progress=True,
):
    """批量验证因子是否能在 BigQuant 完整调用链中正常计算。

    参数
    ----
    end_date : str 或 datetime
        测试区间的最后一个可用日期。函数自动选取此前最近的
        ``target_date_count`` 个 A 股交易日。
    factor_names : str 或 sequence，可选
        待验证因子；None 表示动态发现的全部因子。
    factor_params_by_name : dict，可选
        形式为 ``{因子名: {参数名: 参数值}}``，用于覆盖因子默认参数。
    instruments : sequence[str]，可选
        指定测试股票池；None 表示由适配器加载可用的全 A 股票面板。
        对有截面中性化的因子，不应使用过小股票池。
    target_date_count : int，默认 30
        连续测试的目标交易日数，必须至少为 30。
    adapter_options_by_name : dict，可选
        形式为 ``{因子名: {数据域: {BigQuant 专属选项}}}``，用于市场指数
        等需要适配器专属上下文的因子。
    keep_factor_results : bool，默认 False
        True 时在返回结果中保留每个因子的输出 DataFrame；默认关闭以节省内存。
    show_progress : bool，默认 True
        输出按因子更新的简洁运行状态。

    返回
    ----
    dict
        ``summary`` 为每因子一行的汇总表；``target_dates`` 为测试日期；
        当 ``keep_factor_results=True`` 时，``factor_results`` 保存成功因子的结果。
    """
    if not isinstance(target_date_count, int) or isinstance(target_date_count, bool):
        raise TypeError("target_date_count 必须是整数。")
    if target_date_count < 30:
        raise ValueError("target_date_count 至少为 30。")
    if instruments is not None and isinstance(instruments, str):
        instruments = [instruments]
    if instruments is not None:
        instruments = list(instruments)
    if adapter_options_by_name is not None and not isinstance(
        adapter_options_by_name,
        Mapping,
    ):
        raise TypeError("adapter_options_by_name 必须是字典或 None。")

    factors = discover_factors()
    names = _normalize_factor_names(factors, factor_names)
    requirements_by_name = {}
    max_history_days = 0

    # 先统一解析参数与数据窗口，再只读取一次交易日历。
    for name in names:
        params = _factor_params_for(name, factor_params_by_name)
        requirements = get_factor_data_requirements(name, params)
        requirements_by_name[name] = requirements
        window = requirements.get("data_window", {})
        history = int(window.get("lookback_trading_days", 0))
        if history < 0:
            raise ValueError(f"{name} 的 lookback_trading_days 不能为负数。")
        max_history_days = max(max_history_days, history)

    target_dates, calendar = _query_recent_trading_dates(
        end_date=end_date,
        required_count=target_date_count,
        max_history_days=max_history_days,
    )
    records = []
    factor_results = {}
    all_started_at = time.perf_counter()

    for position, name in enumerate(names, start=1):
        started_at = time.perf_counter()
        metadata = factors[name]
        params = _factor_params_for(name, factor_params_by_name)
        history_days = int(
            requirements_by_name[name]["data_window"].get(
                "lookback_trading_days",
                0,
            )
        )
        adapter_options = (
            None
            if adapter_options_by_name is None
            else adapter_options_by_name.get(name)
        )
        base_record = {
            "factor_name": name,
            "target_date_count": len(target_dates),
            "history_days": history_days,
            "status": "failed",
            "raw_rows": None,
            "output_rows": None,
            "date_coverage": None,
            "non_null_ratio": None,
            "runtime_seconds": None,
            "message": "",
        }

        if show_progress:
            print(
                f"[因子运行验证] {position}/{len(names)} "
                f"开始：{name}（{len(target_dates)} 个目标日）...",
                flush=True,
            )

        try:
            raw_dates = _build_raw_dates(
                calendar=calendar,
                target_dates=target_dates,
                history_days=history_days,
            )
            raw_data = load_factor_raw_data(
                factor_name=name,
                dates=raw_dates,
                factor_params=params,
                instruments=instruments,
                adapter_options=adapter_options,
                show_progress=False,
            )
            base_record["raw_rows"] = _security_daily_row_count(raw_data)

            call_params = {
                key: value
                for key, value in params.items()
                if key not in _SYSTEM_FACTOR_PARAMS
            }
            factor_data = get_factor(
                name,
                raw_data,
                target_dates=target_dates,
                as_of_date=target_dates[-1],
                show_progress=False,
                **call_params,
            )
            if not isinstance(factor_data, pd.DataFrame):
                raise TypeError("因子函数未返回 pandas.DataFrame。")

            factor_column = _output_factor_column(metadata)
            expected_columns = {"date", "instrument", factor_column}
            actual_columns = set(factor_data.columns)
            if actual_columns != expected_columns:
                raise ValueError(
                    "因子输出字段不符合规范："
                    f"期望 {sorted(expected_columns)}，实际 {sorted(actual_columns)}"
                )
            if factor_data.duplicated(["date", "instrument"], keep=False).any():
                raise ValueError("因子输出存在重复的 date + instrument 记录。")

            result_dates = pd.DatetimeIndex(
                pd.to_datetime(factor_data["date"], errors="coerce")
            ).dropna().normalize().unique()
            covered_dates = target_dates.intersection(result_dates)
            base_record["output_rows"] = len(factor_data)
            base_record["date_coverage"] = len(covered_dates) / len(target_dates)
            base_record["non_null_ratio"] = float(
                pd.to_numeric(
                    factor_data[factor_column],
                    errors="coerce",
                ).notna().mean()
            )

            if len(covered_dates) != len(target_dates):
                missing_dates = target_dates.difference(result_dates)
                raise ValueError(
                    "因子输出未覆盖全部目标日期："
                    f"{[date.strftime('%Y-%m-%d') for date in missing_dates[:5]]}"
                )

            base_record["status"] = (
                "warning"
                if base_record["non_null_ratio"] == 0.0
                else "passed"
            )
            if base_record["status"] == "warning":
                base_record["message"] = "调用成功，但全部因子值为 NaN；请检查预热、股票池或截面样本量。"
            if keep_factor_results:
                factor_results[name] = factor_data
        except Exception as error:  # 验证器需要逐因子继续运行。
            base_record["message"] = (
                f"{type(error).__name__}: {error}"
            )[:1000]
            if show_progress:
                traceback.print_exc(limit=1)
        finally:
            base_record["runtime_seconds"] = round(
                time.perf_counter() - started_at,
                3,
            )
            records.append(base_record)
            if show_progress:
                print(
                    f"[因子运行验证] {position}/{len(names)} 完成：{name} | "
                    f"{base_record['status']} | "
                    f"耗时 {base_record['runtime_seconds']:.1f}s",
                    flush=True,
                )

    summary = pd.DataFrame(records).sort_values(
        ["status", "factor_name"],
        kind="mergesort",
    ).reset_index(drop=True)
    if show_progress:
        elapsed = time.perf_counter() - all_started_at
        counts = summary["status"].value_counts().to_dict()
        print(
            f"[因子运行验证] 完成：{len(summary)} 个因子 | "
            f"通过 {counts.get('passed', 0)} | "
            f"警告 {counts.get('warning', 0)} | "
            f"失败 {counts.get('failed', 0)} | 耗时 {elapsed:.1f}s。",
            flush=True,
        )

    result = {
        "summary": summary,
        "target_dates": target_dates,
    }
    if keep_factor_results:
        result["factor_results"] = factor_results
    return result
