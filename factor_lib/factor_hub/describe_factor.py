# -*- coding: utf-8 -*-
"""查看单个因子的运行契约与研究说明。"""

from factor_lib.factor_hub.discover_factors import (
    discover_factor_infos,
    discover_factors,
)


def describe_factor(name):
    """返回 ``FACTOR`` 与原样 ``FACTOR_INFO``，不解析 Markdown 文本。"""
    factors = discover_factors()
    if name not in factors:
        available = ", ".join(sorted(factors)) or "暂无已登记因子"
        raise ValueError(f"未找到因子：{name}；可用因子：{available}")

    contract = factors[name].copy()
    contract["func"] = contract["func"].__name__
    return {
        "FACTOR": contract,
        "FACTOR_INFO": discover_factor_infos().get(name, ""),
    }
