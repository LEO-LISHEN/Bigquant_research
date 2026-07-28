# -*- coding: utf-8 -*-
"""查看单个因子的完整档案。"""

from factor_lib.factor_hub.discover_factors import discover_factors


def describe_factor(name):
    """
    返回指定因子的完整 FACTOR 信息。

    函数对象会转换为函数名称，便于展示与打印。
    """
    factors = discover_factors()

    if name not in factors:
        available = ", ".join(sorted(factors)) or "暂无已登记因子"
        raise ValueError(f"未找到因子：{name}；可用因子：{available}")

    details = factors[name].copy()

    if "func" in details:
        details["func"] = details["func"].__name__

    return details
