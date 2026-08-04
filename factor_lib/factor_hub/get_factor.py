# -*- coding: utf-8 -*-
"""按因子名称计算因子。"""

import inspect

from factor_lib.factor_hub.discover_factors import discover_factors


def get_factor(name, data, as_of_date=None, **params):
    """
    根据因子名称调用对应的计算函数。

    单一股票日频因子遵守接口：
    ``calc_xxx(data, as_of_date=None, **params)``。

    当 ``data`` 是分粒度数据容器时，旧因子仍会收到其中的
    ``security_daily`` DataFrame；显式声明 ``domain_data`` 参数的
    多数据域因子还会同时收到完整容器。因而旧因子无需立即改造，
    新因子也不需要把市场数据广播到每只股票。
    """
    factors = discover_factors()

    if name not in factors:
        available = ", ".join(sorted(factors)) or "暂无已登记因子"
        raise ValueError(f"未找到因子：{name}；可用因子：{available}")

    factor_func = factors[name]["func"]

    is_data_bundle = (
        hasattr(data, "get_security_daily")
        and hasattr(data, "get_domain")
    )
    factor_input = data.get_security_daily() if is_data_bundle else data

    call_params = {"data": factor_input, **params}
    if is_data_bundle:
        try:
            signature = inspect.signature(factor_func)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "domain_data" in signature.parameters:
            call_params["domain_data"] = data

    if as_of_date is not None:
        call_params["as_of_date"] = as_of_date

    return factor_func(**call_params)
