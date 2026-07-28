# -*- coding: utf-8 -*-
"""按因子名称计算因子。"""

from factor_lib.factor_hub.discover_factors import discover_factors


def get_factor(name, data, as_of_date=None, **params):
    """
    根据因子名称调用对应的计算函数。

    所有因子应遵守统一接口：
    calc_xxx(data, as_of_date=None, **params)
    """
    factors = discover_factors()

    if name not in factors:
        available = ", ".join(sorted(factors)) or "暂无已登记因子"
        raise ValueError(f"未找到因子：{name}；可用因子：{available}")

    factor_func = factors[name]["func"]

    call_params = {"data": data, **params}
    if as_of_date is not None:
        call_params["as_of_date"] = as_of_date

    return factor_func(**call_params)
