# -*- coding: utf-8 -*-
"""按因子名称计算因子。"""

import inspect

from factor_lib.factor_hub.discover_factors import discover_factors


def get_factor(
    name,
    data,
    target_dates=None,
    as_of_date=None,
    **params,
):
    """
    根据因子名称调用对应的计算函数。

    因子函数的 ``data`` 参数接收其 ``FACTOR`` 声明的主数据域。
    未声明 ``primary_data_domain`` 的旧因子默认接收
    ``security_daily``，从而保持原有选股因子兼容。

    当 ``data`` 是分粒度数据容器时，显式声明 ``domain_data`` 参数的
    多数据域因子还会同时收到完整容器。因而行业日频、市场日频和个股
    日频数据无需相互广播。
    """
    factors = discover_factors()

    if name not in factors:
        available = ", ".join(sorted(factors)) or "暂无已登记因子"
        raise ValueError(f"未找到因子：{name}；可用因子：{available}")

    metadata = factors[name]
    factor_func = metadata["func"]

    primary_data_domain = metadata.get(
        "primary_data_domain",
        "security_daily",
    )
    if (
        not isinstance(primary_data_domain, str)
        or not primary_data_domain.strip()
    ):
        raise ValueError(
            f"因子 {name!r} 的 primary_data_domain 必须是非空字符串。"
        )
    primary_data_domain = primary_data_domain.strip()

    is_data_bundle = hasattr(data, "get_domain")
    if is_data_bundle:
        try:
            factor_input = data.get_domain(primary_data_domain)
        except KeyError as exc:
            domain_names = getattr(data, "domain_names", ())
            raise ValueError(
                f"因子 {name!r} 需要主数据域 "
                f"{primary_data_domain!r}，但当前数据包仅包含："
                f"{list(domain_names)}。"
            ) from exc
    else:
        factor_input = data

    call_params = {"data": factor_input, **params}
    if is_data_bundle:
        try:
            signature = inspect.signature(factor_func)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "domain_data" in signature.parameters:
            call_params["domain_data"] = data

    if target_dates is not None:
        call_params["target_dates"] = target_dates

    if as_of_date is not None:
        call_params["as_of_date"] = as_of_date

    return factor_func(**call_params)
