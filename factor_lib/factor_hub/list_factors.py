# -*- coding: utf-8 -*-
"""查看因子库简表。"""

import pandas as pd

from factor_lib.factor_hub.discover_factors import discover_factors


def list_factors(category=None):
    """
    返回因子简表。

    参数
    ----
    category : str，可选
        例如 valuation、momentum、quality。
    """
    factors = discover_factors()
    rows = []

    for name, factor in factors.items():
        factor_category = factor.get("category", "未分类")

        if category is not None and factor_category != category:
            continue

        direction = factor.get("direction")
        direction_text = {
            1: "正向（值越大越好）",
            -1: "反向（值越小越好）",
        }.get(direction, "未登记")

        rows.append(
            {
                "name": name,
                "category": factor_category,
                "direction": direction_text,
                "status": factor.get("status", "未登记"),
                "description": factor.get("description", "未登记"),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["name", "category", "direction", "status", "description"],
    ).sort_values(["category", "name"], ignore_index=True)
