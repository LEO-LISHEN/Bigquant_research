# -*- coding: utf-8 -*-
"""按关键词检索因子。"""

import pandas as pd

from factor_lib.factor_hub.discover_factors import discover_factors


def search_factors(keyword):
    """
    在因子名称、类别、说明和已登记扩展信息中检索关键词。
    """
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError("keyword 必须是非空字符串")

    keyword = keyword.strip().lower()
    factors = discover_factors()
    rows = []

    for name, factor in factors.items():
        searchable_text = str(factor).lower()

        if keyword not in searchable_text:
            continue

        direction = factor.get("direction")
        direction_text = {
            1: "正向（值越大越好）",
            -1: "反向（值越小越好）",
        }.get(direction, "未登记")

        rows.append(
            {
                "name": name,
                "category": factor.get("category", "未分类"),
                "direction": direction_text,
                "description": factor.get("description", "未登记"),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["name", "category", "direction", "description"],
    ).sort_values(["category", "name"], ignore_index=True)
