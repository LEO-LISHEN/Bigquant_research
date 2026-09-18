# -*- coding: utf-8 -*-
"""按关键词检索因子名称与研究说明。"""

import pandas as pd

from factor_lib.factor_hub.discover_factors import (
    discover_factor_infos,
    discover_factors,
)
from factor_lib.factor_hub.list_factors import _info_title


def search_factors(keyword):
    """在因子名称及自由格式的 ``FACTOR_INFO`` 中检索关键词。"""
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError("keyword 必须是非空字符串。")

    keyword = keyword.strip().lower()
    factors = discover_factors()
    infos = discover_factor_infos()
    rows = []
    for name in factors:
        info = infos.get(name, "")
        if keyword in name.lower() or keyword in info.lower():
            rows.append(
                {
                    "name": name,
                    "info_title": _info_title(info, name),
                }
            )

    return pd.DataFrame(rows, columns=["name", "info_title"]).sort_values(
        "name",
        ignore_index=True,
    )
