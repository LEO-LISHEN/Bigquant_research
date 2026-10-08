# -*- coding: utf-8 -*-
"""查看因子库简表。"""

import pandas as pd

from factor_lib.factor_hub.discover_factors import (
    discover_factor_infos,
    discover_factors,
)


def _info_title(info, fallback):
    """提取 Markdown 一级标题；说明文本无需遵守其他固定结构。"""
    for line in info.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def list_factors():
    """返回因子名称及其研究说明标题，不再依赖经济类别或方向字段。"""
    factors = discover_factors()
    infos = discover_factor_infos()
    rows = [
        {
            "name": name,
            "info_title": _info_title(infos.get(name, ""), name),
        }
        for name in factors
    ]
    return pd.DataFrame(rows, columns=["name", "info_title"]).sort_values(
        "name",
        ignore_index=True,
    )
