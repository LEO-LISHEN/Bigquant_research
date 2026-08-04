# -*- coding: utf-8 -*-
"""动态发现因子脚本，并分离运行契约与研究说明。"""

import importlib
import sys
from pathlib import Path


def _discover_factor_records():
    """扫描因子模块，返回 ``{name: {"factor", "info"}}``。

    ``FACTOR`` 是运行期严格读取的机器契约；``FACTOR_INFO`` 只是可选的
    Markdown 研究说明。本函数将两者并列保存，但不会把研究说明混入
    ``FACTOR`` 字典。
    """
    library_root = Path(__file__).resolve().parents[1]
    library_parent = str(library_root.parent)

    if library_parent not in sys.path:
        sys.path.insert(0, library_parent)

    excluded_folders = {"common", "factor_hub", "function", "__pycache__"}
    records = {}

    for file_path in sorted(library_root.rglob("*.py")):
        relative_path = file_path.relative_to(library_root)
        if file_path.name == "__init__.py":
            continue
        if relative_path.parts[0] in excluded_folders:
            continue

        module_suffix = ".".join(relative_path.with_suffix("").parts)
        module_name = f"{library_root.name}.{module_suffix}"

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ImportError(
                f"无法导入因子脚本：{relative_path}；原因：{exc}"
            ) from exc

        factor = getattr(module, "FACTOR", None)
        if factor is None:
            continue
        if not isinstance(factor, dict):
            raise TypeError(f"{relative_path} 中的 FACTOR 必须是字典。")

        name = factor.get("name")
        func = factor.get("func")
        if not name or not isinstance(name, str):
            raise ValueError(f"{relative_path} 的 FACTOR 缺少有效 name。")
        if not callable(func):
            raise ValueError(f"{relative_path} 的 FACTOR 缺少可调用 func。")
        if name in records:
            raise ValueError(f"发现重复因子名称：{name}")

        info = getattr(module, "FACTOR_INFO", "")
        if info is None:
            info = ""
        if not isinstance(info, str):
            raise TypeError(
                f"{relative_path} 中的 FACTOR_INFO 必须是字符串或未定义。"
            )
        records[name] = {"factor": factor.copy(), "info": info}

    return records


def discover_factors():
    """返回运行期 FACTOR 契约：``{因子名称: FACTOR字典}``。"""
    return {
        name: record["factor"].copy()
        for name, record in _discover_factor_records().items()
    }


def discover_factor_infos():
    """返回研究说明：``{因子名称: FACTOR_INFO Markdown文本}``。"""
    return {
        name: record["info"]
        for name, record in _discover_factor_records().items()
    }
