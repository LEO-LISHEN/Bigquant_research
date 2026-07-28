# -*- coding: utf-8 -*-
"""自动发现并读取因子脚本中的 FACTOR 信息。"""

import importlib
import sys
from pathlib import Path


def discover_factors():
    """
    扫描 factor_lib 下的因子脚本，返回：
    {因子名称: FACTOR字典}

    跳过 common 与 factor_hub 两类基础设施代码。
    """
    library_root = Path(__file__).resolve().parents[1]
    library_parent = str(library_root.parent)

    # 保证可以按 factor_lib.xxx 的形式导入。
    if library_parent not in sys.path:
        sys.path.insert(0, library_parent)

    excluded_folders = {"common", "factor_hub", "__pycache__"}
    factors = {}

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

        # 允许目录中存在辅助脚本；只有声明 FACTOR 的才算因子。
        if factor is None:
            continue

        if not isinstance(factor, dict):
            raise TypeError(f"{relative_path} 中的 FACTOR 必须是字典")

        name = factor.get("name")
        func = factor.get("func")

        if not name or not isinstance(name, str):
            raise ValueError(f"{relative_path} 的 FACTOR 缺少有效 name")

        if not callable(func):
            raise ValueError(f"{relative_path} 的 FACTOR 缺少可调用 func")

        if name in factors:
            raise ValueError(f"发现重复因子名称：{name}")

        factors[name] = factor.copy()

    return factors
