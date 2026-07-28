# -*- coding: utf-8 -*-
"""绘制因子相关系数热力图。"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


def plot_factor_correlation_heatmap(
    correlation_matrix,
    title="因子截面相关系数热力图",
    figsize=(10, 8),
    annotate=True,
    show=True,
    show_progress=False,
):
    """
    绘制相关系数热力图，并返回 fig、ax。

    参数
    ----
    show_progress : bool，默认 False
        是否在终端用单行刷新方式显示绘图阶段。
    """
    start_time = time.perf_counter()

    if show_progress:
        print(
            "\r[相关性热力图] 1/4 校验相关系数矩阵...",
            end="",
            flush=True,
        )

    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("correlation_matrix 必须是方阵")

    if list(correlation_matrix.index) != list(
        correlation_matrix.columns
    ):
        raise ValueError("相关系数矩阵的行名与列名必须一致")

    if show_progress:
        print(
            "\r[相关性热力图] 2/4 配置字体...",
            end="",
            flush=True,
        )

    available_fonts = {
        font.name for font in font_manager.fontManager.ttflist
    }

    for font_name in [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
    ]:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break

    plt.rcParams["axes.unicode_minus"] = False

    labels = list(correlation_matrix.columns)
    values = correlation_matrix.to_numpy(dtype=float)

    if show_progress:
        print(
            "\r[相关性热力图] 3/4 绘制热力图...",
            end="",
            flush=True,
        )

    fig, ax = plt.subplots(figsize=figsize)

    image = ax.imshow(
        values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("平均截面相关系数")

    # 因子很多时不标数字，避免图表不可读。
    if annotate and len(labels) <= 20:
        total_rows = len(labels)

        for row in range(total_rows):
            for col in range(len(labels)):
                value = values[row, col]

                if np.isfinite(value):
                    color = (
                        "white"
                        if abs(value) > 0.5
                        else "black"
                    )

                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )

            if show_progress:
                print(
                    "\r"
                    f"[相关性热力图] 4/4 添加数值标注 "
                    f"| {row + 1}/{total_rows} 行 "
                    f"| {(row + 1) / total_rows:.1%}",
                    end="",
                    flush=True,
                )
    elif show_progress:
        print(
            "\r[相关性热力图] 4/4 跳过数值标注...",
            end="",
            flush=True,
        )

    fig.tight_layout()

    if show:
        plt.show()

    if show_progress:
        elapsed = time.perf_counter() - start_time
        print(
            f"\r[相关性热力图] 已完成 | 耗时：{elapsed:.1f}s",
            end="",
            flush=True,
        )
        print()

    return fig, ax
