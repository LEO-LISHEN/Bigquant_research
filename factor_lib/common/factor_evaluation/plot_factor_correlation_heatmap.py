# -*- coding: utf-8 -*-
"""绘制因子相关系数热力图。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


def plot_factor_correlation_heatmap(
    correlation_matrix,
    title="因子截面相关系数热力图",
    figsize=(10, 8),
    annotate=True,
    show=True,
):
    """绘制相关系数热力图，并返回 fig、ax。"""
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("correlation_matrix 必须是方阵")

    if list(correlation_matrix.index) != list(correlation_matrix.columns):
        raise ValueError("相关系数矩阵的行名与列名必须一致")

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
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
        for row in range(len(labels)):
            for col in range(len(labels)):
                value = values[row, col]
                if np.isfinite(value):
                    color = "white" if abs(value) > 0.5 else "black"
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
