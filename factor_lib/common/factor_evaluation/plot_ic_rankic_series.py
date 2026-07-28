# -*- coding: utf-8 -*-
"""绘制 IC 与 RankIC 时序图。"""

import matplotlib.pyplot as plt
from matplotlib import font_manager


def plot_ic_rankic_series(
    metrics_timeseries,
    date_column="signal_date",
    ic_column="ic",
    rank_ic_column="rank_ic",
    title="因子 IC 与 RankIC 时序",
    figsize=(14, 5),
    show=True,
):
    """绘制 IC、RankIC 和零轴，并返回 fig、ax。"""
    required_columns = {date_column, ic_column, rank_ic_column}
    missing_columns = required_columns - set(metrics_timeseries.columns)

    if missing_columns:
        raise ValueError(f"指标时序表缺少字段：{sorted(missing_columns)}")

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

    data = metrics_timeseries.sort_values(date_column).copy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data[date_column], data[ic_column], label="IC", linewidth=1.5)
    ax.plot(
        data[date_column],
        data[rank_ic_column],
        label="RankIC",
        linewidth=1.5,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("信号日")
    ax.set_ylabel("相关系数")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
