# -*- coding: utf-8 -*-
"""绘制 IC 与 RankIC 时序图。"""

import time

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
    show_progress=False,
):
    """
    绘制 IC、RankIC 和零轴，并返回 fig、ax。

    参数
    ----
    show_progress : bool，默认 False
        是否在终端用单行刷新方式显示绘图阶段。
    """
    start_time = time.perf_counter()

    if show_progress:
        print(
            "\r[IC 时序图] 1/4 校验数据...",
            end="",
            flush=True,
        )

    required_columns = {date_column, ic_column, rank_ic_column}
    missing_columns = required_columns - set(metrics_timeseries.columns)

    if missing_columns:
        raise ValueError(
            f"指标时序表缺少字段：{sorted(missing_columns)}"
        )

    if show_progress:
        print(
            "\r[IC 时序图] 2/4 配置字体...",
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

    if show_progress:
        print(
            "\r[IC 时序图] 3/4 整理时序数据...",
            end="",
            flush=True,
        )

    data = metrics_timeseries.sort_values(date_column).copy()

    if show_progress:
        print(
            "\r[IC 时序图] 4/4 绘制图形...",
            end="",
            flush=True,
        )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        data[date_column],
        data[ic_column],
        label="IC",
        linewidth=1.5,
    )

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

    if show_progress:
        elapsed = time.perf_counter() - start_time
        print(
            f"\r[IC 时序图] 已完成 | 耗时：{elapsed:.1f}s",
            end="",
            flush=True,
        )
        print()

    return fig, ax
