"""项目科研绘图的统一中文字体与字号规范。"""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_chinese_style() -> None:
    """应用论文插图通用样式；调用后仍可按单图需要覆盖局部参数。"""

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 14,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )

