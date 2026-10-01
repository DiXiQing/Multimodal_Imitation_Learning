"""

画图脚本

plot_finger_tracking.py

读取形如以下列的 CSV：
    time_s, target_area_px2, finger_width_px, velocity_mps

生成两种图：
  1) plot_time_series()  -> 三个纵向排列的子图（Area / Width / Velocity vs time）
                             对应 "medium_trial1 - Area / Width / Velocity over time" 那种图
  2) plot_3d_surface()   -> 三维散点(按时间上色) + 二次曲面拟合，
                             对应 "Aperture vs Area vs Acceleration (surface fit R^2=...)" 那种图
     - 加速度是速度对时间的数值导数 (np.gradient)，脚本内自动计算，不需要 CSV 里有这一列。

用法：
    直接改下面 "==== 配置 ====" 里的三个变量，然后运行：
        python plot_finger_tracking.py

依赖：
    pip install numpy pandas matplotlib

注意：
    脚本里用了 `str | None` 这种类型注解写法，这是 Python 3.10+ 的语法。
    加了下面这行 `from __future__ import annotations` 之后，
    Python 3.8/3.9 也能正常运行，不会再报
    "TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (激活 3D 投影)


# ==== 配置：在这里改文件名 ====
# 曲面图支持多组 CSV。把需要叠加的文件路径都放进这个列表即可。
CSV_PATHS = [
    r"D:\Code\Multimodal_Imitation_Learning\临时文件\Trial1.csv",
    r"D:\Code\Multimodal_Imitation_Learning\临时文件\Trial2.csv",
    r"D:\Code\Multimodal_Imitation_Learning\临时文件\Trial3.csv",
]

OUT_PREFIX = "finger_tracking"            # 输出图片文件名前缀
SHOW_PLOTS = False                        # True 则额外弹窗显示（本地运行时用）


# ---------- 数据读取与预处理 ----------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"time_s", "target_area_px2", "finger_width_px", "velocity_mps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")
    df = df.sort_values("time_s").reset_index(drop=True)
    return df

# ---------- 图1: 三联时间序列 ----------

def plot_time_series(df: pd.DataFrame, title: str = "Area / Width / Velocity over time", save_path: str | None = None):
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)

    axes[0].plot(df["time_s"], df["target_area_px2"], color="darkgreen", linewidth=2.5)
    axes[0].plot(df["time_s"], df["target_area_px2"], color="darkgreen", linewidth=2.5, label="Object apparent size")
    axes[0].legend(loc="upper left", fontsize=14, frameon=False)
    axes[0].set_ylabel("Object Apparent Size\n(px^2)", fontsize=14)

    axes[1].plot(df["time_s"], df["finger_width_px"], color="navy", linewidth=2.5)
    axes[1].plot(df["time_s"], df["finger_width_px"], color="navy", linewidth=2.5, label="Fingertip aperture")
    axes[1].legend(loc="upper left", fontsize=14, frameon=False)
    axes[1].set_ylabel("Fingertip Aperture\n(px)", fontsize=14)

    axes[2].plot(df["time_s"], df["velocity_mps"], color="firebrick", linewidth=2.5)
    axes[2].plot(df["time_s"], df["velocity_mps"], color="firebrick", linewidth=2.5, label="Hand velocity")
    axes[2].legend(loc="upper left", fontsize=14, frameon=False)
    axes[2].set_ylabel("Hand Velocity\n(m/s)", fontsize=14)
    axes[2].set_xlabel("Time (s)", fontsize=14)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"saved: {save_path}")

    return fig


# ---------- 图2: 三维散点 + 拟合曲面 ----------

def fit_quadratic_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    拟合 z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    返回 (系数数组, R^2)
    """
    A = np.column_stack([x**2, y**2, x * y, x, y, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    z_pred = A @ coeffs
    ss_res = np.sum((z - z_pred) ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return coeffs, r2


def plot_3d_surface(
    data_frames: list[pd.DataFrame] | pd.DataFrame,
    title: str = "Aperture vs Area vs Acceleration",
    save_path: str | None = None,
    labels: list[str] | None = None,
):
    """绘制一组或多组 CSV 的三维散点和各自二次拟合曲面。"""
    if isinstance(data_frames, pd.DataFrame):
        data_frames = [data_frames]
    if not data_frames:
        raise ValueError("至少需要提供一个 CSV 数据集")

    if labels is None:
        labels = [f"Trial {i + 1}" for i in range(len(data_frames))]
    if len(labels) != len(data_frames):
        raise ValueError("labels 的数量必须与 CSV 数据集数量一致")

    all_times = np.concatenate([df["time_s"].to_numpy() for df in data_frames])
    norm = Normalize(vmin=all_times.min(), vmax=all_times.max())
    surface_colors = ["gray", "#2F6B9A", "#C96B28", "#5B8C5A", "#8A5A9B"]
    marker_styles = ["o", "^", "s", "D", "P"]
    fit_results = []

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i, (df, label) in enumerate(zip(data_frames, labels)):
        x = df["target_area_px2"].to_numpy()
        y = df["velocity_mps"].to_numpy()
        z = df["finger_width_px"].to_numpy()
        t = df["time_s"].to_numpy()

        coeffs, r2 = fit_quadratic_surface(x, y, z)
        fit_results.append((label, r2))

        xi = np.linspace(x.min(), x.max(), 40)
        yi = np.linspace(y.min(), y.max(), 40)
        XI, YI = np.meshgrid(xi, yi)
        ZI = (
            coeffs[0] * XI**2
            + coeffs[1] * YI**2
            + coeffs[2] * XI * YI
            + coeffs[3] * XI
            + coeffs[4] * YI
            + coeffs[5]
        )

        color = surface_colors[i % len(surface_colors)]
        ax.plot_surface(
            XI,
            YI,
            ZI,
            color=color,
            alpha=0.22,
            linewidth=0,
            antialiased=True,
        )
        ax.scatter(
            x,
            y,
            z,
            c=t,
            cmap="viridis",
            norm=norm,
            s=30,
            marker=marker_styles[i % len(marker_styles)],
            depthshade=True,
        )

    # 用代理图例标记每一组数据对应的曲面/散点。
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_styles[i % len(marker_styles)],
            color="w",
            markerfacecolor=surface_colors[i % len(surface_colors)],
            markeredgecolor="black",
            markersize=8,
            label=f"{label} (R²={r2:.3f})",
        )
        for i, (label, r2) in enumerate(fit_results)
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, frameon=True)

    ax.set_xlabel("Object apparent size", fontsize=16, labelpad=12)
    ax.set_ylabel("Hand velocity", fontsize=16, labelpad=12)
    ax.set_zlabel("Fingertip aperture", fontsize=16, labelpad=12)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="z", labelsize=12)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    scalar_mappable.set_array(all_times)
    cbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Time (s)", fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"saved: {save_path}")
    return fig, fit_results


# ---------- 主入口 ----------

def main():
    if not CSV_PATHS:
        raise ValueError("CSV_PATHS 不能为空，请至少填写一个 CSV 路径")

    data_frames = [load_data(path) for path in CSV_PATHS]
    labels = [Path(path).stem for path in CSV_PATHS]

    # 时间序列图保持原行为：绘制列表中的第一组数据。
    plot_time_series(data_frames[0], save_path=f"{OUT_PREFIX}_timeseries.png")
    _, fit_results = plot_3d_surface(
        data_frames,
        labels=labels,
        save_path=f"{OUT_PREFIX}_3d_surface.png",
    )
    for label, r2 in fit_results:
        print(f"{label}: surface fit R^2 = {r2:.3f}")

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
