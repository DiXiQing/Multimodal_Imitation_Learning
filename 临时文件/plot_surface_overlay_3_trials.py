"""
将三组抓取数据的三变量二次曲面叠加到同一个 3D 坐标轴。

依赖：
    pip install numpy pandas matplotlib

使用方法：
    1. 修改下面 CSV_PATHS 中的三个文件路径；
    2. 运行：python plot_surface_overlay_3_trials.py

支持两种 CSV 列名：

原始 grasp_data.csv：
    t_camera, finger_width, object_area, velocity

处理后的 CSV：
    time_s, finger_width_px, target_area_px2, velocity_mps
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath


# ==================== 配置区 ====================

CSV_PATHS = [
    r"D:\Code\Multimodal_Imitation_Learning\Data\Validation\D\TRIAL_Black_20260929_104825\grasp_data.csv",
    r"D:\Code\Multimodal_Imitation_Learning\Data\Validation\D\TRIAL_Black_20260929_104907\grasp_data.csv",
    r"D:\Code\Multimodal_Imitation_Learning\Data\Validation\D\TRIAL_Black_20260929_104931\grasp_data.csv",
]

LABELS = ["Trial 1", "Trial 2", "Trial 3"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]

OUT_PATH = "finger_tracking_surface_overlay_3_trials.png"
SHOW_PLOT = False

# True：只在每组数据 x-y 点的凸包范围内绘制曲面，避免无数据区域的过度外推。
MASK_SURFACE_TO_DATA_HULL = True


# ==================== 数据与拟合 ====================

def load_trial(csv_path: str):
    """读取一组数据，并统一返回 time、x、y、z。"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    df = pd.read_csv(path)

    raw_columns = {"t_camera", "finger_width", "object_area", "velocity"}
    processed_columns = {
        "time_s",
        "finger_width_px",
        "target_area_px2",
        "velocity_mps",
    }

    if raw_columns.issubset(df.columns):
        time_col = "t_camera"
        x_col = "object_area"
        y_col = "velocity"
        z_col = "finger_width"
    elif processed_columns.issubset(df.columns):
        time_col = "time_s"
        x_col = "target_area_px2"
        y_col = "velocity_mps"
        z_col = "finger_width_px"
    else:
        raise ValueError(
            f"{path} 的列名不匹配。需要以下两组列之一：\n"
            f"  原始格式：{sorted(raw_columns)}\n"
            f"  处理格式：{sorted(processed_columns)}\n"
            f"实际列名：{list(df.columns)}"
        )

    df = df[[time_col, x_col, y_col, z_col]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().sort_values(time_col).reset_index(drop=True)

    if len(df) < 6:
        raise ValueError(f"{path} 有效数据点太少，至少需要 6 个点。")

    time = df[time_col].to_numpy(float)
    time = time - time[0]
    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)
    z = df[z_col].to_numpy(float)
    return time, x, y, z


def fit_quadratic_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """拟合 z = a*x² + b*y² + c*x*y + d*x + e*y + f。"""
    design = np.column_stack([x**2, y**2, x * y, x, y, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    prediction = design @ coeffs
    ss_res = np.sum((z - prediction) ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
    return coeffs, r2


def convex_hull(points: np.ndarray) -> np.ndarray:
    """返回二维点集的凸包顶点，使用单调链算法。"""
    unique = sorted(set(map(tuple, points)))
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def make_surface(x, y, coeffs, grid_size=50):
    """在本组数据范围内生成曲面，并可选地遮掉凸包外区域。"""
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    ZI = (
        coeffs[0] * XI**2
        + coeffs[1] * YI**2
        + coeffs[2] * XI * YI
        + coeffs[3] * XI
        + coeffs[4] * YI
        + coeffs[5]
    )

    if MASK_SURFACE_TO_DATA_HULL:
        hull = convex_hull(np.column_stack([x, y]))
        if len(hull) >= 3:
            polygon = MplPath(hull)
            inside = polygon.contains_points(
                np.column_stack([XI.ravel(), YI.ravel()])
            ).reshape(XI.shape)
            ZI = np.where(inside, ZI, np.nan)

    return XI, YI, ZI


# ==================== 绘图 ====================

def padded_limits(values, fraction=0.05, lower_bound=None):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    lo, hi = finite.min(), finite.max()
    padding = (hi - lo) * fraction
    if padding == 0:
        padding = max(abs(lo) * fraction, 1.0)
    lo -= padding
    hi += padding
    if lower_bound is not None:
        lo = max(lower_bound, lo)
    return lo, hi


def main():
    if len(CSV_PATHS) != 3:
        raise ValueError("CSV_PATHS 必须正好包含三组数据路径。")
    if len(LABELS) != 3 or len(COLORS) != 3:
        raise ValueError("LABELS 和 COLORS 都必须正好包含三个元素。")

    trials = []
    for path, label, color in zip(CSV_PATHS, LABELS, COLORS):
        time, x, y, z = load_trial(path)
        coeffs, r2 = fit_quadratic_surface(x, y, z)
        XI, YI, ZI = make_surface(x, y, coeffs)
        trials.append(
            {
                "path": path,
                "label": label,
                "color": color,
                "time": time,
                "x": x,
                "y": y,
                "z": z,
                "r2": r2,
                "XI": XI,
                "YI": YI,
                "ZI": ZI,
            }
        )

    all_x = np.concatenate([trial["x"] for trial in trials])
    all_y = np.concatenate([trial["y"] for trial in trials])
    all_z = np.concatenate([trial["z"] for trial in trials])
    all_surface_z = np.concatenate(
        [trial["ZI"][np.isfinite(trial["ZI"])] for trial in trials]
    )

    xlim = padded_limits(all_x)
    ylim = padded_limits(all_y, lower_bound=0)
    zlim = padded_limits(np.concatenate([all_z, all_surface_z]))

    fig = plt.figure(figsize=(12.5, 10.0))
    ax = fig.add_subplot(111, projection="3d")

    for trial in trials:
        ax.plot_surface(
            trial["XI"],
            trial["YI"],
            trial["ZI"],
            color=trial["color"],
            alpha=0.24,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        ax.scatter(
            trial["x"],
            trial["y"],
            trial["z"],
            color=trial["color"],
            s=25,
            alpha=0.9,
            depthshade=False,
            edgecolors="white",
            linewidths=0.3,
        )

    # 先计算完所有曲面，再统一设置坐标范围，避免曲面或散点被裁剪。
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.view_init(elev=27, azim=-61)
    try:
        ax.set_box_aspect((1.45, 1.0, 0.95))
    except AttributeError:
        pass

    ax.set_xlabel("Object area (px²)", fontsize=13, labelpad=10)
    ax.set_ylabel("Hand velocity (m/s)", fontsize=13, labelpad=10)
    ax.set_zlabel("Finger aperture (px)", fontsize=13, labelpad=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(
        "Three grasp trials: overlaid quadratic surfaces",
        fontsize=16,
        pad=18,
    )

    legend_handles = []
    for trial in trials:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=trial["color"],
                markeredgecolor="white",
                markersize=8,
                label=f"{trial['label']}  (R²={trial['r2']:.3f})",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.96),
        frameon=True,
        fontsize=10,
        title="Trial / fit quality",
        title_fontsize=10,
    )

    fig.text(
        0.5,
        0.02,
        "Transparent colored surfaces are fitted separately; points are the original samples.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.06, top=0.90)
    fig.savefig(OUT_PATH, dpi=240, bbox_inches="tight", pad_inches=0.05)

    print(f"saved: {OUT_PATH}")
    print(f"x limits: {xlim}")
    print(f"y limits: {ylim}")
    print(f"z limits: {zlim}")
    for trial in trials:
        print(f"{trial['label']}: n={len(trial['x'])}, R²={trial['r2']:.4f}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
