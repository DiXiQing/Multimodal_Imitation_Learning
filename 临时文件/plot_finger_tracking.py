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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (激活 3D 投影)


# ==== 配置：在这里改文件名 ====
CSV_PATH = r"D:\Code\Multimodal_Imitation_Learning\临时文件\grasp_smoothed.csv"

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

def plot_time_series(
    df: pd.DataFrame,
    title: str = "Area / Width / Velocity over time",
    save_path: str | None = None,
):
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6), sharex=True)

    axes[0].plot(df["time_s"], df["target_area_px2"], color="crimson", linewidth=1.6)
    axes[0].set_ylabel("Target Area (px^2)")

    axes[1].plot(df["time_s"], df["finger_width_px"], color="#1f77a4", linewidth=1.6)
    axes[1].set_ylabel("Finger Width (px)")

    axes[2].plot(df["time_s"], df["velocity_mps"], color="#7b52ab", linewidth=1.6)
    axes[2].set_ylabel("Velocity (m/s)")
    axes[2].set_xlabel("time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=150)
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
    df: pd.DataFrame,
    title: str = "Aperture vs Area vs Acceleration",
    save_path: str | None = None,
):
    x = df["target_area_px2"].to_numpy()
    y = df["velocity_mps"].to_numpy()
    z = df["finger_width_px"].to_numpy()
    t = df["time_s"].to_numpy()

    coeffs, r2 = fit_quadratic_surface(x, y, z)

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

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(XI, YI, ZI, color="gray", alpha=0.35, linewidth=0, antialiased=True)
    sc = ax.scatter(x, y, z, c=t, cmap="viridis", s=25, depthshade=True)

    ax.set_xlabel("Object Size (px^2)")
    ax.set_ylabel("Velocity (mm/s)")
    ax.set_zlabel("Finger Aperture (px)")
    # ax.set_title(f"{title}  (surface fit R\u00b2={r2:.2f})")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("time (s)")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"saved: {save_path}")
    return fig, r2


# ---------- 主入口 ----------

def main():
    df = load_data(CSV_PATH)

    plot_time_series(df, save_path=f"{OUT_PREFIX}_timeseries.png")
    _, r2 = plot_3d_surface(df, save_path=f"{OUT_PREFIX}_3d_surface.png")
    print(f"surface fit R^2 = {r2:.3f}")

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()