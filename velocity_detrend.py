"""
velocity_detrend.py
最简单版本的"加速度积分求速度 + 去漂移"。

思路:
    1. 对加速度做梯形积分,得到原始速度 v_raw(t) (会带漂移)
    2. 假设录制开头和结尾手都是静止的,即 v(t0)=0, v(tend)=0
    3. 用一条直线连接 v_raw(t0) 和 v_raw(tend),把这条"漂移线"从 v_raw 里减掉
       得到修正后的速度 v_detrend(t)

前提假设(重要,分析结果之前先确认这批数据符不符合):
    - 你的这段数据开头和结尾,手确实是(接近)静止的
    - 如果结尾手还在动(比如还没抓稳/还在回收),这一步的修正会不准,
      修正结果仅供参考,不要当成精确的速度值使用

用法:
    直接改下面 ─────── 配置 ─────── 区块里的路径和列名,然后运行。
    输入可以是:
        (a) sync_and_plot.py 产出的 merged_data.csv (列: t_video, ax, ay, az, mag)
        (b) 原始 acc_data_xxx.csv (列: time_s, ax_m/s2, ay_m/s2, az_m/s2, mag_m/s2)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (启用3D投影)


# ─────────────────────── 配置 ───────────────────────

INPUT_CSV   = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData\Middle\2\merged_data.csv"   # 输入文件路径
TIME_COL    = "t_video"            # 时间列名 (merged_data.csv用 t_video, 原始acc用 time_s)
ACCEL_COL   = "ax"                 # 用哪个轴做积分 (只能用分轴的ax/ay/az,不能用mag——mag恒正,积分没有意义)
OUT_CSV     = os.path.join(os.path.dirname(INPUT_CSV), "velocity_detrend.csv")
OUT_PATH    = os.path.join(os.path.dirname(INPUT_CSV), "velocity_detrend.png")
OUT_3D_PATH = os.path.join(os.path.dirname(INPUT_CSV), "surface_3d_velocity.png")
# ─────────────────────────────────────────────────


def load_column(path, time_col, accel_col):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    t = np.array([float(r[time_col]) for r in rows])
    a = np.array([float(r[accel_col]) for r in rows])
    # 也顺手把aperture/area读出来,如果有的话,方便画图对照(没有就跳过)
    extra = {}
    for k in ("finger_dist", "cube_area"):
        if k in rows[0]:
            extra[k] = np.array([float(r[k]) for r in rows])
    return t, a, extra


def cumulative_trapezoid(y, x, initial=0.0):
    """
    梯形积分,得到累积曲线。等价于 scipy.integrate.cumulative_trapezoid,
    这里手写一遍避免额外依赖 scipy。
    """
    dx = np.diff(x)
    avg = (y[1:] + y[:-1]) / 2.0
    increments = avg * dx
    return np.concatenate([[initial], initial + np.cumsum(increments)])


def integrate_and_detrend(t, a):
    """
    核心函数: 加速度 -> 原始速度(带漂移) -> 首尾对齐0的去漂移速度
    """
    v_raw = cumulative_trapezoid(a, t, initial=0.0)

    # 漂移线: 从 v_raw[0] 线性变化到 v_raw[-1]
    drift_line = np.linspace(v_raw[0], v_raw[-1], len(v_raw))
    v_detrend = v_raw - drift_line

    return v_raw, v_detrend, drift_line


def save_csv(path, t, a, v_raw, drift_line, v_detrend, extra):
    fieldnames = ["t", "accel", "v_raw", "drift_line", "v_detrend"] + list(extra.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(t)):
            row = [t[i], a[i], v_raw[i], drift_line[i], v_detrend[i]]
            row += [extra[k][i] for k in extra]
            writer.writerow(row)
    print(f"[已保存] 数值CSV → {path}")

def fit_quadratic_surface(x, y, z):
    A = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)

    def surface_fn(xx, yy):
        return (coeffs[0] + coeffs[1]*xx + coeffs[2]*yy
                + coeffs[3]*xx**2 + coeffs[4]*xx*yy + coeffs[5]*yy**2)

    z_pred = surface_fn(x, y)
    ss_res = np.sum((z - z_pred) ** 2)
    ss_tot = np.sum((z - z.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coeffs, surface_fn, r2


def plot_3d_surface_velocity(area, v_detrend, dist, t, out_path):
    coeffs, surface_fn, r2 = fit_quadratic_surface(area, v_detrend, dist)
    print(f"[拟合] area-velocity-aperture 曲面 R² = {r2:.3f}")

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    sc = ax3d.scatter(area, v_detrend, dist, c=t, cmap="viridis", s=25)
    fig.colorbar(sc, ax=ax3d, shrink=0.6, label="time (s)")

    xi = np.linspace(area.min(), area.max(), 30)
    yi = np.linspace(v_detrend.min(), v_detrend.max(), 30)
    xg, yg = np.meshgrid(xi, yi)
    zg = surface_fn(xg, yg)
    ax3d.plot_surface(xg, yg, zg, alpha=0.25, color="gray")

    ax3d.set_xlabel("Target Area (px^2)")
    ax3d.set_ylabel("Velocity v_detrend (m/s)")
    ax3d.set_zlabel("Finger Aperture (px)")
    ax3d.set_title(f"Aperture vs Area vs Velocity (surface fit R²={r2:.2f})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[已保存] 三维曲面图 → {out_path}")

def main():
    t, a, extra = load_column(INPUT_CSV, TIME_COL, ACCEL_COL)

    v_raw, v_detrend, drift_line = integrate_and_detrend(t, a)

    print(f"[数据] {len(t)} 点, 时长 {t[-1]-t[0]:.2f}s")
    print(f"[积分] 原始速度(未修正) 结尾值 v_raw[-1] = {v_raw[-1]:.3f} (这就是被当成'漂移'减掉的量)")
    print(f"[结果] 去漂移后速度范围: {v_detrend.min():.3f} ~ {v_detrend.max():.3f}")

    save_csv(OUT_CSV, t, a, v_raw, drift_line, v_detrend, extra)

    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axs[0].plot(t, extra.get("cube_area", []), color="tab:red")
    axs[0].set_ylabel("Target Area (px^2)")
    axs[0].set_title("Area / Aperture / Velocity over time")
    axs[0].grid(alpha=0.3)

    axs[1].plot(t, extra.get("finger_dist", []), color="tab:blue")
    axs[1].set_ylabel("Finger Aperture (px)")
    axs[1].grid(alpha=0.3)

    axs[2].plot(t, v_detrend, color="tab:purple", linewidth=2)
    axs[2].axhline(0, color="black", linewidth=0.5)
    axs[2].set_ylabel("Velocity v_detrend (m/s)")
    axs[2].set_xlabel("time (s)")
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    plt.close()
    print(f"[已保存] {OUT_PATH}")

    if "cube_area" in extra and "finger_dist" in extra:
        plot_3d_surface_velocity(extra["cube_area"], v_detrend, extra["finger_dist"], t, OUT_3D_PATH)
    else:
        print("[提示] 输入数据没有 cube_area/finger_dist 列,跳过三维曲面图(用merged_data.csv作为输入即可)")


if __name__ == "__main__":
    main()