"""
sync_and_plot.py
把 IMU 加速度数据(acc_data_*.csv) 和 视频MediaPipe数据(*_mp_data.csv, 含 cube_area / finger_dist)
按时间对齐，输出:
  1. 合并后的CSV (每个视频帧对应的 area / aperture / 加速度)
  2. 时间序列对比图 (area, aperture, 加速度 三条曲线叠在一起看是否对得上)
  3. 三维关系图 (area, 加速度, aperture) 的散点 + 拟合曲面

用法:
    直接改下面 ─────── 配置 ─────── 区块里的三个值,然后 `python sync_and_plot.py` 运行即可。

示例(用尖峰/基线突变做同步标记时的典型流程):
    1. 先把 OFFSET 设成 None 跑一次,脚本会打印检测到的尖峰位置,不出图
    2. 结合视频里动作真正开始的那一帧(自己肉眼看/或看 finger_dist 什么时候开始变化),
       估算出 OFFSET(视频frame=0对应到acc时间轴的第几秒)
    3. 把 OFFSET 改成估算出来的数值,重新跑一次,生成正式结果
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (启用3D投影)


# ─────────────────────── 配置 ───────────────────────

ACC_CSV = r"acc_data_20260820_224100.csv"      # 加速度CSV路径
MP_CSV  = r"VID_20260820_224047_mp_data.csv"    # 视频MediaPipe CSV路径
FPS     = 30                                     # 视频帧率,固定为30
OFFSET  = 7                                   # 同步偏移量(秒); 设为 None 则只做尖峰检测提示,不出图
OUT_DIR = "sync_output"                          # 输出文件夹
# ─────────────────────────────────────────────────


# ─────────────────────── 读取数据 ───────────────────────

def load_acc(path):
    t, ax, ay, az, mag = [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["time_s"]))
            ax.append(float(row["ax_m/s2"]))
            ay.append(float(row["ay_m/s2"]))
            az.append(float(row["az_m/s2"]))
            mag.append(float(row["mag_m/s2"]))
    return (np.array(t), np.array(ax), np.array(ay),
            np.array(az), np.array(mag))


def load_mp(path):
    frame, area, dist = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame.append(int(row["frame"]))
            area.append(float(row["cube_area"]))
            dist.append(float(row["finger_dist"]))
    return np.array(frame), np.array(area), np.array(dist)


# ─────────────────────── 自动寻找尖峰(仅供参考) ───────────────────────

def suggest_spike(t_acc, mag_acc, threshold=3.0):
    """
    找到第一段 mag 超过 threshold 的区间，打印起止时间，供人工判断 offset 用。
    不做任何自动决定，只是辅助信息。
    """
    above = mag_acc > threshold
    if not above.any():
        print(f"[提示] 没有找到超过 {threshold} m/s^2 的尖峰，"
              f"如果这次同步没有做甩动标记，可以忽略这条提示。")
        return None

    idx = np.where(above)[0]
    # 找到第一段连续区间
    start = idx[0]
    end = start
    for i in idx[1:]:
        if i - end <= 3:   # 允许中间断几个点也算同一段
            end = i
        else:
            break

    print(f"[提示] 检测到疑似尖峰区间: t = {t_acc[start]:.3f}s ~ {t_acc[end]:.3f}s "
          f"(峰值 mag={mag_acc[start:end+1].max():.2f})")
    print(f"       如果这是你的同步标记(甩动/敲击),可以以此为参考手动估算 --offset")
    return t_acc[start], t_acc[end]


# ─────────────────────── 同步 + 合并 ───────────────────────

def sync_merge(t_acc, ax, ay, az, mag, frame, area, dist, fps, offset):
    """
    对每一个视频帧,计算它在acc时间轴上的时间点(t_video + offset),
    然后在acc数据上做线性插值,取出对应的加速度值。
    """
    t_video = frame / fps
    t_query = t_video + offset

    # 只保留落在acc数据时间范围内的帧,避免插值外推出离谱的值
    valid = (t_query >= t_acc.min()) & (t_query <= t_acc.max())
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"[提示] 有 {n_dropped} 帧的同步时间落在加速度数据范围之外,已丢弃"
              f"(通常是视频末尾,加速度程序关闭更早或更晚导致)")

    frame_v   = frame[valid]
    t_video_v = t_video[valid]
    area_v    = area[valid]
    dist_v    = dist[valid]
    tq        = t_query[valid]

    ax_i  = np.interp(tq, t_acc, ax)
    ay_i  = np.interp(tq, t_acc, ay)
    az_i  = np.interp(tq, t_acc, az)
    mag_i = np.interp(tq, t_acc, mag)

    return {
        "frame": frame_v, "t_video": t_video_v,
        "cube_area": area_v, "finger_dist": dist_v,
        "ax": ax_i, "ay": ay_i, "az": az_i, "mag": mag_i,
    }


def save_merged_csv(merged, path):
    keys = ["frame", "t_video", "cube_area", "finger_dist", "ax", "ay", "az", "mag"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        n = len(merged["frame"])
        for i in range(n):
            writer.writerow([merged[k][i] for k in keys])
    print(f"[已保存] 合并数据 → {path}")


# ─────────────────────── 画图: 时间序列对比 ───────────────────────

def plot_timeseries(merged, out_path):
    t = merged["t_video"]

    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axs[0].plot(t, merged["cube_area"], color="tab:red")
    axs[0].set_ylabel("Target Area (px^2)")
    axs[0].set_title("Synced time series (common timeline = video time)")
    axs[0].grid(alpha=0.3)

    axs[1].plot(t, merged["finger_dist"], color="tab:blue")
    axs[1].set_ylabel("Finger Aperture (px)")
    axs[1].grid(alpha=0.3)

    axs[2].plot(t, merged["mag"], color="tab:orange", label="|a| magnitude")
    axs[2].plot(t, merged["ax"], color="tab:green", alpha=0.6, linewidth=1, label="ax")
    axs[2].set_ylabel("Acceleration (m/s²)")
    axs[2].set_xlabel("Video time (s)")
    axs[2].legend(loc="upper right")
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[已保存] 时间序列对比图 → {out_path}")


# ─────────────────────── 画图: 三维关系 + 拟合曲面 ───────────────────────

def fit_quadratic_surface(x, y, z):
    """
    用二阶多项式拟合 z = f(x, y):
    z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2
    返回拟合系数和一个可以对任意(x,y)网格求值的函数。
    """
    A = np.column_stack([
        np.ones_like(x), x, y, x**2, x * y, y**2
    ])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)

    def surface_fn(xx, yy):
        return (coeffs[0] + coeffs[1] * xx + coeffs[2] * yy
                 + coeffs[3] * xx**2 + coeffs[4] * xx * yy + coeffs[5] * yy**2)

    # 拟合优度
    z_pred = surface_fn(x, y)
    ss_res = np.sum((z - z_pred) ** 2)
    ss_tot = np.sum((z - z.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return coeffs, surface_fn, r2


def plot_3d_surface(merged, out_path):
    x = merged["cube_area"]      # 目标面积
    y = merged["mag"]            # 加速度大小
    z = merged["finger_dist"]    # 手指开合(aperture)

    coeffs, surface_fn, r2 = fit_quadratic_surface(x, y, z)
    print(f"[拟合] 二阶曲面拟合 R² = {r2:.3f}")

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection="3d")

    # 散点: 按时间上色,方便看轨迹方向
    sc = ax3d.scatter(x, y, z, c=merged["t_video"], cmap="viridis", s=25)
    fig.colorbar(sc, ax=ax3d, shrink=0.6, label="time (s)")

    # 拟合曲面网格
    xi = np.linspace(x.min(), x.max(), 30)
    yi = np.linspace(y.min(), y.max(), 30)
    xg, yg = np.meshgrid(xi, yi)
    zg = surface_fn(xg, yg)
    ax3d.plot_surface(xg, yg, zg, alpha=0.25, color="gray")

    ax3d.set_xlabel("Target Area (px^2)")
    ax3d.set_ylabel("Acceleration |a| (m/s²)")
    ax3d.set_zlabel("Finger Aperture (px)")
    ax3d.set_title(f"Aperture vs Area vs Acceleration  (surface fit R²={r2:.2f})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[已保存] 三维关系图 → {out_path}")
    return coeffs, r2


# ─────────────────────── 主流程 ───────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t_acc, ax, ay, az, mag = load_acc(ACC_CSV)
    frame, area, dist = load_mp(MP_CSV)

    print(f"[数据] 加速度: {len(t_acc)} 点, 时长 {t_acc[-1]-t_acc[0]:.2f}s")
    print(f"[数据] 视频:   {len(frame)} 帧, 时长 {frame[-1]/FPS:.2f}s (按fps={FPS})")

    suggest_spike(t_acc, mag)

    if OFFSET is None:
        print("\n[OFFSET 为 None] 只做了尖峰检测提示,不会生成图表。")
        print("请结合上面的提示和你自己的同步分析,把 OFFSET 改成估算值后重新运行。")
        return

    merged = sync_merge(t_acc, ax, ay, az, mag, frame, area, dist, FPS, OFFSET)

    save_merged_csv(merged, os.path.join(OUT_DIR, "merged_data.csv"))
    plot_timeseries(merged, os.path.join(OUT_DIR, "timeseries.png"))
    plot_3d_surface(merged, os.path.join(OUT_DIR, "surface_3d.png"))


if __name__ == "__main__":
    main()