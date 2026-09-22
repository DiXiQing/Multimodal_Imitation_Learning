"""
analyze_grasp_frames.py

功能：
1. 从 frames/ 读取 Camera JPG
2. 从 camera.csv 读取 frame + t_camera
3. 从 camera.csv 直接读取录制阶段得到的 Finger Width
4. 从 camera.csv 直接读取录制阶段得到的 Object Area
5. 不再重复进行 MediaPipe / 黑色物体检测
6. 从 imu.csv 读取加速度,按时间戳对齐到每一帧,再做积分+去漂移求速度
7. 生成带标注的视频
8. 生成 grasp_data.csv (含同步后的 ax/ay/az/mag/velocity)
9. 生成 Area + Finger Width + Velocity 曲线
10. 生成 Object Area / Velocity / Finger Width 三维曲面图

当前数据结构：

TRIAL_NAME/
├── frames/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── camera.csv   (frame, t_camera)
└── imu.csv      (imu_index, t_imu, ax, ay, az, mag)

同步假设(重要,如果对不上要检查):
    t_camera 和 t_imu 假设是同一个绝对时钟(比如都是 time.time() 打出来的),
    所以直接按时间戳数值做线性插值对齐,不需要手动调 OFFSET。
    如果实际上 t_camera 是从0开始的相对秒数,这里的对齐会错位,需要告诉我改成
    "t_camera + 录制起始的epoch时间"这种转换方式。

积分去漂移的假设(和 velocity_detrend.py 一致,继承过来的前提):
    假设这整段 trial 数据的开头和结尾,手都是(接近)静止的,
    用首尾两点连线做漂移修正。如果这份 imu.csv 其实是"多个trial拼在一起"的连续记录,
    而不是单次trial掐头去尾的数据,这个假设会不成立,算出来的velocity不可信,
    需要先按trial切割再分别积分。
"""

import os
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (启用3D投影)


# ============================================================
# 配置
# ============================================================

TRIAL_DIR = (
    r"Data\BackupData\TRIAL_20260919_133940"
)

FRAME_DIR = os.path.join(TRIAL_DIR, "frames")
CAMERA_CSV = os.path.join(TRIAL_DIR, "camera.csv")
IMU_CSV = os.path.join(TRIAL_DIR, "imu.csv")

# 用IMU哪个轴做积分求速度 (只能用分轴的ax/ay/az,不能用mag——mag恒正,积分没有意义)
ACCEL_AXIS = "ax"


# ============================================================
# Camera FPS
# ============================================================

TARGET_FPS = 30.0

# ============================================================
# Manual Frame Crop
# ============================================================

FRAME_START = 120
FRAME_END = 170


# ============================================================
# Load Camera CSV
# ============================================================

def load_camera_csv():

    if not os.path.exists(CAMERA_CSV):
        print(f"[Error] 找不到 camera.csv:")
        print(CAMERA_CSV)
        return None

    camera_data = []

    with open(CAMERA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_idx = int(row["frame"])
                t_camera = float(row["t_camera"])
                finger_width = float(row["finger_width"])
                object_area = float(row["object_area"])
                camera_data.append({"frame": frame_idx, "t_camera": t_camera, "finger_width": finger_width, "object_area": object_area})
            except Exception as e:
                print(f"[Warning] camera.csv 数据读取失败: {row}")
                print(e)

    return camera_data


# ============================================================
# Load IMU CSV
# ============================================================

def load_imu_csv():
    """
    读取 imu.csv (imu_index, t_imu, ax, ay, az, mag),
    返回按 t_imu 排好序的 numpy 数组,方便后面做插值。
    """

    if not os.path.exists(IMU_CSV):
        print(f"[Error] 找不到 imu.csv:")
        print(IMU_CSV)
        return None

    t_list, ax_list, ay_list, az_list, mag_list = [], [], [], [], []

    with open(IMU_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t_list.append(float(row["t_imu"]))
                ax_list.append(float(row["ax"]))
                ay_list.append(float(row["ay"]))
                az_list.append(float(row["az"]))
                mag_list.append(float(row["mag"]))
            except Exception as e:
                print(f"[Warning] imu.csv 数据读取失败: {row}")
                print(e)

    order = np.argsort(t_list)
    return {
        "t": np.array(t_list)[order],
        "ax": np.array(ax_list)[order],
        "ay": np.array(ay_list)[order],
        "az": np.array(az_list)[order],
        "mag": np.array(mag_list)[order],
    }


# ============================================================
# 同步: 把IMU数据按时间戳插值对齐到每一帧camera的时间
# ============================================================

def sync_imu_to_camera(imu_data, t_camera_array):
    """
    对每一个camera帧的时间戳 t_camera,在IMU时间轴上线性插值,
    取出对应的 ax/ay/az/mag。假设两者共用同一个绝对时钟。
    """

    t_imu = imu_data["t"]

    out_of_range = (
        (t_camera_array < t_imu.min()) | (t_camera_array > t_imu.max())
    ).sum()
    if out_of_range:
        print(f"[提示] 有 {out_of_range} 帧的时间戳落在IMU数据范围之外,"
              f"会用边界值外推,可能不准确")

    ax_i = np.interp(t_camera_array, t_imu, imu_data["ax"])
    ay_i = np.interp(t_camera_array, t_imu, imu_data["ay"])
    az_i = np.interp(t_camera_array, t_imu, imu_data["az"])
    mag_i = np.interp(t_camera_array, t_imu, imu_data["mag"])

    return {"ax": ax_i, "ay": ay_i, "az": az_i, "mag": mag_i}


# ============================================================
# 积分 + 去漂移 (和 velocity_detrend.py 完全一致的逻辑)
# ============================================================

def cumulative_trapezoid(y, x, initial=0.0):
    dx = np.diff(x)
    avg = (y[1:] + y[:-1]) / 2.0
    increments = avg * dx
    return np.concatenate([[initial], initial + np.cumsum(increments)])


def integrate_and_detrend(t, a):
    """
    加速度 -> 原始速度(带漂移) -> 首尾对齐0的去漂移速度。
    前提: 这段t,a对应的时间范围内,开头和结尾手都是(接近)静止的。
    """
    v_raw = cumulative_trapezoid(a, t, initial=0.0)
    drift_line = np.linspace(v_raw[0], v_raw[-1], len(v_raw))
    v_detrend = v_raw - drift_line
    return v_raw, v_detrend, drift_line


# ============================================================
# 三维曲面拟合 (和 sync_and_plot.py / velocity_detrend.py 一致)
# ============================================================

def fit_quadratic_surface(x, y, z):
    A = np.column_stack([np.ones_like(x), x, y, x**2, x * y, y**2])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)

    def surface_fn(xx, yy):
        return (coeffs[0] + coeffs[1] * xx + coeffs[2] * yy
                 + coeffs[3] * xx**2 + coeffs[4] * xx * yy + coeffs[5] * yy**2)

    z_pred = surface_fn(x, y)
    ss_res = np.sum((z - z_pred) ** 2)
    ss_tot = np.sum((z - z.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coeffs, surface_fn, r2


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("Reach-to-Grasp Frame Post-Processing")
    print("=" * 70)
    print()
    print(f"Trial directory:")
    print(TRIAL_DIR)
    print()
    print(f"Frames:")
    print(FRAME_DIR)
    print()

    # ========================================================
    # Check folders
    # ========================================================

    if not os.path.exists(FRAME_DIR):
        print("[Error] 找不到 frames 文件夹")
        return

    # ========================================================
    # Load camera CSV
    # ========================================================

    camera_data = load_camera_csv()
    if camera_data is None or len(camera_data) == 0:
        print("[Error] camera.csv 没有数据")
        return

    print(f"[Camera CSV] {len(camera_data)} frames")

    # ========================================================
    # Manual crop by frame
    # ========================================================

    camera_data = [row for row in camera_data if FRAME_START <= row["frame"] <= FRAME_END]

    if len(camera_data) == 0:
        print(f"[Error] Frame {FRAME_START} ~ {FRAME_END} 范围内没有 Camera 数据")
        return

    print(f"[Crop] Frame {FRAME_START} ~ {FRAME_END}")
    print(f"[Crop] 保留 Camera {len(camera_data)} frames")

    crop_start_time = camera_data[0]["t_camera"]
    crop_end_time = camera_data[-1]["t_camera"]

    # ========================================================
    # Load IMU CSV + crop by camera timestamps
    # ========================================================

    imu_data = load_imu_csv()
    has_imu = imu_data is not None and len(imu_data["t"]) > 0

    if has_imu:
        print(f"[IMU CSV] {len(imu_data['t'])} 点, 时长 {imu_data['t'][-1] - imu_data['t'][0]:.2f}s")

        imu_mask = (imu_data["t"] >= crop_start_time) & (imu_data["t"] <= crop_end_time)
        for key in ["t", "ax", "ay", "az", "mag"]:
            imu_data[key] = imu_data[key][imu_mask]

        if len(imu_data["t"]) < 2:
            print("[Error] 裁剪后 IMU 数据不足")
            return

        print(f"[Crop] 保留 IMU {len(imu_data['t'])} samples")

        t_camera_array = np.array([row["t_camera"] for row in camera_data])
        synced_imu = sync_imu_to_camera(imu_data, t_camera_array)

        v_raw, v_detrend, _ = integrate_and_detrend(imu_data["t"], imu_data[ACCEL_AXIS])
        velocity_per_frame = np.interp(t_camera_array, imu_data["t"], v_detrend)

        print(f"[积分] 用 {ACCEL_AXIS} 轴积分, 去漂移后速度范围: {v_detrend.min():.3f} ~ {v_detrend.max():.3f} m/s")
    else:
        print("[提示] 没有找到 imu.csv 或数据为空，跳过速度计算")
        synced_imu = None
        velocity_per_frame = None

    # ========================================================
    # Find first frame
    # ========================================================

    first_frame_idx = camera_data[0]["frame"]
    first_frame_path = os.path.join(FRAME_DIR, f"frame_{first_frame_idx:06d}.jpg")
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        print("[Error] 无法读取第一帧:")
        print(first_frame_path)
        return

    h, w = first_frame.shape[:2]
    print(f"[Frame] {w} x {h}")

    # ========================================================
    # Output paths
    # ========================================================

    output_video = os.path.join(TRIAL_DIR, "analyzed_video.mp4")
    output_csv = os.path.join(TRIAL_DIR, "grasp_data.csv")
    output_plot = os.path.join(TRIAL_DIR, "grasp_curve.png")
    output_scatter = os.path.join(TRIAL_DIR, "area_vs_finger_width.png")
    output_3d = os.path.join(TRIAL_DIR, "area_velocity_width_3d.png")
    output_accel = os.path.join(TRIAL_DIR, "acceleration_curve.png")

    # ========================================================
    # Video Writer
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, TARGET_FPS, (w, h))

    if not video_writer.isOpened():
        print("[Error] 无法创建输出视频")
        return

    # ========================================================
    # Result data
    # ========================================================

    result_data = []
    valid_frames = []
    areas = []
    finger_widths = []
    velocities = []
    finger_detected_count = 0
    object_detected_count = 0

    total_frames = len(camera_data)

    print()
    print("=" * 70)
    print("开始后处理")
    print("=" * 70)
    print()

    for process_idx, row in enumerate(camera_data):

        frame_idx = row["frame"]
        t_camera = row["t_camera"]

        frame_path = os.path.join(FRAME_DIR, f"frame_{frame_idx:06d}.jpg")
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"[Warning] 无法读取: frame_{frame_idx:06d}.jpg")
            continue

        vis = frame.copy()

        # 直接使用录制阶段已经保存到 camera.csv 的视觉检测结果
        finger_width = row["finger_width"]
        area = row["object_area"]

        if finger_width > 0:
            finger_detected_count += 1
        if area > 0:
            object_detected_count += 1

        # ── Text overlay ──
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        finger_text = f"Finger width: {finger_width:.1f} px" if finger_width > 0 else "Finger width: N/A"
        cv2.putText(vis, finger_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        area_text = f"Object area: {area:.0f} px^2" if area > 0 else "Object area: N/A"
        cv2.putText(vis, area_text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        velocity_val = float(velocity_per_frame[process_idx]) if has_imu else None
        ax_val = float(synced_imu["ax"][process_idx]) if has_imu else None
        ay_val = float(synced_imu["ay"][process_idx]) if has_imu else None
        az_val = float(synced_imu["az"][process_idx]) if has_imu else None
        mag_val = float(synced_imu["mag"][process_idx]) if has_imu else None
        if has_imu:
            cv2.putText(vis, f"Velocity: {velocity_val:.3f} m/s", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        video_writer.write(vis)

        # ── Save data ──
        row_data = {
            "frame": frame_idx,
            "t_camera": t_camera,
            "finger_width": finger_width,
            "object_area": area,
            "ax": ax_val,
            "ay": ay_val,
            "az": az_val,
            "mag": mag_val,
            "velocity": velocity_val,
        }
        result_data.append(row_data)

        if finger_width > 0 and area > 0 and (not has_imu or velocity_val is not None):
            valid_frames.append(frame_idx)
            areas.append(area)
            finger_widths.append(finger_width)
            if has_imu:
                velocities.append(velocity_val)

        if process_idx % 30 == 0 or process_idx == total_frames - 1:
            progress = (process_idx + 1) / total_frames * 100
            print(f"\rProcessing: {process_idx + 1}/{total_frames} ({progress:.1f}%)", end="")

    print()
    print()

    video_writer.release()

    # ========================================================
    # Save CSV
    # ========================================================

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t_camera", "finger_width", "object_area", "ax", "ay", "az", "mag", "velocity"])
        for row in result_data:
            writer.writerow([row["frame"], f'{row["t_camera"]:.9f}', f'{row["finger_width"]:.3f}', f'{row["object_area"]:.3f}',
                             f'{row["ax"]:.6f}' if row["ax"] is not None else "", f'{row["ay"]:.6f}' if row["ay"] is not None else "",
                             f'{row["az"]:.6f}' if row["az"] is not None else "", f'{row["mag"]:.6f}' if row["mag"] is not None else "",
                             f'{row["velocity"]:.6f}' if row["velocity"] is not None else ""])

    # ========================================================
    # Print statistics
    # ========================================================

    print()
    print("=" * 70)
    print("处理完成")
    print("=" * 70)
    print()
    print(f"总 Camera frames       : {total_frames}")
    print(f"检测到手指标记的 frames  : {finger_detected_count}")
    print(f"检测到物体的 frames    : {object_detected_count}")
    print(f"手 + 物体 (+速度) 同时检测 : {len(valid_frames)}")

    if total_frames > 0:
        print()
        print(f"Finger detection rate  : {finger_detected_count / total_frames * 100:.1f}%")
        print(f"Object detection rate  : {object_detected_count / total_frames * 100:.1f}%")
        print(f"Both detection rate    : {len(valid_frames) / total_frames * 100:.1f}%")

    # ========================================================
    # Save curve (Area / Finger Width / Velocity)
    # ========================================================

    if len(valid_frames) >= 2:

        frames_np = np.array([row["frame"] for row in result_data])

        areas_time = np.array([
            row["object_area"] if row["object_area"] > 0 else np.nan
            for row in result_data
        ])

        finger_widths_time = np.array([
            row["finger_width"] if row["finger_width"] > 0 else np.nan
            for row in result_data
        ])

        n_rows = 3 if has_imu else 2
        fig, axs = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=True)

        axs[0].plot(frames_np, areas_time, "r-", linewidth=1.5)
        axs[0].set_ylabel("Object Area (px²)")
        axs[0].set_title("Object Area / Finger Width" + (" / Velocity" if has_imu else "") + " over Time")
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(frames_np, finger_widths_time, "b-", linewidth=1.5)
        axs[1].set_ylabel("Finger Width (px)")
        axs[1].grid(True, alpha=0.3)

        if has_imu:
            velocity_time = np.array([row["velocity"] for row in result_data])
            axs[2].plot(frames_np, velocity_time, "-", linewidth=1.5)
            axs[2].axhline(0, color="black", linewidth=0.5)
            axs[2].set_ylabel("Velocity (m/s)")
            axs[2].set_xlabel("Frame")
            axs[2].grid(True, alpha=0.3)
        else:
            axs[1].set_xlabel("Frame")

        fig.tight_layout()
        plt.savefig(output_plot, dpi=150)
        plt.close(fig)

        valid_frames_np = np.array(valid_frames)
        areas_np = np.array(areas)
        finger_widths_np = np.array(finger_widths)
        
        print()
        print(f"[Saved] 曲线:")
        print(output_plot)
        print()
        print(f"[Saved] Area vs Finger Width:")
        print(output_scatter)

        # ── 3D: Area, Velocity, Finger Width + 拟合曲面 ──
        if has_imu:
            velocities_np = np.array(velocities)
            coeffs, surface_fn, r2 = fit_quadratic_surface(areas_np, velocities_np, finger_widths_np)
            print(f"[拟合] Area-Velocity-Width 曲面 R² = {r2:.3f}")

            fig3 = plt.figure(figsize=(10, 8))
            ax3d = fig3.add_subplot(111, projection="3d")
            sc = ax3d.scatter(areas_np, velocities_np, finger_widths_np,
                               c=valid_frames_np, cmap="viridis", s=25)
            fig3.colorbar(sc, ax=ax3d, shrink=0.6, label="Frame")

            xi = np.linspace(areas_np.min(), areas_np.max(), 30)
            yi = np.linspace(velocities_np.min(), velocities_np.max(), 30)
            xg, yg = np.meshgrid(xi, yi)
            zg = surface_fn(xg, yg)
            ax3d.plot_surface(xg, yg, zg, alpha=0.25, color="gray")

            ax3d.set_xlabel("Object Area (px²)")
            ax3d.set_ylabel("Velocity (m/s)")
            ax3d.set_zlabel("Finger Width (px)")
            ax3d.set_title(f"Finger Width vs Area vs Velocity (surface fit R²={r2:.2f})")

            fig3.tight_layout()
            plt.savefig(output_3d, dpi=150)
            plt.close(fig3)

            print()
            print(f"[Saved] 三维关系图:")
            print(output_3d)

    else:
        print()
        print("[Warning] 同时检测到手、物体(+速度)的有效帧太少")
        print("无法生成可靠的曲线。")

    # ========================================================
    # Save acceleration curve (ax / ay / az / magnitude)
    # ========================================================

    if has_imu:
        frames_acc = np.array([row["frame"] for row in result_data])
        ax_time = np.array([row["ax"] for row in result_data])
        ay_time = np.array([row["ay"] for row in result_data])
        az_time = np.array([row["az"] for row in result_data])
        mag_time = np.array([row["mag"] for row in result_data])

        fig_acc, axs_acc = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        axs_acc[0].plot(frames_acc, ax_time, linewidth=1.5)
        axs_acc[0].set_ylabel("Ax (m/s²)")
        axs_acc[0].grid(True, alpha=0.3)

        axs_acc[1].plot(frames_acc, ay_time, linewidth=1.5)
        axs_acc[1].set_ylabel("Ay (m/s²)")
        axs_acc[1].grid(True, alpha=0.3)

        axs_acc[2].plot(frames_acc, az_time, linewidth=1.5)
        axs_acc[2].set_ylabel("Az (m/s²)")
        axs_acc[2].grid(True, alpha=0.3)

        axs_acc[3].plot(frames_acc, mag_time, linewidth=1.5)
        axs_acc[3].set_ylabel("Magnitude (m/s²)")
        axs_acc[3].set_xlabel("Frame")
        axs_acc[3].grid(True, alpha=0.3)

        fig_acc.suptitle("3-Axis Acceleration and Magnitude")
        fig_acc.tight_layout()
        plt.savefig(output_accel, dpi=150)
        plt.close(fig_acc)

        print()
        print(f"[Saved] 加速度曲线:")
        print(output_accel)

    # ========================================================
    # Final output
    # ========================================================

    print()
    print(f"[Saved] 视频:")
    print(output_video)
    print()
    print(f"[Saved] 数据:")
    print(output_csv)
    print()
    print("=" * 70)
    print("全部完成")
    print("=" * 70)


if __name__ == "__main__":
    main()