"""Generate grasp_data.csv with corrected three-axis speed.

Processing:
  1. Crop camera and IMU with the selected frame range.
  2. Low-pass filter ax, ay, az.
  3. Integrate all three axes with trapezoidal integration.
  4. Apply a start/end zero-velocity constraint to each velocity axis.
  5. Compute non-negative speed = sqrt(vx^2 + vy^2 + vz^2).

Outputs:
  grasp_data.csv (overwritten)
  grasp_curve.png
  velocity_3d_speed.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Edit these values for each experiment.
# ============================================================

TRIAL_DIR = Path(
    r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData"
    r"\TRIAL_Red_20260928_112151"
)
FRAME_START = 60
FRAME_END = 99

LPF_ALPHA = 0.15


def low_pass_filter(acceleration: np.ndarray, alpha: float) -> np.ndarray:
    filtered = np.zeros_like(acceleration, dtype=np.float64)
    state = np.zeros(3, dtype=np.float64)
    for index, sample in enumerate(acceleration):
        state = alpha * sample + (1.0 - alpha) * state
        filtered[index] = state
    return filtered


def cumulative_trapezoid_3d(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    integrated = np.zeros_like(values, dtype=np.float64)
    for index in range(1, len(values)):
        dt = timestamps[index] - timestamps[index - 1]
        if dt <= 0 or dt > 0.2:
            integrated[index] = integrated[index - 1]
            continue
        integrated[index] = (
            integrated[index - 1]
            + 0.5 * (values[index - 1] + values[index]) * dt
        )
    return integrated


def calculate_3d_speed(timestamps: np.ndarray, acceleration: np.ndarray):
    filtered_acceleration = low_pass_filter(acceleration, LPF_ALPHA)
    raw_velocity = cumulative_trapezoid_3d(filtered_acceleration, timestamps)

    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        raise ValueError("IMU timestamps must be increasing")

    # The selected action starts and ends stationary. Correct each velocity
    # axis using actual timestamps, then calculate the vector magnitude.
    fraction = (timestamps - timestamps[0]) / duration
    drift = fraction[:, None] * raw_velocity[-1]
    corrected_velocity = raw_velocity - drift
    speed = np.linalg.norm(corrected_velocity, axis=1)

    return filtered_acceleration, raw_velocity, corrected_velocity, speed


def main() -> None:
    parser = argparse.ArgumentParser(description="Test three-axis speed integration")
    parser.add_argument("--trial-dir", type=Path, default=TRIAL_DIR)
    parser.add_argument("--frame-start", type=int, default=FRAME_START)
    parser.add_argument("--frame-end", type=int, default=FRAME_END)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    camera_path = args.trial_dir / "camera.csv"
    imu_path = args.trial_dir / "imu.csv"
    output_dir = args.output_dir or args.trial_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = pd.read_csv(camera_path)
    imu = pd.read_csv(imu_path)

    camera = camera.loc[
        (camera["frame"] >= args.frame_start)
        & (camera["frame"] <= args.frame_end)
    ].copy()
    if len(camera) < 2:
        raise ValueError("Selected camera frame range has fewer than two rows")

    camera_time = camera["t_camera"].to_numpy(dtype=np.float64)
    crop_start = camera_time[0]
    crop_end = camera_time[-1]

    imu = imu.loc[
        (imu["t_imu"] >= crop_start)
        & (imu["t_imu"] <= crop_end)
    ].copy()
    if len(imu) < 2:
        raise ValueError("Selected time range has fewer than two IMU samples")

    imu_time = imu["t_imu"].to_numpy(dtype=np.float64)
    acceleration = imu[["ax", "ay", "az"]].to_numpy(dtype=np.float64)

    filtered_acc, raw_velocity, velocity_xyz, speed = calculate_3d_speed(
        imu_time, acceleration
    )

    result = camera.copy()
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        result[f"a{axis_name}"] = np.interp(
            camera_time, imu_time, acceleration[:, axis_index]
        )
        result[f"a{axis_name}_filtered"] = np.interp(
            camera_time, imu_time, filtered_acc[:, axis_index]
        )
        result[f"v{axis_name}"] = np.interp(
            camera_time, imu_time, velocity_xyz[:, axis_index]
        )

    if "mag" in imu.columns:
        result["mag"] = np.interp(
            camera_time, imu_time, imu["mag"].to_numpy(dtype=np.float64)
        )
    result["velocity"] = np.interp(camera_time, imu_time, speed)

    # Keep the exact columns expected by the existing training scripts.
    output_csv = output_dir / "grasp_data.csv"
    model_columns = [
        "frame",
        "t_camera",
        "finger_width",
        "object_area",
        "ax",
        "ay",
        "az",
        "mag",
        "velocity",
    ]
    result[model_columns].to_csv(output_csv, index=False)

    frames = result["frame"].to_numpy()

    # Main grasp curve: use the newly calculated three-axis speed.
    grasp_figure, grasp_axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    grasp_axes[0].plot(
        frames,
        result["object_area"],
        color="tab:red",
        linewidth=1.8,
    )
    grasp_axes[0].set_ylabel("Object Area (px^2)")
    grasp_axes[0].set_title("Object Area / Finger Width / 3D Speed")
    grasp_axes[0].grid(alpha=0.3)

    grasp_axes[1].plot(
        frames,
        result["finger_width"],
        color="tab:blue",
        linewidth=1.8,
    )
    grasp_axes[1].set_ylabel("Finger Width (px)")
    grasp_axes[1].grid(alpha=0.3)

    grasp_axes[2].plot(
        frames,
        result["velocity"],
        color="tab:purple",
        linewidth=2,
        label="3D speed",
    )
    grasp_axes[2].axhline(0, color="black", linewidth=0.7)
    grasp_axes[2].set_xlabel("Frame")
    grasp_axes[2].set_ylabel("Speed (m/s)")
    grasp_axes[2].grid(alpha=0.3)
    grasp_axes[2].legend()

    grasp_figure.tight_layout()
    grasp_curve_path = output_dir / "grasp_curve.png"
    grasp_figure.savefig(grasp_curve_path, dpi=160)
    plt.close(grasp_figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(frames, result["ax_filtered"], label="ax")
    axes[0].plot(frames, result["ay_filtered"], label="ay")
    axes[0].plot(frames, result["az_filtered"], label="az")
    axes[0].set_ylabel("Filtered acceleration (m/s^2)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        frames,
        result["velocity"],
        color="tab:purple",
        linewidth=2,
        label="3D speed",
    )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    output_plot = output_dir / "velocity_3d_speed.png"
    figure.savefig(output_plot, dpi=160)
    plt.close(figure)

    peak_index = int(np.argmax(result["velocity"].to_numpy()))
    print(f"Frames: {int(frames[0])} - {int(frames[-1])}")
    print(f"Start speed: {result['velocity'].iloc[0]:.6f} m/s")
    print(
        f"Peak speed: {result['velocity'].iloc[peak_index]:.6f} m/s "
        f"at frame {int(result['frame'].iloc[peak_index])}"
    )
    print(f"End speed: {result['velocity'].iloc[-1]:.6f} m/s")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved grasp curve: {grasp_curve_path}")
    print(f"Saved plot: {output_plot}")


if __name__ == "__main__":
    main()