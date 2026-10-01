from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALIDATION_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\Validation\D")
TRAINING_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\Training")
OUTPUT_DIR = Path(__file__).resolve().parent / "imu_velocity_drift"
BLACK_TRIALS = [
    "TRIAL_Black_20260929_102317",
    "TRIAL_Black_20260929_102345",
    "TRIAL_Black_20260929_102412",
]
LPF_ALPHA = 0.15


def reconstruct_raw_integrated_speed(imu: pd.DataFrame) -> np.ndarray:
    timestamps = imu["t_imu"].to_numpy(dtype=float)
    acceleration = imu[["ax", "ay", "az"]].to_numpy(dtype=float)
    filtered = np.zeros_like(acceleration, dtype=float)
    state = np.zeros(3, dtype=float)
    for index, sample in enumerate(acceleration):
        state = LPF_ALPHA * sample + (1.0 - LPF_ALPHA) * state
        filtered[index] = state

    velocity = np.zeros_like(filtered, dtype=float)
    for index in range(1, len(filtered)):
        dt = timestamps[index] - timestamps[index - 1]
        if dt <= 0 or dt > 0.2:
            velocity[index] = velocity[index - 1]
            continue
        velocity[index] = velocity[index - 1] + 0.5 * (
            filtered[index - 1] + filtered[index]
        ) * dt
    return np.linalg.norm(velocity, axis=1)


def load_trial(trial_dir: Path, display_name: str | None = None, source: str = "Validation"):
    camera = pd.read_csv(trial_dir / "camera.csv")
    imu = pd.read_csv(trial_dir / "imu.csv")
    grasp = pd.read_csv(trial_dir / "grasp_data.csv")

    imu_time = imu["t_imu"].to_numpy(dtype=float)
    if "velocity" in imu.columns:
        realtime_velocity = imu["velocity"].to_numpy(dtype=float)
        realtime_method = "recorded"
    else:
        realtime_velocity = reconstruct_raw_integrated_speed(imu)
        realtime_method = "reconstructed"
    camera_time = camera["t_camera"].to_numpy(dtype=float)
    camera_frames = camera["frame"].to_numpy(dtype=int)
    realtime_at_camera = np.interp(camera_time, imu_time, realtime_velocity)

    grasp_time = grasp["t_camera"].to_numpy(dtype=float)
    grasp_frames = grasp["frame"].to_numpy(dtype=int)
    realtime_at_grasp = np.interp(grasp_time, imu_time, realtime_velocity)
    offline_velocity = grasp["velocity"].to_numpy(dtype=float)

    return {
        "trial": display_name or trial_dir.name,
        "source": source,
        "realtime_method": realtime_method,
        "camera_frames": camera_frames,
        "realtime_at_camera": realtime_at_camera,
        "grasp_frames": grasp_frames,
        "realtime_at_grasp": realtime_at_grasp,
        "offline_velocity": offline_velocity,
        "realtime_start": float(realtime_velocity[0]),
        "realtime_end": float(realtime_velocity[-1]),
        "realtime_peak": float(np.max(realtime_velocity)),
        "offline_peak": float(np.max(offline_velocity)),
    }


def short_name(name: str) -> str:
    return name.replace("TRIAL_", "")


def plot_full_recordings(records):
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharey=True)
    axes = axes.ravel()
    global_max = max(record["realtime_peak"] for record in records) * 1.08

    for axis, record in zip(axes, records):
        axis.plot(
            record["camera_frames"],
            record["realtime_at_camera"],
            color="#d62728",
            linewidth=2.2,
            label="Uncorrected realtime-style integration",
        )
        axis.axvspan(
            record["grasp_frames"][0],
            record["grasp_frames"][-1],
            color="#4c78a8",
            alpha=0.14,
            label="Selected grasp interval",
        )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_ylim(-0.05, global_max)
        axis.set_title(
            f"{short_name(record['trial'])}\n"
            f"final = {record['realtime_end']:.2f} m/s",
            fontsize=11,
        )
        axis.set_xlabel("Frame")
        axis.set_ylabel("Uncorrected integrated speed (m/s)")
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=True,
    )
    fig.suptitle(
        "Realtime IMU velocity drift across nine representative recordings",
        fontsize=18,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUTPUT_DIR / "realtime_velocity_drift.png", dpi=180)
    plt.close(fig)


def plot_realtime_vs_offline(records):
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes = axes.ravel()

    for axis, record in zip(axes, records):
        corrected_axis = axis.twinx()

        realtime_line = axis.plot(
            record["grasp_frames"],
            record["realtime_at_grasp"],
            color="#d62728",
            linewidth=2.2,
            label="Uncorrected integration",
        )[0]
        corrected_line = corrected_axis.plot(
            record["grasp_frames"],
            record["offline_velocity"],
            color="#1f77b4",
            linewidth=2.2,
            label="Offline drift-corrected",
        )[0]

        axis.set_title(short_name(record["trial"]), fontsize=11)
        axis.set_xlabel("Frame")
        axis.set_ylabel("Realtime speed (m/s)", color="#d62728")
        corrected_axis.set_ylabel("Corrected speed (m/s)", color="#1f77b4")
        axis.tick_params(axis="y", colors="#d62728")
        corrected_axis.tick_params(axis="y", colors="#1f77b4")
        axis.grid(alpha=0.22)
        axis.legend(
            [realtime_line, corrected_line],
            ["Uncorrected integration", "Offline drift-corrected"],
            loc="upper left",
            fontsize=8,
        )

    fig.suptitle(
        "Uncorrected integration versus offline zero-velocity correction",
        fontsize=18,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_DIR / "realtime_vs_offline_velocity.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [
        load_trial(trial_dir)
        for trial_dir in sorted(VALIDATION_ROOT.glob("TRIAL_*"))
        if not trial_dir.name.startswith("TRIAL_Black_")
        if (trial_dir / "imu.csv").exists()
        and (trial_dir / "camera.csv").exists()
        and (trial_dir / "grasp_data.csv").exists()
    ]
    records.extend(
        load_trial(TRAINING_ROOT / trial, source="Training")
        for trial in BLACK_TRIALS
    )
    records.sort(key=lambda record: record["trial"])
    if not records:
        raise RuntimeError("No usable trials found")

    summary = pd.DataFrame(
        [
            {
                "trial": record["trial"],
                "source": record["source"],
                "realtime_method": record["realtime_method"],
                "realtime_start_m_s": record["realtime_start"],
                "realtime_end_m_s": record["realtime_end"],
                "realtime_peak_m_s": record["realtime_peak"],
                "offline_peak_m_s": record["offline_peak"],
            }
            for record in records
        ]
    )
    summary.to_csv(OUTPUT_DIR / "velocity_drift_summary.csv", index=False)
    plot_full_recordings(records)
    plot_realtime_vs_offline(records)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
