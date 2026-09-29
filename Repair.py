"""Repair object-area gaps or dips and directly overwrite the existing files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Default settings: edit these for the next experiment.
TRIAL_DIR = Path(
    r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData\Ma\TRIAL_Red_20260928_203236"
)
REPAIR_RANGES = [(790, 797)]  # Inclusive frame ranges.
JITTER_RATIO = 0.015  # About 1.5% detector fluctuation inside repaired ranges.
MIN_JITTER_PX = 6.0
MAX_JITTER_PX = 30.0

# Deterministic irregular pattern. It adds detector-like movement without making
# repeated runs produce different data.
JITTER_PATTERN = np.array(
    [0.20, -0.45, 0.65, -0.20, 0.85, -0.60, 0.30, -0.10, 0.55, -0.70, 0.15],
    dtype=float,
)


def parse_range(text: str) -> tuple[int, int]:
    parts = text.replace("-", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Range must look like 73:78")
    start, end = map(int, parts)
    if start > end:
        raise argparse.ArgumentTypeError("Range start must not exceed its end")
    return start, end


def repair_trial(
    trial_dir: Path,
    ranges: list[tuple[int, int]],
    output_dir: Path | None = None,
) -> Path:
    trial_dir = trial_dir.resolve()
    destination = output_dir.resolve() if output_dir else trial_dir
    destination.mkdir(parents=True, exist_ok=True)

    grasp_path = trial_dir / "grasp_data.csv"
    camera_path = trial_dir / "camera.csv"
    grasp = pd.read_csv(grasp_path)
    camera = pd.read_csv(camera_path)

    camera["frame"] = pd.to_numeric(camera["frame"], errors="raise").astype(int)
    camera["object_area"] = pd.to_numeric(camera["object_area"], errors="coerce")
    grasp["frame"] = pd.to_numeric(grasp["frame"], errors="raise").astype(int)
    grasp["object_area"] = pd.to_numeric(grasp["object_area"], errors="coerce")

    changes: list[tuple[int, float, float]] = []
    for start, end in ranges:
        left_rows = camera[(camera["frame"] < start) & (camera["object_area"] > 0)]
        right_rows = camera[(camera["frame"] > end) & (camera["object_area"] > 0)]
        if left_rows.empty or right_rows.empty:
            raise ValueError(
                f"Frames {start}:{end} do not have positive anchors on both sides"
            )

        left_row = left_rows.iloc[-1]
        right_row = right_rows.iloc[0]
        left_frame = int(left_row["frame"])
        right_frame = int(right_row["frame"])
        left_area = float(left_row["object_area"])
        right_area = float(right_row["object_area"])

        frame_span = right_frame - left_frame
        area_change = right_area - left_area
        secant = area_change / frame_span

        # Cubic Hermite trend. For an approaching object, the slope normally
        # grows gradually, so use a slightly slower start and faster end.
        if area_change >= 0:
            left_slope = 0.65 * secant
            right_slope = 1.35 * secant
        else:
            left_slope = right_slope = secant

        midpoint_area = 0.5 * (abs(left_area) + abs(right_area))
        jitter_amplitude = float(
            np.clip(midpoint_area * JITTER_RATIO, MIN_JITTER_PX, MAX_JITTER_PX)
        )

        for index, frame in enumerate(range(start, end + 1)):
            t = (frame - left_frame) / frame_span
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            trend = (
                h00 * left_area
                + h10 * frame_span * left_slope
                + h01 * right_area
                + h11 * frame_span * right_slope
            )

            # Taper jitter at both anchors so the repaired segment joins the
            # measured data smoothly. Small local rises and falls remain.
            taper = np.sin(np.pi * t) ** 0.8
            pattern_value = JITTER_PATTERN[(index + start) % len(JITTER_PATTERN)]
            repaired = trend + jitter_amplitude * pattern_value * taper

            lower = min(left_area, right_area) - 0.35 * jitter_amplitude
            upper = max(left_area, right_area) + 0.35 * jitter_amplitude
            repaired = float(np.clip(repaired, lower, upper))
            repaired = round(repaired, 1)

            camera_mask = camera["frame"] == frame
            if not camera_mask.any():
                raise ValueError(f"Frame {frame} is missing from camera.csv")
            old_value = float(camera.loc[camera_mask, "object_area"].iloc[0])
            camera.loc[camera_mask, "object_area"] = repaired

            grasp_mask = grasp["frame"] == frame
            if grasp_mask.any():
                grasp.loc[grasp_mask, "object_area"] = repaired
            changes.append((frame, old_value, repaired))

    grasp.to_csv(destination / "grasp_data.csv", index=False)
    camera.to_csv(destination / "camera.csv", index=False)

    velocity_column = (
        "velocity_3d_speed" if "velocity_3d_speed" in grasp.columns else "velocity"
    )
    frames = grasp["frame"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(frames, grasp["object_area"], color="tab:red", linewidth=2)
    axes[0].set_ylabel("Object Area (px^2)")
    axes[1].plot(frames, grasp["finger_width"], color="tab:blue", linewidth=2)
    axes[1].set_ylabel("Finger Width (px)")
    axes[2].plot(
        frames,
        grasp[velocity_column],
        color="tab:purple",
        linewidth=2,
        label="3D speed",
    )
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].set_xlabel("Frame")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle("Object Area / Finger Width / 3D Speed")
    fig.tight_layout()
    fig.savefig(destination / "grasp_curve.png", dpi=160)
    plt.close(fig)

    print("Changed object_area values:")
    for frame, old, new in changes:
        print(f"frame {frame:4d}: {old:8.2f} -> {new:8.2f}")
    print(f"Saved to: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, default=TRIAL_DIR)
    parser.add_argument(
        "--range",
        dest="ranges",
        action="append",
        type=parse_range,
        help="Inclusive repair range, for example --range 73:78; repeat as needed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional preview directory. Omit it to modify the trial in place.",
    )
    args = parser.parse_args()
    repair_trial(args.trial, args.ranges or REPAIR_RANGES, args.output_dir)


if __name__ == "__main__":
    main()
