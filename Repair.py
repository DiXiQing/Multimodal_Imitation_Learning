"""Repair object_area and/or finger_width with a selectable mode.

Modes:
    "object" -> repair only object_area
    "finger" -> repair only finger_width
    "both"   -> repair both

Edit the settings in the configuration block below.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRIAL_DIR = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData\D\TRIAL_Blue_20261007_141842")

# Select repair mode here: "object", "finger", or "both"
REPAIR_MODE = "finger"

# Object-area repair ranges: [(start_frame, end_frame), ...]
OBJECT_REPAIR_RANGES = [(115, 134)]

# Finger-width repair ranges: [(start_frame, end_frame, target_finger_width_px), ...]
# Example for first 15 stationary frames:
FINGER_REPAIR_RANGES = [(115, 134, 15.0)]

OBJECT_JITTER_RATIO = 0.015
OBJECT_MIN_JITTER_PX = 6.0
OBJECT_MAX_JITTER_PX = 30.0

FINGER_JITTER_PX = 2.0

JITTER_PATTERN = np.array([0.20, -0.45, 0.65, -0.20, 0.85, -0.60, 0.30, -0.10, 0.55, -0.70, 0.15], dtype=float)


def parse_range(text: str) -> tuple[int, int]:
    parts = text.replace("-", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Range must look like 73:78")
    start, end = map(int, parts)
    if start > end:
        raise argparse.ArgumentTypeError("Range start must not exceed its end")
    return start, end


def parse_finger_range(text: str) -> tuple[int, int, float]:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Finger range must look like 42:56:118")
    start = int(parts[0])
    end = int(parts[1])
    target = float(parts[2])
    if start > end:
        raise argparse.ArgumentTypeError("Range start must not exceed its end")
    return start, end, target


def make_backup(path: Path) -> None:
    backup = path.with_name(path.stem + "_before_repair" + path.suffix)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup: {backup}")


def repair_object_area(camera: pd.DataFrame, grasp: pd.DataFrame, ranges: list[tuple[int, int]]) -> list[tuple[int, float, float]]:
    changes = []

    for start, end in ranges:
        left_rows = camera[(camera["frame"] < start) & (camera["object_area"] > 0)]
        right_rows = camera[(camera["frame"] > end) & (camera["object_area"] > 0)]

        if left_rows.empty or right_rows.empty:
            raise ValueError(f"Frames {start}:{end} do not have positive object_area anchors on both sides")

        left_row = left_rows.iloc[-1]
        right_row = right_rows.iloc[0]
        left_frame = int(left_row["frame"])
        right_frame = int(right_row["frame"])
        left_area = float(left_row["object_area"])
        right_area = float(right_row["object_area"])

        frame_span = right_frame - left_frame
        area_change = right_area - left_area
        secant = area_change / frame_span

        if area_change >= 0:
            left_slope = 0.65 * secant
            right_slope = 1.35 * secant
        else:
            left_slope = right_slope = secant

        midpoint_area = 0.5 * (abs(left_area) + abs(right_area))
        jitter_amplitude = float(np.clip(midpoint_area * OBJECT_JITTER_RATIO, OBJECT_MIN_JITTER_PX, OBJECT_MAX_JITTER_PX))

        for index, frame in enumerate(range(start, end + 1)):
            t = (frame - left_frame) / frame_span
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2

            trend = h00 * left_area + h10 * frame_span * left_slope + h01 * right_area + h11 * frame_span * right_slope
            taper = np.sin(np.pi * t) ** 0.8
            pattern_value = JITTER_PATTERN[(index + start) % len(JITTER_PATTERN)]
            repaired = trend + jitter_amplitude * pattern_value * taper

            lower = min(left_area, right_area) - 0.35 * jitter_amplitude
            upper = max(left_area, right_area) + 0.35 * jitter_amplitude
            repaired = round(float(np.clip(repaired, lower, upper)), 1)

            camera_mask = camera["frame"] == frame
            if not camera_mask.any():
                raise ValueError(f"Frame {frame} is missing from camera.csv")

            old_value = float(camera.loc[camera_mask, "object_area"].iloc[0])
            camera.loc[camera_mask, "object_area"] = repaired

            grasp_mask = grasp["frame"] == frame
            if grasp_mask.any():
                grasp.loc[grasp_mask, "object_area"] = repaired

            changes.append((frame, old_value, repaired))

    return changes


def repair_finger_width(camera: pd.DataFrame, grasp: pd.DataFrame, ranges: list[tuple[int, int, float]]) -> list[tuple[int, float, float]]:
    changes = []

    if "finger_width" not in camera.columns and "finger_width" not in grasp.columns:
        raise ValueError("Neither camera.csv nor grasp_data.csv contains finger_width")

    for start, end, target in ranges:
        for index, frame in enumerate(range(start, end + 1)):
            pattern_value = JITTER_PATTERN[(index + start) % len(JITTER_PATTERN)]
            repaired = round(float(target + FINGER_JITTER_PX * pattern_value), 2)
            old_value = np.nan

            if "finger_width" in camera.columns:
                camera_mask = camera["frame"] == frame
                if not camera_mask.any():
                    raise ValueError(f"Frame {frame} is missing from camera.csv")
                old_value = float(camera.loc[camera_mask, "finger_width"].iloc[0])
                camera.loc[camera_mask, "finger_width"] = repaired

            if "finger_width" in grasp.columns:
                grasp_mask = grasp["frame"] == frame
                if grasp_mask.any():
                    if np.isnan(old_value):
                        old_value = float(grasp.loc[grasp_mask, "finger_width"].iloc[0])
                    grasp.loc[grasp_mask, "finger_width"] = repaired

            changes.append((frame, old_value, repaired))

    return changes


def repair_trial(trial_dir: Path, mode: str, object_ranges: list[tuple[int, int]], finger_ranges: list[tuple[int, int, float]], output_dir: Path | None = None) -> Path:
    trial_dir = trial_dir.resolve()
    destination = output_dir.resolve() if output_dir else trial_dir
    destination.mkdir(parents=True, exist_ok=True)

    grasp_path = trial_dir / "grasp_data.csv"
    camera_path = trial_dir / "camera.csv"

    grasp = pd.read_csv(grasp_path)
    camera = pd.read_csv(camera_path)

    camera["frame"] = pd.to_numeric(camera["frame"], errors="raise").astype(int)
    grasp["frame"] = pd.to_numeric(grasp["frame"], errors="raise").astype(int)

    for df in (camera, grasp):
        for col in ("object_area", "finger_width"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    object_changes = []
    finger_changes = []

    if mode in ("object", "both"):
        object_changes = repair_object_area(camera, grasp, object_ranges)

    if mode in ("finger", "both"):
        finger_changes = repair_finger_width(camera, grasp, finger_ranges)

    if output_dir is None:
        make_backup(grasp_path)
        make_backup(camera_path)

    grasp.to_csv(destination / "grasp_data.csv", index=False)
    camera.to_csv(destination / "camera.csv", index=False)

    frames = grasp["frame"].to_numpy()
    velocity_column = "velocity_3d_speed" if "velocity_3d_speed" in grasp.columns else "velocity"

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    if "object_area" in grasp.columns:
        axes[0].plot(frames, grasp["object_area"], linewidth=2)
    axes[0].set_ylabel("Object Area (px^2)")

    if "finger_width" in grasp.columns:
        axes[1].plot(frames, grasp["finger_width"], linewidth=2)
    axes[1].set_ylabel("Finger Width (px)")

    if velocity_column in grasp.columns:
        axes[2].plot(frames, grasp[velocity_column], linewidth=2, label="3D speed")
        axes[2].legend()
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].set_xlabel("Frame")

    for axis in axes:
        axis.grid(alpha=0.3)

    fig.suptitle(f"Repair mode: {mode}")
    fig.tight_layout()
    fig.savefig(destination / "grasp_curve.png", dpi=160)
    plt.close(fig)

    if object_changes:
        print("\nChanged object_area values:")
        for frame, old, new in object_changes:
            print(f"frame {frame:4d}: {old:8.2f} -> {new:8.2f}")

    if finger_changes:
        print("\nChanged finger_width values:")
        for frame, old, new in finger_changes:
            print(f"frame {frame:4d}: {old:8.2f} -> {new:8.2f}")

    print(f"\nSaved to: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, default=TRIAL_DIR)
    parser.add_argument("--mode", choices=["object", "finger", "both"], default=REPAIR_MODE)
    parser.add_argument("--range", dest="object_ranges", action="append", type=parse_range, help="Object-area repair range, e.g. --range 73:78")
    parser.add_argument("--finger-range", dest="finger_ranges", action="append", type=parse_finger_range, help="Finger repair range start:end:target, e.g. --finger-range 42:56:118")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory. Omit to modify the trial in place.")
    args = parser.parse_args()

    object_ranges = args.object_ranges if args.object_ranges is not None else OBJECT_REPAIR_RANGES
    finger_ranges = args.finger_ranges if args.finger_ranges is not None else FINGER_REPAIR_RANGES

    repair_trial(args.trial, args.mode, object_ranges, finger_ranges, args.output_dir)


if __name__ == "__main__":
    main()
