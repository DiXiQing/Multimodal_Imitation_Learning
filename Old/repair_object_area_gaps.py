"""Repair missing object_area values in one analyzed trial.

Rules:
1. A zero, negative, NaN, or infinite object_area is treated as missing.
2. A gap between valid samples is filled by linear interpolation.
3. A gap at the beginning or end is filled with the nearest valid value.
4. grasp_data.csv is overwritten after a backup is created.
5. Repaired frame values are also synchronized to camera.csv.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration: change this path for another experiment
# ============================================================

TRIAL_DIR = Path(
    r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData"
    r"\TRIAL_Red_20260928_112825"
)

GRASP_CSV = TRIAL_DIR / "grasp_data.csv"
CAMERA_CSV = TRIAL_DIR / "camera.csv"
COMPARISON_PLOT = TRIAL_DIR / "object_area_repair_comparison.png"


def missing_mask(values: np.ndarray) -> np.ndarray:
    return ~np.isfinite(values) | (values <= 0)


def missing_ranges(frames: np.ndarray, mask: np.ndarray):
    ranges = []
    start = None

    for index, is_missing in enumerate(mask):
        if is_missing and start is None:
            start = index
        elif not is_missing and start is not None:
            ranges.append((int(frames[start]), int(frames[index - 1])))
            start = None

    if start is not None:
        ranges.append((int(frames[start]), int(frames[-1])))

    return ranges


def repair_values(frames: np.ndarray, values: np.ndarray):
    mask = missing_mask(values)
    valid = ~mask

    if valid.sum() < 2:
        raise ValueError("object_area has fewer than two valid values")

    repaired = values.copy()
    repaired[mask] = np.interp(
        frames[mask],
        frames[valid],
        values[valid],
    )

    return repaired, mask


def create_backup(path: Path) -> Path:
    backup = path.with_name(f"{path.stem}_before_area_repair{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def update_camera_csv(repaired_by_frame: dict[int, float]) -> Path | None:
    if not CAMERA_CSV.exists():
        return None

    camera = pd.read_csv(CAMERA_CSV)
    if "frame" not in camera.columns or "object_area" not in camera.columns:
        raise ValueError("camera.csv must contain frame and object_area columns")

    create_backup(CAMERA_CSV)

    frame_numbers = pd.to_numeric(camera["frame"], errors="coerce")
    for row_index, frame_number in frame_numbers.items():
        if np.isfinite(frame_number):
            frame_number = int(frame_number)
            if frame_number in repaired_by_frame:
                camera.at[row_index, "object_area"] = repaired_by_frame[frame_number]

    camera.to_csv(CAMERA_CSV, index=False)
    return CAMERA_CSV


def save_comparison_plot(
    frames: np.ndarray,
    original: np.ndarray,
    repaired: np.ndarray,
    mask: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(frames, original, label="Original", color="tab:red", linewidth=2)
    axis.plot(frames, repaired, label="Repaired", color="tab:blue", linewidth=2)
    axis.scatter(
        frames[mask],
        repaired[mask],
        label="Filled frames",
        color="black",
        s=30,
        zorder=3,
    )
    axis.set_xlabel("Frame")
    axis.set_ylabel("Object Area (px^2)")
    axis.set_title("Object Area Repair")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(COMPARISON_PLOT, dpi=180)
    plt.close(figure)


def main() -> None:
    if not GRASP_CSV.exists():
        raise FileNotFoundError(GRASP_CSV)

    grasp = pd.read_csv(GRASP_CSV)
    if "frame" not in grasp.columns or "object_area" not in grasp.columns:
        raise ValueError("grasp_data.csv must contain frame and object_area columns")

    frames = pd.to_numeric(grasp["frame"], errors="raise").to_numpy(dtype=np.int64)
    original = pd.to_numeric(
        grasp["object_area"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    repaired, mask = repair_values(frames, original)
    ranges = missing_ranges(frames, mask)

    if not mask.any():
        print("No missing object_area values found.")
        return

    grasp_backup = create_backup(GRASP_CSV)
    grasp.loc[:, "object_area"] = repaired
    grasp.to_csv(GRASP_CSV, index=False)

    repaired_by_frame = {
        int(frame): float(value)
        for frame, value, was_missing in zip(frames, repaired, mask)
        if was_missing
    }
    updated_camera = update_camera_csv(repaired_by_frame)
    save_comparison_plot(frames, original, repaired, mask)

    print(f"Missing ranges: {ranges}")
    print(f"Repaired frames: {int(mask.sum())}")
    print(f"Saved: {GRASP_CSV}")
    print(f"Backup: {grasp_backup}")
    if updated_camera is not None:
        print(f"Updated: {updated_camera}")
    print(f"Plot: {COMPARISON_PLOT}")


if __name__ == "__main__":
    main()
