"""Extract temporal-coupling, grasp-characteristic, and detection-quality metrics.

Each trial folder must contain a grasp_data.csv with these columns:
    frame, t_camera, finger_width, velocity, object_area

The output CSV contains:
    1) Temporal relationship metrics
    2) Grasp characteristic values
    3) Detection-quality diagnostics

No trial is automatically deleted by QC thresholds. The QC columns are intended
to identify suspicious trials for visual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\临时数据\D2")
DEFAULT_OUTPUT_NAME = "temporal_coupling_metrics.csv"
RESERVED_FRAMES = 15
SMOOTHING_WINDOW = 3
REQUIRED_COLUMNS = ["frame", "t_camera", "finger_width", "velocity", "object_area"]


def max_consecutive_jump(values: np.ndarray, frames: np.ndarray) -> float:
    """Maximum absolute change between two consecutive camera frames."""
    if len(values) < 2:
        return float("nan")
    frame_diff = np.diff(frames)
    value_diff = np.abs(np.diff(values))
    consecutive = frame_diff == 1
    return float(np.max(value_diff[consecutive])) if np.any(consecutive) else float("nan")


def max_consecutive_relative_jump(values: np.ndarray, frames: np.ndarray) -> float:
    """Maximum |x[t]-x[t-1]| / x[t-1] between consecutive frames."""
    if len(values) < 2:
        return float("nan")
    frame_diff = np.diff(frames)
    previous = values[:-1]
    value_diff = np.abs(np.diff(values))
    valid = (frame_diff == 1) & (previous > 0)
    return float(np.max(value_diff[valid] / previous[valid])) if np.any(valid) else float("nan")


def calculate_trial(csv_path: Path) -> dict[str, float | int | str]:
    df_raw = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df_raw.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    for column in REQUIRED_COLUMNS:
        df_raw[column] = pd.to_numeric(df_raw[column], errors="coerce")

    total_rows = len(df_raw)
    invalid_frame = df_raw["frame"].isna()
    invalid_time = df_raw["t_camera"].isna()
    invalid_finger = df_raw["finger_width"].isna() | (df_raw["finger_width"] <= 0)
    invalid_velocity = df_raw["velocity"].isna() | (df_raw["velocity"] < 0)
    invalid_area = df_raw["object_area"].isna() | (df_raw["object_area"] <= 0)
    invalid_any = invalid_frame | invalid_time | invalid_finger | invalid_velocity | invalid_area

    df = df_raw.loc[~invalid_any, REQUIRED_COLUMNS].copy()
    df = df.sort_values(["frame", "t_camera"]).reset_index(drop=True)

    if len(df) <= RESERVED_FRAMES + 2:
        raise ValueError(f"only {len(df)} valid rows; need more than {RESERVED_FRAMES + 2}")

    aperture = df["finger_width"].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).median()
    speed = df["velocity"].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).median()

    time = df["t_camera"].to_numpy(dtype=float)
    frames = df["frame"].to_numpy(dtype=float)
    finger_values = df["finger_width"].to_numpy(dtype=float)
    area_values = df["object_area"].to_numpy(dtype=float)

    aperture_index = RESERVED_FRAMES + int(aperture.iloc[RESERVED_FRAMES:].to_numpy().argmax())
    speed_index = RESERVED_FRAMES + int(speed.iloc[RESERVED_FRAMES:].to_numpy().argmax())
    final_index = len(df) - 1

    initial_aperture = float(aperture.iloc[:RESERVED_FRAMES].median())
    max_aperture = float(aperture.iloc[aperture_index])
    final_aperture = float(aperture.iloc[final_index])

    start_object_area = float(df["object_area"].iloc[:RESERVED_FRAMES].median())
    final_object_area = float(df.loc[final_index, "object_area"])
    object_area_at_max_aperture = float(df.loc[aperture_index, "object_area"])

    final_over_start = final_object_area / start_object_area if start_object_area > 0 else float("nan")
    area_at_max_over_final = object_area_at_max_aperture / final_object_area if final_object_area > 0 else float("nan")

    closure_amplitude = max_aperture - final_aperture
    peak_to_half_closure_s = float("nan")
    if closure_amplitude > 0:
        half_level = max_aperture - 0.5 * closure_amplitude
        after_peak = aperture.iloc[aperture_index:].to_numpy()
        half_candidates = np.flatnonzero(after_peak <= half_level)
        if len(half_candidates) > 0:
            half_index = aperture_index + int(half_candidates[0])
            peak_to_half_closure_s = float(time[half_index] - time[aperture_index])

    frame_diff = np.diff(frames)
    frame_gap_count = int(np.sum(frame_diff > 1))
    max_frame_gap = float(np.max(frame_diff)) if len(frame_diff) else float("nan")

    return {
        "trial": csv_path.parent.name,

        # Temporal / input-output relationship
        "max_aperture_minus_peak_speed_s": float(time[aperture_index] - time[speed_index]),
        "object_area_at_max_aperture_over_final": area_at_max_over_final,

        # Grasp characteristics
        "initial_aperture_px": initial_aperture,
        "max_aperture_px": max_aperture,
        "final_aperture_px": final_aperture,
        "start_object_area_px2": start_object_area,
        "final_object_area_px2": final_object_area,
        "final_area_over_start_area": final_over_start,
        "object_area_at_max_aperture_px2": object_area_at_max_aperture,
        "peak_to_half_closure_s": peak_to_half_closure_s,

        # Detection-quality diagnostics
        "total_rows": int(total_rows),
        "valid_rows": int(len(df)),
        "valid_row_ratio": float(len(df) / total_rows) if total_rows else float("nan"),
        "invalid_finger_width_count": int(invalid_finger.sum()),
        "invalid_object_area_count": int(invalid_area.sum()),
        "invalid_velocity_count": int(invalid_velocity.sum()),
        "invalid_any_count": int(invalid_any.sum()),
        "max_finger_width_jump_px": max_consecutive_jump(finger_values, frames),
        "max_object_area_jump_px2": max_consecutive_jump(area_values, frames),
        "max_object_area_relative_jump": max_consecutive_relative_jump(area_values, frames),
        "frame_gap_count": frame_gap_count,
        "max_frame_gap": max_frame_gap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract grasp, temporal-coupling, and detection-quality metrics.")
    parser.add_argument("data_root", nargs="?", default=str(DEFAULT_DATA_ROOT), help="Dataset root containing trial folders.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to temporal_coupling_metrics.csv inside data_root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    if not data_root.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {data_root}")

    output_path = Path(args.output).expanduser() if args.output else data_root / DEFAULT_OUTPUT_NAME
    trial_csvs = sorted(data_root.rglob("grasp_data.csv"))
    if not trial_csvs:
        raise SystemExit(f"No grasp_data.csv found below: {data_root}")

    rows = []
    skipped = []
    for csv_path in trial_csvs:
        try:
            rows.append(calculate_trial(csv_path))
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            skipped.append((csv_path.parent.name, str(exc)))

    if not rows:
        raise SystemExit("No valid trials were found.")

    result = pd.DataFrame(rows).sort_values("trial")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    print(f"Processed trials: {len(result)}")
    print(f"Saved CSV: {output_path}")
    if skipped:
        print(f"Skipped trials: {len(skipped)}")
        for trial, reason in skipped:
            print(f"  - {trial}: {reason}")


if __name__ == "__main__":
    main()
