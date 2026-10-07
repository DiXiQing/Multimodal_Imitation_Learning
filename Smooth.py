"""Smooth all grasp trials and save ONE combined Raw-vs-Smoothed preview image per trial.

For every TRIAL_* folder:
    grasp_data.csv
        -> grasp_data_smoothed.csv
        -> smoothing_preview.png

The original grasp_data.csv is never modified.
All filters are causal (current + past samples only), so they are compatible
with future real-time use.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\临时数据\D9")

INPUT_CSV = "grasp_data.csv"
OUTPUT_CSV = "grasp_data_smoothed.csv"
PREVIEW_NAME = "smoothing_preview.png"

MEDIAN_WINDOW = 3
EMA_SPAN = 5
SMOOTH_FINGER_WIDTH = True

REQUIRED_COLUMNS = ["frame", "finger_width", "object_area", "velocity", "ax", "ay", "az"]


def median_then_ema(series: pd.Series) -> pd.Series:
    return series.rolling(MEDIAN_WINDOW, min_periods=1).median().ewm(span=EMA_SPAN, adjust=False).mean()


def ema(series: pd.Series) -> pd.Series:
    return series.ewm(span=EMA_SPAN, adjust=False).mean()


def smooth_trial(df: pd.DataFrame) -> pd.DataFrame:
    smooth = df.copy()

    smooth["object_area"] = median_then_ema(df["object_area"])

    if SMOOTH_FINGER_WIDTH:
        smooth["finger_width"] = median_then_ema(df["finger_width"])

    smooth["velocity"] = ema(df["velocity"])

    for col in ["ax", "ay", "az"]:
        smooth[col] = ema(df[col])

    return smooth


def save_preview(raw: pd.DataFrame, smooth: pd.DataFrame, save_path: Path) -> None:
    frame = raw["frame"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(frame, raw["object_area"], alpha=0.5, label="Raw")
    axes[0].plot(frame, smooth["object_area"], linewidth=2, label="Smoothed")
    axes[0].set_ylabel("Object Area")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(frame, raw["finger_width"], alpha=0.5, label="Raw")
    axes[1].plot(frame, smooth["finger_width"], linewidth=2, label="Smoothed")
    axes[1].set_ylabel("Finger Width")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    acceleration_colors = {
        "ax": "tab:red",
        "ay": "tab:green",
        "az": "tab:blue",
    }

    for col, color in acceleration_colors.items():
        axes[2].plot(
            frame,
            raw[col],
            color=color,
            alpha=0.28,
            linestyle="--",
            linewidth=1.0,
            label=f"{col} raw",
        )
        axes[2].plot(
            frame,
            smooth[col],
            color=color,
            linewidth=2.0,
            label=f"{col} smoothed",
        )

    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].set_xlabel("Frame")
    axes[2].set_title("Three-axis acceleration")
    axes[2].legend(ncol=3)
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def process_trial(csv_path: Path) -> None:
    trial_dir = csv_path.parent
    trial_name = trial_dir.name

    raw = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        print(f"[Skip] {trial_name}: missing {missing}")
        return

    for col in REQUIRED_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    smooth = smooth_trial(raw)

    output_csv = trial_dir / OUTPUT_CSV
    preview_path = trial_dir / PREVIEW_NAME

    smooth.to_csv(output_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    save_preview(raw, smooth, preview_path)

    print(f"[OK] {trial_name}")
    print(f"     CSV: {output_csv}")
    print(f"     PNG: {preview_path}")


def main() -> None:
    csv_paths = sorted(DATA_ROOT.glob(f"TRIAL_*/{INPUT_CSV}"))

    if not csv_paths:
        raise SystemExit(f"No {INPUT_CSV} found under {DATA_ROOT}")

    print(f"Trials found: {len(csv_paths)}")
    print(f"MEDIAN_WINDOW={MEDIAN_WINDOW}, EMA_SPAN={EMA_SPAN}, SMOOTH_FINGER_WIDTH={SMOOTH_FINGER_WIDTH}\n")

    for csv_path in csv_paths:
        process_trial(csv_path)

    print("\nDone.")
    print("Original grasp_data.csv files were NOT modified.")


if __name__ == "__main__":
    main()
