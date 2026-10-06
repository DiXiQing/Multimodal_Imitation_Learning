"""Generate D2-style grasp_curve.png files for every D3 trial.

The script reads each trial's existing grasp_data.csv and only writes
grasp_curve.png. It does not modify the CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = ["frame", "finger_width", "object_area", "velocity"]


def generate_grasp_curve(csv_path: Path) -> Path:
    trial_dir = csv_path.parent
    data = pd.read_csv(csv_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    frames = data["frame"].to_numpy()

    figure, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(
        frames,
        data["object_area"],
        color="tab:red",
        linewidth=1.8,
    )
    axes[0].set_ylabel("Object Area (px^2)")
    axes[0].set_title("Object Area / Finger Width / 3D Speed")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        frames,
        data["finger_width"],
        color="tab:blue",
        linewidth=1.8,
    )
    axes[1].set_ylabel("Finger Width (px)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        frames,
        data["velocity"],
        color="tab:purple",
        linewidth=2,
        label="3D speed",
    )
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    output_path = trial_dir / "grasp_curve.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate D2-style grasp_curve.png files for trial folders"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(r"D:\Code\Multimodal_Imitation_Learning\Data\临时数据\D3"),
    )
    args = parser.parse_args()

    csv_paths = sorted(
        path / "grasp_data.csv"
        for path in args.root_dir.iterdir()
        if path.is_dir() and (path / "grasp_data.csv").is_file()
    )
    if not csv_paths:
        raise FileNotFoundError(f"No trial grasp_data.csv files found in {args.root_dir}")

    for csv_path in csv_paths:
        output_path = generate_grasp_curve(csv_path)
        print(output_path)

    print(f"Generated {len(csv_paths)} grasp_curve.png files.")


if __name__ == "__main__":
    main()
