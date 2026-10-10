"""Regenerate trial plots from existing CSVs without rewriting training data."""
from pathlib import Path
import argparse
import hashlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from Smooth import save_preview

DEFAULT_ROOT = Path(__file__).resolve().parent / "Data" / "临时数据" / "D10" / "3"


def sync_trial(trial):
    raw_path = trial / "grasp_data.csv"
    smooth_path = trial / "grasp_data_smoothed.csv"
    hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in (raw_path, smooth_path)}
    raw = pd.read_csv(raw_path)
    smooth = pd.read_csv(smooth_path)
    if not raw.frame.equals(smooth.frame):
        raise ValueError(f"Raw and smoothed frame ranges differ: {trial}")
    frames = raw.frame
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(frames, raw.object_area, color="tab:red", linewidth=1.8)
    axes[0].set_ylabel("Object area (px²)")
    axes[1].plot(frames, raw.finger_width, color="tab:blue", linewidth=1.8)
    axes[1].set_ylabel("Finger width (px)")
    for col in ("ax", "ay", "az"):
        axes[2].plot(frames, raw[col], label=col, linewidth=1.8)
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].legend()
    axes[2].set_xlabel("Frame")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.suptitle(f"{trial.name}\nCurrent grasp_data.csv")
    fig.tight_layout()
    fig.savefig(trial / "grasp_curve.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for col in ("ax", "ay", "az"):
        axes[0].plot(frames, raw[col], label=col)
    axes[0].set_ylabel("CSV acceleration (m/s²)")
    axes[0].legend()
    axes[1].plot(frames, raw.velocity, color="tab:purple", label="Stored CSV speed")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_xlabel("Frame")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.3)
    fig.suptitle(f"{trial.name}\nExisting CSV values; speed not recalculated")
    fig.tight_layout()
    fig.savefig(trial / "velocity_3d_speed.png", dpi=160)
    plt.close(fig)

    save_preview(raw, smooth, trial / "smoothing_preview.png")
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).digest() == digest, f"CSV changed: {path}"
    print(f"[OK] {trial.name}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    trials = sorted(p for p in args.data_root.glob("TRIAL_*") if p.is_dir())
    for trial in trials:
        sync_trial(trial)
    print(f"Updated three plots for {len(trials)} trials; CSV files unchanged.")


if __name__ == "__main__":
    main()
