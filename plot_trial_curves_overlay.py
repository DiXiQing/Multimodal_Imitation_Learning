"""Overlay object area, finger width, and 3D speed from several trials."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Configuration: edit only this section
# ============================================================

DATA_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData")

TRIAL_NAMES = [
    "TRIAL_Red_20260928_112852",
    "TRIAL_Red_20260928_112825",
    "TRIAL_Red_20260928_112734",
    "TRIAL_Red_20260928_112711",
    "TRIAL_Red_20260928_112619",
    "TRIAL_Red_20260928_112525",
    "TRIAL_Red_20260928_112505",
    "TRIAL_Red_20260928_112417",
    "TRIAL_Red_20260928_112250",
    "TRIAL_Red_20260928_112151",
]

# "relative_frame": each curve starts at frame 0.
# "progress": rescales every trial to 0-100% for shape comparison.
X_AXIS_MODE = "relative_frame"

OUTPUT_FILE = Path(__file__).resolve().parent / "trial_curves_overlay.png"


PLOTS = [
    ("object_area", "Object Area (px^2)", "Object Area"),
    ("finger_width", "Finger Width (px)", "Finger Width"),
    ("velocity", "3D Speed (m/s)", "3D Speed"),
]


def load_trial(trial_name: str) -> pd.DataFrame:
    csv_path = DATA_ROOT / trial_name / "grasp_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = ["frame", *(column for column, _, _ in PLOTS)]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{trial_name} is missing columns: {missing}")

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{trial_name} has no valid rows")
    return frame


def make_x_axis(frame: pd.DataFrame):
    if X_AXIS_MODE == "relative_frame":
        return frame["frame"] - frame["frame"].iloc[0], "Relative Frame"
    if X_AXIS_MODE == "progress":
        if len(frame) == 1:
            return [0.0], "Movement Progress (%)"
        return frame.index.to_numpy() * 100.0 / (len(frame) - 1), "Movement Progress (%)"
    raise ValueError('X_AXIS_MODE must be "relative_frame" or "progress"')


def short_label(trial_name: str) -> str:
    return trial_name.removeprefix("TRIAL_Blue_").removeprefix("TRIAL_")


def main() -> None:
    trials = [(name, load_trial(name)) for name in TRIAL_NAMES]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    x_label = ""
    for trial_name, frame in trials:
        x, x_label = make_x_axis(frame)
        label = short_label(trial_name)
        for axis, (column, y_label, _) in zip(axes, PLOTS):
            axis.plot(x, frame[column], linewidth=2.0, label=label)

    for axis, (_, y_label, title) in zip(axes, PLOTS):
        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.suptitle("Trial Curve Comparison", fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.84, 0.97))
    fig.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Compared trials: {len(trials)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()