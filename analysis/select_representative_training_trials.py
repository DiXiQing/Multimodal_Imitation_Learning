from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\Training")
OUTPUT_DIR = Path(__file__).resolve().parent / "representative_training_trials"
GROUPS = ("Black", "Blue", "Red")
FEATURES = ("object_area", "finger_width", "velocity")
POINTS = 120


def resample(values: np.ndarray, points: int = POINTS) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, source, values.astype(float))


def load_group(group: str):
    records = []
    for csv_path in sorted(DATA_ROOT.glob(f"TRIAL_{group}_*/grasp_data.csv")):
        frame = pd.read_csv(csv_path)
        if not set(FEATURES).issubset(frame.columns):
            continue
        curves = np.stack(
            [resample(frame[column].to_numpy(dtype=float)) for column in FEATURES]
        )
        records.append(
            {
                "trial": csv_path.parent.name,
                "rows": len(frame),
                "curves": curves,
            }
        )
    return records


def select_trio(records):
    if len(records) < 3:
        raise RuntimeError("At least three trials are required")
    if len(records) == 3:
        return records, 0.0

    all_curves = np.stack([record["curves"] for record in records])
    flattened = all_curves.transpose(1, 0, 2).reshape(len(FEATURES), -1)
    scales = np.percentile(flattened, 95, axis=1) - np.percentile(
        flattened, 5, axis=1
    )
    scales = np.where(scales > 1e-9, scales, 1.0)
    median_rows = float(np.median([record["rows"] for record in records]))

    def distance(left, right):
        difference = (left["curves"] - right["curves"]) / scales[:, None]
        curve_distance = float(np.sqrt(np.mean(difference**2)))
        duration_distance = abs(left["rows"] - right["rows"]) / median_rows
        return curve_distance + 0.20 * duration_distance

    best = None
    best_score = float("inf")
    for indices in itertools.combinations(range(len(records)), 3):
        trio = [records[index] for index in indices]
        pairs = itertools.combinations(trio, 2)
        score = float(np.mean([distance(left, right) for left, right in pairs]))
        if score < best_score:
            best_score = score
            best = trio
    return best, best_score


def plot_selected(selected_by_group):
    labels = ("Object area (px²)", "Finger width (px)", "3D speed (m/s)")
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    progress = np.linspace(0, 100, POINTS)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True)
    for column, group in enumerate(GROUPS):
        trials = selected_by_group[group]
        for row, feature_label in enumerate(labels):
            axis = axes[row, column]
            for color, record in zip(colors, trials):
                short_name = record["trial"].replace(f"TRIAL_{group}_", "")
                axis.plot(
                    progress,
                    record["curves"][row],
                    linewidth=2.2,
                    color=color,
                    label=short_name,
                )
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel(feature_label)
            if row == 0:
                axis.set_title(f"{group} object", fontsize=15)
                axis.legend(title="Trial", fontsize=9)
            if row == 2:
                axis.set_xlabel("Normalized trial progress (%)")

    fig.suptitle("Representative training trials", fontsize=19, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUTPUT_DIR / "representative_training_trials.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_by_group = {}
    rows = []

    for group in GROUPS:
        records = load_group(group)
        selected, score = select_trio(records)
        selected_by_group[group] = selected
        print(f"{group}: similarity score={score:.5f}")
        for record in selected:
            print(f"  {record['trial']} | rows={record['rows']}")
            rows.append(
                {
                    "object": group,
                    "trial": record["trial"],
                    "rows": record["rows"],
                    "trio_similarity_score": score,
                }
            )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "representative_training_trials.csv",
        index=False,
    )
    plot_selected(selected_by_group)


if __name__ == "__main__":
    main()
