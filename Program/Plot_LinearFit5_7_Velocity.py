"""Plot raw backward-difference, Linear Fit 5 and Linear Fit 7 velocities."""
from pathlib import Path
from datetime import datetime
import csv
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("1101_D", "1115_Song")
MAX_GAP_S = .10


def read_trial(folder):
    with (folder / "camera.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    def values(key):
        return np.array([float(row[key]) if row.get(key) else np.nan for row in rows])
    times = values("t_camera")
    distance = values("index_mcp_object_distance_mm")
    frames = values("frame")
    valid = np.array([row["valid"] == "1" and row["index_mcp_distance_valid"] == "1" for row in rows])
    valid &= np.isfinite(times) & np.isfinite(distance)
    return times, distance, frames, valid


def velocities(times, distance, frames, valid):
    result = {window: np.full(len(times), np.nan) for window in (2, 5, 7)}
    run_start = 0
    for i in range(len(times)):
        if not valid[i]:
            run_start = i + 1
            continue
        if i and (not valid[i-1] or frames[i]-frames[i-1] != 1
                  or not 0 < times[i]-times[i-1] <= MAX_GAP_S):
            run_start = i
        for window, values in result.items():
            if i-run_start+1 < window:
                continue
            t = times[i-window+1:i+1] - times[i]
            d = distance[i-window+1:i+1]
            centered = t - t.mean()
            values[i] = np.dot(centered, d-d.mean()) / np.dot(centered, centered)
    return result


def main():
    output = PROJECT / "Data/Result" / ("Velocity_LinearFit5_7_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    selected = []
    colors = {2: "#aaaaaa", 5: "#ed9b29", 7: "#3285bd"}
    labels = {2: "Raw backward difference", 5: "Linear fit 5 frames", 7: "Linear fit 7 frames"}
    for row, dataset in enumerate(DATASETS):
        for col, size in enumerate((20, 30, 40)):
            home = PROJECT / "Data" / dataset / (f"{size}mm" if dataset == "1101_D" else str(size))
            candidates = []
            for folder in sorted(home.iterdir()):
                if not folder.is_dir() or not re.fullmatch(r"\d{8}_\d{6}", folder.name) or not (folder/"camera.csv").is_file():
                    continue
                times, distance, frames, valid = read_trial(folder)
                if len(times) >= 7 and valid.sum() >= 7:
                    candidates.append((times[-1]-times[0], folder, times, distance, frames, valid))
            if not candidates:
                raise ValueError(f"No usable trials in {home}")
            candidates.sort(key=lambda item: (item[0], item[1].name))
            duration, folder, times, distance, frames, valid = candidates[len(candidates)//2]
            values = velocities(times, distance, frames, valid)
            t = times-times[0]
            ax = axes[row, col]
            for window in (2, 5, 7):
                ax.plot(t, values[window], color=colors[window],
                        alpha=.65 if window == 2 else 1, linewidth=1 if window == 2 else 1.6,
                        label=labels[window])
            ax.set(title=f"{dataset} | {size} mm | {folder.name}",
                   xlabel="Time (s)", ylabel="Radial velocity (mm/s)")
            ax.axhline(0, color="#555555", linewidth=.7)
            ax.grid(alpha=.2)
            with (output/f"{dataset}_{size}mm_{folder.name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_s", "distance_mm", "raw_velocity_mm_s", "linearfit5_mm_s", "linearfit7_mm_s"])
                writer.writerows(zip(t, distance, values[2], values[5], values[7]))
            selected.append(dict(dataset=dataset, size_mm=size, trial=folder.name, source=str(folder), duration_s=duration))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    image = output / "velocity_comparison.png"
    fig.savefig(image, dpi=200)
    plt.close(fig)
    with (output/"selected_trials.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(f"Saved: {image}", flush=True)
    return image


if __name__ == "__main__":
    import os
    os.startfile(str(main()))
