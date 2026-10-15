from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


trial = Path(__file__).resolve().parent
raw = pd.read_csv(trial / "grasp_data.csv")
smooth = pd.read_csv(trial / "grasp_data_smoothed.csv")
start = raw["t_camera"].min()
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for axis, column, label, color in (
    (axes[0], "object_area", "Object area (px²)", "#22845c"),
    (axes[1], "finger_width", "Finger width (px)", "#2456a6"),
):
    axis.plot(smooth["frame"], smooth[column], color=color, linewidth=2)
    axis.set_ylabel(label)
axes[2].plot(smooth["frame"], smooth["forward_acceleration"], color="#ce4848", linewidth=2)
axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
axes[2].set_ylabel("Acceleration (m/s²)")
axes[2].set_xlabel("Frame")
for axis in axes:
    axis.grid(alpha=0.25)
    axis.margins(x=0)
fig.suptitle(trial.name)
fig.tight_layout()
fig.savefig(trial / "area_width_acceleration.png", dpi=180)
plt.close(fig)
print(trial / "area_width_acceleration.png")
