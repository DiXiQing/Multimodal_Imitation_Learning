"""Plot the 14 D10 4.5 cm training-input trajectories without editing data."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "临时数据" / "D10" / "4.5"
OUT = ROOT / "analysis" / "d10_4p5cm_overlay_14trials"
FEATURES = [("finger_width", "Finger width (px)"),
            ("object_area", "Object area (px²)"),
            ("ax", "ax (m/s²)"), ("ay", "ay (m/s²)"), ("az", "az (m/s²)")]

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    trials = {p.name: pd.read_csv(p / "grasp_data_smoothed.csv")
              for p in sorted(DATA.glob("TRIAL_Size_4p5cm_*")) if p.is_dir()}
    if len(trials) != 14:
        raise ValueError(f"Expected 14 trials, found {len(trials)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for alignment in ("relative_progress", "relative_frames"):
        fig, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True)
        for i, (name, df) in enumerate(trials.items()):
            elapsed = df.frame - df.frame.iloc[0]
            x = elapsed / elapsed.iloc[-1] * 100 if alignment == "relative_progress" else elapsed
            for ax, (col, label) in zip(axes, FEATURES):
                ax.plot(x, df[col], color=plt.get_cmap("tab20")(i),
                        linewidth=1.7, alpha=.8, label=name.removeprefix("TRIAL_Size_4p5cm_"))
        for ax, (_, label) in zip(axes, FEATURES):
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
        axes[-1].set_xlabel("Trial progress (%)" if alignment == "relative_progress" else "Frames from crop start")
        fig.suptitle("D10 / 4.5 cm: 14 trials — grasp_data_smoothed.csv", fontsize=16)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10)
        fig.tight_layout(rect=(0, .095, 1, .97))
        path = OUT / f"overlay_{alignment}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(path)

if __name__ == "__main__":
    main()
