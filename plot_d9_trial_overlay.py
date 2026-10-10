"""Overlay the trials in the latest no-velocity model split; no data changes."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "临时数据" / "D9"
OUT = ROOT / "analysis" / "d9_trial_overlay"
SPECIAL = "TRIAL_Size_3cm_20260927_151649"
ORIGINAL = ROOT / "analysis" / "d9_151649_before_delay2_20261010_120135" / "grasp_data_smoothed.csv"
FEATURES = [("finger_width", "Finger width (px)"), ("object_area", "Object area (px²)"), ("ax", "ax (m/s²)"), ("ay", "ay (m/s²)"), ("az", "az (m/s²)")]

def main():
    split = pd.read_csv(ROOT / "window_mlp_d9_no_velocity" / "data_split.csv")
    names = sorted(split.trial.unique())
    output_dir = OUT.with_name(f"d9_trial_overlay_{len(names)}trials")
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = {name: pd.read_csv(DATA / name / "grasp_data_smoothed.csv") for name in names}
    original = pd.read_csv(ORIGINAL) if SPECIAL in names and ORIGINAL.exists() else None
    for alignment in ["relative_frames", "relative_progress"]:
        fig, axes = plt.subplots(5, 1, figsize=(15, 17), sharex=True)
        colors = plt.get_cmap("tab20")
        for index, (name, df) in enumerate(trials.items()):
            elapsed = df.frame - df.frame.iloc[0]
            x = elapsed if alignment == "relative_frames" else elapsed / elapsed.iloc[-1] * 100
            special = name == SPECIAL
            label = name.removeprefix("TRIAL_Size_3cm_") + (" (modified)" if special else "")
            for ax, (column, ylabel) in zip(axes, FEATURES):
                ax.plot(x, df[column], color="red" if special else colors(index), linewidth=3 if special else 1.4, alpha=1 if special else .65, label=label, zorder=5 if special else 2)
        if original is not None:
            elapsed = original.frame - original.frame.iloc[0]
            x = elapsed if alignment == "relative_frames" else elapsed / elapsed.iloc[-1] * 100
            for ax, (column, ylabel) in zip(axes, FEATURES):
                ax.plot(x, original[column], color="black", linestyle="--", linewidth=2, label="20260927_151649 (original)", zorder=6)
        for ax, (_, ylabel) in zip(axes, FEATURES):
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
        axes[-1].set_xlabel("Frames from crop start" if alignment == "relative_frames" else "Trial progress (%)")
        subtitle = " — red: modified 151649; dashed black: original" if original is not None else " — measured trajectories"
        fig.suptitle(f"D9: {len(names)} trials{subtitle}", fontsize=14)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
        fig.tight_layout(rect=(0,.095,1,.975))
        fig.savefig(output_dir / f"overlay_{alignment}.png", dpi=160)
        plt.close(fig)
    print(f"Saved overlays for {len(names)} trials to {output_dir}")

if __name__ == "__main__":
    main()
