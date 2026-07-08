"""
visualize_segments.py
可视化录制数据，显示角度曲线和valid段，方便定位需要修改的行号

Usage:
    python visualize_segments.py recording_20260701_212500
    python visualize_segments.py recording_20260701_212500 --csv data_20260701_212500.csv
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TrainingData"
RECORDING = "recording_20260705_184225"


def load_csv(rec_dir):
    csv_files = [f for f in os.listdir(rec_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"[Error] No CSV found in {rec_dir}")
        sys.exit(1)
    csv_path = os.path.join(rec_dir, csv_files[0])
    print(f"[Load] {csv_path}")
    df = pd.read_csv(csv_path)
    return df, csv_path


def find_segments(valid_series):
    """找出所有连续valid段的起止行号"""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(valid_series):
        if v == 1 and not in_seg:
            in_seg = True
            start = i
        elif v != 1 and in_seg:
            in_seg = False
            segments.append((start, i - 1))
    if in_seg:
        segments.append((start, len(valid_series) - 1))
    return segments


def plot(df, csv_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{os.path.basename(csv_path)}", fontsize=12)

    frame_idx = df["frame_index"].values
    angle     = pd.to_numeric(df["angle"], errors="coerce").values
    valid     = df["valid"].fillna(0).astype(int).values

    # ── 找出valid段 ──
    segments = find_segments(valid)
    print(f"\nValid segments found: {len(segments)}")
    for i, (s, e) in enumerate(segments):
        print(f"  Segment {i+1}: row {s+2} ~ {e+2}  "
              f"(frame {frame_idx[s]} ~ {frame_idx[e]}, "
              f"{e - s + 1} frames)")  # +2 因为CSV有表头行，行号从1开始

    # ── Plot 1: Angle ──
    ax = axes[0]
    ax.plot(frame_idx, angle, color="steelblue", linewidth=0.8, label="Angle")
    for s, e in segments:
        ax.axvspan(frame_idx[s], frame_idx[e], alpha=0.2, color="green")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Servo Angle  (green = valid)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ── Plot 2: Valid flag ──
    ax = axes[1]
    ax.fill_between(frame_idx, valid, step="mid", alpha=0.6,
                    color="green", label="valid=1")
    ax.set_ylabel("Valid")
    ax.set_ylim(-0.1, 1.3)
    ax.set_title("Valid Flag  (1=valid, 0=invalid)")
    ax.grid(True, alpha=0.3)

    # 标注每个段的起止行号
    for i, (s, e) in enumerate(segments):
        mid = (frame_idx[s] + frame_idx[e]) / 2
        ax.text(mid, 1.1, f"#{i+1}\nrow {s+2}~{e+2}",
                ha="center", va="bottom", fontsize=7,
                color="darkgreen", fontweight="bold")

    # ── Plot 3: EMG RMS（从emg_file读取，或从CSV估算）──
    ax = axes[2]
    if "emg_ch1_t99" in df.columns:
        # 有EMG列，取最后10个时间步的RMS
        emg_cols_last = [f"emg_ch{ch}_t{t}"
                         for t in range(90, 100)
                         for ch in range(1, 5)]
        available = [c for c in emg_cols_last if c in df.columns]
        if available:
            emg_rms = df[available].apply(
                pd.to_numeric, errors="coerce").fillna(0).values
            rms_per_frame = np.sqrt(np.mean(emg_rms ** 2, axis=1))
            ax.plot(frame_idx, rms_per_frame,
                    color="orange", linewidth=0.8, label="EMG RMS")
            for s, e in segments:
                ax.axvspan(frame_idx[s], frame_idx[e], alpha=0.2, color="green")
            ax.set_ylabel("EMG RMS (mV)")
            ax.set_title("EMG Activity  (green = valid)")
            ax.grid(True, alpha=0.3)
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No EMG columns in CSV\n(EMG stored in .npy files)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title("EMG Activity")

    axes[2].set_xlabel("Frame Index")
    plt.tight_layout()

    # 保存图片
    save_path = csv_path.replace(".csv", "_segments.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n[Saved] {save_path}")
    plt.show()


def main():
    rec_dir = os.path.join(BASE_DIR, RECORDING)
    if not os.path.exists(rec_dir):
        print(f"[Error] Folder not found: {rec_dir}")
        sys.exit(1)

    df, csv_path = load_csv(rec_dir)
    plot(df, csv_path)


if __name__ == "__main__":
    main()
