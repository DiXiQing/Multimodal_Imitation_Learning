"""
plot_labels.py
可视化标注结果：在角度曲线上标出每个山峰的 open_target / close_target

读 _labeled.csv，画出：
  - 角度曲线
  - valid段（绿色背景）
  - 张开端点（红点，峰顶）
  - 夹紧端点（蓝点，峰后谷底）
  - 每个山峰标注目标值

用法：改 RECORDING 后运行 python plot_labels.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TrainingData"

# ─────── 改这里 ───────
RECORDING = "recording_20260705_184225"
# ──────────────────────


def find_valid_segments(valid):
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(valid):
        if v == 1 and not in_seg:
            in_seg = True; start = i
        elif v != 1 and in_seg:
            in_seg = False; segments.append((start, i - 1))
    if in_seg:
        segments.append((start, len(valid) - 1))
    return segments


def main():
    rec_dir = os.path.join(BASE_DIR, RECORDING)
    labeled = [f for f in os.listdir(rec_dir) if f.endswith("_labeled.csv")]
    if not labeled:
        print("[Error] 没找到 _labeled.csv，先运行 label_targets.py")
        sys.exit(1)
    csv_path = os.path.join(rec_dir, labeled[0])
    print(f"[Load] {csv_path}")

    df = pd.read_csv(csv_path).reset_index(drop=True)
    df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
    frame_idx = df["frame_index"].values
    angle     = df["angle"].values
    valid     = df["valid"].fillna(0).astype(int).values

    fig, ax = plt.subplots(figsize=(16, 6))

    # 角度曲线
    ax.plot(frame_idx, angle, color="gray", linewidth=1, label="Angle", zorder=1)

    segments = find_valid_segments(valid)
    for seg_i, (s, e) in enumerate(segments):
        # valid段绿色背景
        ax.axvspan(frame_idx[s], frame_idx[e], alpha=0.12, color="green", zorder=0)

        seg_angles = angle[s:e+1]
        peak_local = int(np.nanargmax(seg_angles))
        peak_idx   = s + peak_local
        # 峰后谷底
        after = seg_angles[peak_local:]
        valley_local = peak_local + int(np.nanargmin(after))
        valley_idx = s + valley_local

        # 张开端点（红点）
        ax.scatter(frame_idx[peak_idx], angle[peak_idx],
                   color="red", s=80, zorder=3)
        ax.annotate(f"open={angle[peak_idx]:.0f}",
                    (frame_idx[peak_idx], angle[peak_idx]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", color="red", fontsize=8, fontweight="bold")

        # 夹紧端点（蓝点）
        ax.scatter(frame_idx[valley_idx], angle[valley_idx],
                   color="blue", s=80, zorder=3)
        ax.annotate(f"close={angle[valley_idx]:.0f}",
                    (frame_idx[valley_idx], angle[valley_idx]),
                    textcoords="offset points", xytext=(0, -15),
                    ha="center", color="blue", fontsize=8, fontweight="bold")

        # 段号
        mid = (frame_idx[s] + frame_idx[e]) / 2
        ax.text(mid, ax.get_ylim()[1]*0.95, f"#{seg_i+1}",
                ha="center", fontsize=9, color="darkgreen")

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(f"Labeled Endpoints: {RECORDING}  ({len(segments)} peaks)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = csv_path.replace(".csv", "_plot.png")
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
