"""
clean_downsample.py
清洗 + 降采样：
  1. 只保留 valid=1 的帧
  2. 对停留帧降采样，平衡角度分布（过渡帧全保留，停留帧每N帧留1帧）

输出一个新的 CSV（不修改原始数据，图片和npy不动）

Usage:
    python clean_downsample.py recording_20260703_150615
    python clean_downsample.py recording_20260703_150615 --keep-every 10
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData"

RECORDING  = "recording_20260705_192855"   # 要处理的文件夹
KEEP_EVERY = 10                             # 停留帧每N帧保留1帧

def clean_and_downsample(df, keep_every):
    """
    过滤valid=1，并对停留帧降采样
    过渡帧（角度变化）全部保留
    停留帧（角度不变）每 keep_every 帧保留1帧
    """
    # 第一步：只保留 valid=1
    if "valid" in df.columns:
        before = len(df)
        df = df[df["valid"] == 1].reset_index(drop=True)
        print(f"[Step 1] valid=1 过滤: {before} → {len(df)} 帧")
    else:
        print("[Step 1] 没有valid列，跳过")

    # 角度转数值
    df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
    df = df[df["angle"].notna()].reset_index(drop=True)

    # 第二步：降采样停留帧
    keep_mask   = []
    prev_angle  = None
    stable_count = 0

    for _, row in df.iterrows():
        angle = row["angle"]
        if angle == prev_angle:
            # 停留帧
            stable_count += 1
            keep_mask.append(stable_count % keep_every == 0)
        else:
            # 过渡帧，全部保留
            stable_count = 0
            keep_mask.append(True)
        prev_angle = angle

    df_clean = df[keep_mask].reset_index(drop=True)
    print(f"[Step 2] 停留帧降采样(每{keep_every}帧留1): {len(df)} → {len(df_clean)} 帧")

    return df, df_clean


def plot_comparison(df_before, df_after, save_path):
    """对比清洗前后的角度分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_before["angle"], bins=71, color="steelblue", edgecolor="none")
    axes[0].set_title(f"Before (valid only): {len(df_before)} frames")
    axes[0].set_xlabel("Angle (deg)")
    axes[0].set_ylabel("Frame Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(df_after["angle"], bins=71, color="orange", edgecolor="none")
    axes[1].set_title(f"After downsample: {len(df_after)} frames")
    axes[1].set_xlabel("Angle (deg)")
    axes[1].set_ylabel("Frame Count")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] 分布对比图: {save_path}")
    plt.show()

def main():
    rec_dir = os.path.join(BASE_DIR, RECORDING)
    if not os.path.exists(rec_dir):
        print(f"[Error] Folder not found: {rec_dir}")
        sys.exit(1)

    csv_files = [f for f in os.listdir(rec_dir) if f.endswith(".csv") and "clean" not in f]
    if not csv_files:
        print(f"[Error] No CSV in {rec_dir}")
        sys.exit(1)
    csv_path = os.path.join(rec_dir, csv_files[0])
    print(f"[Load] {csv_path}\n")

    df = pd.read_csv(csv_path)
    df_valid, df_clean = clean_and_downsample(df, KEEP_EVERY)

    out_path = csv_path.replace(".csv", "_clean.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"\n[Saved] 清洗后CSV: {out_path}")

    print("\n清洗后角度分布:")
    dist = df_clean["angle"].value_counts().sort_index()
    print(dist.to_dict())

    plot_path = csv_path.replace(".csv", "_distribution.png")
    plot_comparison(df_valid, df_clean, plot_path)

if __name__ == "__main__":
    main()
