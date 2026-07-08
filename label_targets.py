"""
label_targets.py
自动标注抓取端点：给每个 valid 山峰段打上 open_target / close_target

逻辑：
  每个 valid=1 的连续段 = 一次抓取 = 一个山峰
    open_target  = 段内角度最大值（张开端点/峰顶）
    close_target = 峰顶之后的角度最小值（夹紧端点/峰后谷底）
  段内每一帧都打上这两个值

在原始CSV（已手动改好valid）上运行，输出 _labeled.csv，不改原文件
之后再对 _labeled.csv 做降采样

用法：改 RECORDING 后运行 python label_targets.py
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TrainingData"

# ─────── 改这里 ───────
RECORDING = "recording_20260705_184225"
# ──────────────────────


def find_valid_segments(valid):
    """找出所有 valid=1 的连续段，返回 [(start_idx, end_idx), ...]（含端点）"""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(valid):
        if v == 1 and not in_seg:
            in_seg = True
            start = i
        elif v != 1 and in_seg:
            in_seg = False
            segments.append((start, i - 1))
    if in_seg:
        segments.append((start, len(valid) - 1))
    return segments


def label(df):
    df = df.reset_index(drop=True).copy()
    df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
    valid = df["valid"].fillna(0).astype(int).values

    # 初始化两列（无效段为空）
    df["open_target"]  = np.nan
    df["close_target"] = np.nan

    segments = find_valid_segments(valid)
    print(f"找到 {len(segments)} 个 valid 段（山峰）\n")

    for seg_i, (s, e) in enumerate(segments):
        seg_angles = df["angle"].iloc[s:e+1].values

        # 张开端点 = 段内最大值，及其在段内的位置
        peak_local_idx = int(np.nanargmax(seg_angles))
        open_target = float(seg_angles[peak_local_idx])

        # 夹紧端点 = 峰顶之后的最小值
        after_peak = seg_angles[peak_local_idx:]
        close_target = float(np.nanmin(after_peak))

        # 段内每帧打上这两个值
        df.loc[s:e, "open_target"]  = open_target
        df.loc[s:e, "close_target"] = close_target

        peak_frame  = df["frame_index"].iloc[s + peak_local_idx]
        print(f"段 {seg_i+1}: 行 {s+2}~{e+2}  "
              f"张开端点={open_target:.0f}° (frame {peak_frame})  "
              f"夹紧端点={close_target:.0f}°")

    return df


def main():
    rec_dir = os.path.join(BASE_DIR, RECORDING)
    if not os.path.exists(rec_dir):
        print(f"[Error] 文件夹不存在: {rec_dir}")
        sys.exit(1)

    # 读原始CSV（排除已生成的 _clean / _labeled）
    csv_files = [f for f in os.listdir(rec_dir)
                 if f.endswith(".csv") and "clean" not in f and "labeled" not in f]
    if not csv_files:
        print(f"[Error] 没找到原始CSV")
        sys.exit(1)
    csv_path = os.path.join(rec_dir, csv_files[0])
    print(f"[Load] {csv_path}\n")

    df = pd.read_csv(csv_path)
    if "valid" not in df.columns:
        print("[Error] CSV没有valid列")
        sys.exit(1)

    df_labeled = label(df)

    out_path = csv_path.replace(".csv", "_labeled.csv")
    df_labeled.to_csv(out_path, index=False)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()