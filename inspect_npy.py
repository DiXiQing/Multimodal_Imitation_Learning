"""快速查看单个/整段 recording 目录下 npy 文件的内容与统计信息。

用法:
    python inspect_npy.py <npy文件路径>              # 查看单个文件
    python inspect_npy.py <emg目录路径> --all         # 汇总整段统计 + 画图
"""
import argparse
import glob
import os

import numpy as np


def inspect_one(path: str) -> None:
    a = np.load(path)
    print(f"file: {path}")
    print(f"shape: {a.shape}  dtype: {a.dtype}")
    print(f"min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f} nonzero={np.count_nonzero(a)}/{a.size}")
    print(a)


def inspect_dir(d: str) -> None:
    files = sorted(glob.glob(os.path.join(d, "*.npy")))
    print(f"total files: {len(files)}")
    if not files:
        return

    arrs = np.stack([np.load(f) for f in files])  # (N, 100, 4)
    print(f"stacked shape: {arrs.shape}")

    all_zero = [i for i, a in enumerate(arrs) if not np.any(a)]
    print(f"all-zero frames: {len(all_zero)} -> indices: {all_zero[:20]}{'...' if len(all_zero) > 20 else ''}")

    per_channel_mean = arrs.reshape(-1, arrs.shape[-1]).mean(axis=0)
    per_channel_max = arrs.reshape(-1, arrs.shape[-1]).max(axis=0)
    print(f"per-channel mean: {per_channel_mean}")
    print(f"per-channel max:  {per_channel_max}")

    try:
        import matplotlib.pyplot as plt

        flat = arrs.reshape(arrs.shape[0], -1, arrs.shape[-1]).mean(axis=1)  # (N, 4) avg per window
        plt.figure(figsize=(12, 4))
        for ch in range(flat.shape[1]):
            plt.plot(flat[:, ch], label=f"ch{ch}")
        plt.legend()
        plt.title(os.path.basename(d.rstrip("/\\")))
        plt.xlabel("frame index")
        plt.ylabel("mean emg value (per 100-sample window)")
        out = os.path.join(os.path.dirname(d.rstrip("/\\")), "emg_overview.png")
        plt.savefig(out, dpi=150)
        print(f"saved plot: {out}")
    except ImportError:
        print("matplotlib not available, skip plotting")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="npy文件路径,或目录路径(配合 --all)")
    ap.add_argument("--all", action="store_true", help="将 path 视为目录,汇总统计整段数据")
    args = ap.parse_args()

    if args.all:
        inspect_dir(args.path)
    else:
        inspect_one(args.path)


if __name__ == "__main__":
    main()
