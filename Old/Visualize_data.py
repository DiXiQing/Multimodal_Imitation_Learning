"""
plot_combined.py
把 角度曲线 + 4通道EMG曲线 + OPEN/CLOSE/KEEP 状态 + valid段 放进一张图里对比看。

和 plot_labels.py 用法一致：改下面 RECORDING 后直接跑。
    python plot_combined.py

依赖同一份 data_*.csv + emg/*.npy （Record.py 的输出结构）。

四个子图（共享 x 轴 = frame_index）：
  1. Angle（灰线）+ valid段（绿色背景）
  2. EMG 4通道曲线（取每帧100ms窗口的均值，和 Record.py 里
     activity/OPEN-CLOSE 判断用的是同一个量，方便对照）
  3. State 轨道：按 Record.py 同样的规则（activity<0.18→KEEP，
     否则 ch2>ch3→OPEN，否则→CLOSE）逐帧分类，画成色块条
     （绿=OPEN，红=CLOSE，灰=KEEP）
  4. Valid flag（细色块条，和你现在 segments.png 的做法一样）
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TrainingData"

# ─────── 改这里 ───────
RECORDING = "recording_20260705_184225"
ACTIVITY_THRESHOLD = 0.18   # 和 Record.py 里的阈值保持一致
# ──────────────────────

EMG_CHANNEL_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
STATE_COLORS = {"OPEN": "#2ecc71", "CLOSE": "#e74c3c", "KEEP": "#bdbdbd"}


def find_valid_segments(valid):
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


def find_state_segments(states):
    """把逐帧的 state 列表压缩成 (state, start_idx, end_idx) 的连续段"""
    segments = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            segments.append((states[start], start, i - 1))
            start = i
    return segments


def load_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    def pf(v):
        v = (v or "").strip()
        return float(v) if v else np.nan

    frame_index = np.array([int(float(r["frame_index"])) for r in rows])
    angle = np.array([pf(r.get("angle", "")) for r in rows])
    valid = np.array([int(float(r["valid"])) if r.get("valid", "") not in ("", None) else 0 for r in rows])
    emg_file = [r.get("emg_file", "") or "" for r in rows]

    order = np.argsort(frame_index)
    return (frame_index[order], angle[order], valid[order],
            [emg_file[i] for i in order])


def load_emg_means(rec_dir, emg_files):
    """逐帧读取 emg_xxxxxx.npy (100x4)，取窗口内每通道均值，
    和 Record.py 里 ch_mean = np.mean(emg_snap, axis=0) 一致"""
    import time
    emg_dir = os.path.join(rec_dir, "emg")
    n = len(emg_files)
    means = np.zeros((n, 4), dtype=np.float32)
    t0 = time.time()
    for i, fname in enumerate(emg_files):
        if not fname:
            continue
        p = os.path.join(emg_dir, fname)
        if not os.path.exists(p):
            continue
        arr = np.load(p)
        means[i] = np.mean(arr, axis=0)
        if i > 0 and i % 500 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n - i) / rate if rate > 0 else float("nan")
            print(f"  [EMG] {i}/{n}  ({rate:.0f} files/s, ETA {eta:.0f}s)")
    print(f"  [EMG] done: {n} files in {time.time()-t0:.1f}s")
    return means


def classify_states(emg_means, activity_threshold):
    """逐帧套用 Record.py 的判断逻辑（不含5帧防抖，方便看原始波动）"""
    activity = emg_means.sum(axis=1)
    ch2, ch3 = emg_means[:, 1], emg_means[:, 2]
    states = []
    for a, c2, c3 in zip(activity, ch2, ch3):
        if a < activity_threshold:
            states.append("KEEP")
        elif c2 > c3:
            states.append("OPEN")
        else:
            states.append("CLOSE")
    return states


def main():
    rec_dir = os.path.join(BASE_DIR, RECORDING)
    csvs = [f for f in os.listdir(rec_dir) if f.startswith("data_") and f.endswith(".csv")]
    if not csvs:
        print("[Error] 没找到 data_*.csv")
        sys.exit(1)
    csv_path = os.path.join(rec_dir, sorted(csvs)[0])
    print(f"[Load] {csv_path}")

    frame_idx, angle, valid, emg_files = load_csv(csv_path)
    emg_means = load_emg_means(rec_dir, emg_files)
    states = classify_states(emg_means, ACTIVITY_THRESHOLD)

    valid_segments = find_valid_segments(valid)

    STATE_TO_NUM = {"CLOSE": -1, "KEEP": 0, "OPEN": 1}
    state_num = np.array([STATE_TO_NUM[s] for s in states])

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1]},
    )
    fig.suptitle(f"{RECORDING}", fontsize=13)

    def add_valid_shading(ax):
        for s, e in valid_segments:
            ax.axvspan(frame_idx[s], frame_idx[e], alpha=0.12, color="green")

    # ── 1. Angle ──
    ax = axes[0]
    ax.plot(frame_idx, angle, color="gray", linewidth=1, label="Angle")
    add_valid_shading(ax)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Servo Angle  (green = valid segment)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 2. EMG 4通道 ──
    ax = axes[1]
    for ch in range(4):
        ax.plot(frame_idx, emg_means[:, ch], color=EMG_CHANNEL_COLORS[ch],
                linewidth=0.8, label=f"CH{ch+1}")
    ax.axhline(ACTIVITY_THRESHOLD / 4, color="black", linewidth=0.6,
               linestyle="--", alpha=0.4, label="ref level")
    add_valid_shading(ax)
    ax.set_ylabel("EMG window mean (mV)")
    ax.set_title("EMG 4 Channels (per-frame 100ms window mean)")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 3. State 折线 ──
    ax = axes[2]
    ax.step(frame_idx, state_num, where="post", color="black", linewidth=1)
    add_valid_shading(ax)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["CLOSE", "KEEP", "OPEN"])
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("State (raw per-frame, no debounce)", fontsize=9)
    ax.set_xlabel("Frame Index")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = csv_path.replace(".csv", "_combined.png")
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] {save_path}")
    plt.show()


if __name__ == "__main__":
    main()