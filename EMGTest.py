"""
test_emg_device.py

单独测试 BITalino EMG 设备是否正常工作,不涉及摄像头/Arduino/模型。

流程和 inference.py 里的 emg_reader() 保持一致(同样的采样率/通道/滤波方式),
方便你确认"设备本身没问题"之后,能直接放心地把 inference.py 跑起来。

用法:
    python test_emg_device.py
    (贴好电极、保持放松几秒 -> 再故意做几次张开/握拳动作 -> 输入 q 回车结束)

结束后会:
    1. 保存原始数据 CSV (方便回看)
    2. 画一张4通道波形图,肉眼确认: 静息时是不是接近0、动作时是不是有明显峰值、
       4个通道是不是都有反应(而不是有的通道全程一条直线)
"""

import time
import threading
import csv
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from bitalino import BITalino

from emg_interface.funcs import adc_to_mV, setup_realtime_envelop_filter, realtime_filter

# ─────────── 配置 ───────────
BITALINO_MAC = "COM4"      # 和 inference.py 保持一致
EMG_SAMPLING = 1000
EMG_CHANNELS = 4
EMG_CHUNK    = 10
EMG_ACQ_CH   = [1, 2, 3, 4]

OUTPUT_DIR = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData\EMG_Device_Test"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, f"emg_test_{TS}.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, f"emg_test_{TS}.png")
# ────────────────────────────

running = True
records = []   # 每项: [t, raw_mv_ch1..4, envelope_ch1..4]


def keyboard_listener():
    global running
    while running:
        try:
            key = input()
            if key.strip().lower() == "q":
                print("\n[System] Q pressed, 停止采集...")
                running = False
                break
        except EOFError:
            break


def save_and_plot():
    if not records:
        print("[Warning] 没有采集到数据")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── CSV ──
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time"] +
            [f"raw_mv_ch{c}" for c in range(1, EMG_CHANNELS + 1)] +
            [f"envelope_ch{c}" for c in range(1, EMG_CHANNELS + 1)]
        )
        writer.writerows(records)
    print(f"[Saved] CSV → {OUTPUT_CSV}  ({len(records)} 个采样点)")

    # ── 画图: 4通道 raw + envelope 叠加,各自一个子图 ──
    arr = np.array(records, dtype=float)
    t = arr[:, 0]

    fig, axs = plt.subplots(EMG_CHANNELS, 1, figsize=(11, 9), sharex=True)
    for ch in range(EMG_CHANNELS):
        raw_col = 1 + ch
        env_col = 1 + EMG_CHANNELS + ch
        axs[ch].plot(t, arr[:, raw_col], color="tab:blue", linewidth=0.6, alpha=0.6, label="raw (mV)")
        axs[ch].plot(t, arr[:, env_col], color="tab:red", linewidth=1.5, label="envelope")
        axs[ch].set_ylabel(f"CH{ch+1}")
        axs[ch].grid(alpha=0.3)
        if ch == 0:
            axs[ch].legend(loc="upper right")
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("EMG Device Test — raw (blue) vs envelope (red)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"[Saved] 波形图 → {OUTPUT_PLOT}")

    # ── 简单的"设备是否正常"自检提示 ──
    print("\n[自检提示] 每个通道的 envelope 幅值范围(动作时应明显大于静息噪声):")
    for ch in range(EMG_CHANNELS):
        env_col = 1 + EMG_CHANNELS + ch
        col = arr[:, env_col]
        print(f"  CH{ch+1}: min={col.min():.4f}  max={col.max():.4f}  "
              f"peak/baseline ratio≈{(col.max() / (col.min()+1e-6)):.1f}x")
    print("  → 如果某个通道 min/max 几乎相等(一条直线),说明该通道大概率没接触好或电极有问题。")


def main():
    global running

    print("=" * 70)
    print("BITalino EMG 设备测试")
    print("=" * 70)
    print(f"Port: {BITALINO_MAC}   Sampling: {EMG_SAMPLING} Hz   Channels: {EMG_ACQ_CH}")
    print()

    try:
        device = BITalino(BITALINO_MAC)
        device.start(EMG_SAMPLING, EMG_ACQ_CH)
        print("[EMG] 已连接,开始采集")
    except Exception as e:
        print(f"[Error] 连接失败: {e}")
        return

    b, a, z_list = setup_realtime_envelop_filter(1, fs=EMG_SAMPLING)
    z = [z_list.copy() for _ in range(EMG_CHANNELS)]

    threading.Thread(target=keyboard_listener, daemon=True).start()
    print("先保持放松几秒,再做几次张开/握拳动作,输入 q 回车结束\n")

    t0 = time.time()
    try:
        while running:
            raw = device.read(EMG_CHUNK)[:, 5:]           # 去掉前5列(序号/数字IO)
            raw_mv = adc_to_mV(raw)
            envelope = np.abs(raw_mv).copy()
            for ch in range(EMG_CHANNELS):
                envelope[:, ch], z[ch] = realtime_filter(envelope[:, ch], b, z[ch], a)

            now = time.time() - t0
            # chunk 里每个采样点都记录一行(用近似时间戳,按chunk内均匀分布)
            for i in range(EMG_CHUNK):
                t_i = now - (EMG_CHUNK - 1 - i) / EMG_SAMPLING
                records.append([t_i] + raw_mv[i].tolist() + envelope[i].tolist())

            rms_str = "  ".join(
                f"CH{c+1}:{np.sqrt(np.mean(envelope[:, c]**2)):.3f}" for c in range(EMG_CHANNELS))
            print(f"\r{rms_str}", end="", flush=True)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[Error] 采集中出错: {e}")

    finally:
        running = False
        try:
            device.stop()
            device.close()
        except Exception:
            pass
        print("\n[EMG] 已断开")
        save_and_plot()


if __name__ == "__main__":
    main()