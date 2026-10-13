"""
IMUTest.py

WT9011DCL-BT50
Real-Time Acceleration -> Velocity Test

Processing:
    Raw acceleration
          ↓
    Gravity compensation
          ↓
    Bias calibration
          ↓
    Low-pass filter
          ↓
    Trapezoidal integration
          ↓
    Zero-velocity detection / correction
          ↓
    Velocity

操作：
    程序启动
        ↓
    IMU 静止 3 秒进行 calibration
        ↓
    开始实时计算
        ↓
    按 Q 结束
        ↓
    自动保存 CSV
"""

import asyncio
import csv
import os
import time
import threading
from collections import deque

import numpy as np
from bleak import BleakClient

from datetime import datetime


# ============================================================
# Configuration
# ============================================================

DEVICE_ADDRESS = "D8:76:A8:5D:CC:CF"

NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"

OUTPUT_DIR = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData\IMU_Velocity_Test"


OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"velocity_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

# Calibration time
CALIBRATION_TIME = 3.0

# Low-pass filter
LPF_ALPHA = 0.15

# Zero velocity threshold
ZUPT_ACCEL_THRESHOLD = 0.20
ZUPT_VELOCITY_THRESHOLD = 0.15

# Maximum reasonable velocity
MAX_VELOCITY = 5.0

ZUPT_STATIC_DURATION = 0.15


# ============================================================
# Global data
# ============================================================

data_rows = []

running = True

sample_count = 0

previous_time = None

previous_acc = np.zeros(3)

velocity = np.zeros(3)

bias = np.zeros(3)

gravity = np.zeros(3)

filtered_acc = np.zeros(3)

calibration_samples = []

stationary_since = None


# ============================================================
# Keyboard
# ============================================================

def keyboard_listener():

    global running

    while running:

        try:

            key = input()

            if key.strip().lower() == "q":

                print()
                print("[System] Q pressed")

                running = False

                break

        except EOFError:

            break


# ============================================================
# Parse WT9011DCL 0x61 packet
# ============================================================

def parse_packet(packet):

    """
    WT9011DCL 0x61 acceleration packet

    Packet:

    55 61
    AX_L AX_H
    AY_L AY_H
    AZ_L AZ_H
    GX_L GX_H
    GY_L GY_H
    GZ_L GZ_H
    ...

    Acceleration:
        raw / 32768 * 16 g

    Result:
        m/s^2
    """

    if len(packet) < 20:
        return None

    if packet[0] != 0x55:
        return None

    if packet[1] != 0x61:
        return None

    ax_raw = int.from_bytes(
        packet[2:4],
        byteorder="little",
        signed=True
    )

    ay_raw = int.from_bytes(
        packet[4:6],
        byteorder="little",
        signed=True
    )

    az_raw = int.from_bytes(
        packet[6:8],
        byteorder="little",
        signed=True
    )

    # ±16 g
    ax_g = ax_raw / 32768.0 * 16.0
    ay_g = ay_raw / 32768.0 * 16.0
    az_g = az_raw / 32768.0 * 16.0

    # g -> m/s²
    G = 9.80665

    acceleration = np.array([
        ax_g * G,
        ay_g * G,
        az_g * G
    ])

    return acceleration


# ============================================================
# Extract packets from BLE notification
# ============================================================

def extract_packets(data):

    packets = []

    i = 0

    while i <= len(data) - 20:

        if data[i] == 0x55 and data[i + 1] == 0x61:

            packet = data[i:i + 20]

            packets.append(packet)

            i += 20

        else:

            i += 1

    return packets


# ============================================================
# Gravity / Bias calibration
# ============================================================

def calibration_phase():

    global bias
    global gravity

    if len(calibration_samples) < 10:

        return

    samples = np.array(
        calibration_samples
    )

    mean_acc = np.mean(
        samples,
        axis=0
    )

    print()
    print("=" * 70)
    print("Calibration Result")
    print("=" * 70)

    print(
        f"Mean acceleration:"
    )

    print(
        f"X = {mean_acc[0]: .4f} m/s²"
    )

    print(
        f"Y = {mean_acc[1]: .4f} m/s²"
    )

    print(
        f"Z = {mean_acc[2]: .4f} m/s²"
    )

    print()

    # Gravity vector is the measured static acceleration
    gravity = mean_acc.copy()

    # For a static IMU, measured acceleration consists
    # mainly of gravity. Therefore the dynamic acceleration
    # is obtained by subtracting the static gravity vector.

    print(
        f"Gravity vector:"
    )

    print(
        f"X = {gravity[0]: .4f}"
    )

    print(
        f"Y = {gravity[1]: .4f}"
    )

    print(
        f"Z = {gravity[2]: .4f}"
    )

    print()


# ============================================================
# Low pass filter
# ============================================================

def low_pass_filter(acc):

    global filtered_acc

    filtered_acc = (
        LPF_ALPHA * acc
        +
        (1.0 - LPF_ALPHA) * filtered_acc
    )

    return filtered_acc.copy()


# ============================================================
# Zero velocity detection
# ============================================================

def is_stationary(acc, current_time):

    global stationary_since

    acc_norm = np.linalg.norm(acc)

    if acc_norm < ZUPT_ACCEL_THRESHOLD:
        if stationary_since is None:
            stationary_since = current_time
        duration = current_time - stationary_since
        return duration >= ZUPT_STATIC_DURATION
    else:
        stationary_since = None
        return False

# ============================================================
# Save CSV
# ============================================================

def save_csv():

    if len(data_rows) == 0:

        print(
            "[Warning] 没有测试数据"
        )

        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "time",
            "dt",

            "ax_raw",
            "ay_raw",
            "az_raw",

            "ax_corrected",
            "ay_corrected",
            "az_corrected",

            "ax_filtered",
            "ay_filtered",
            "az_filtered",

            "vx",
            "vy",
            "vz",

            "speed",

            "stationary"
        ])

        writer.writerows(
            data_rows
        )

    print()
    print(
        f"[Saved] CSV:"
    )
    print(
        OUTPUT_FILE
    )

    print()
    print(
        f"Samples saved: "
        f"{len(data_rows)}"
    )


# ============================================================
# Process one acceleration sample
# ============================================================

def process_sample(acc):

    global previous_time
    global previous_acc
    global velocity
    global sample_count

    current_time = time.perf_counter()

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    if (
        current_time - calibration_start
        <
        CALIBRATION_TIME
    ):

        calibration_samples.append(
            acc.copy()
        )

        return

    # First sample after calibration
    if previous_time is None:

        previous_time = current_time

        previous_acc = acc.copy()

        calibration_phase()

        print()
        print("=" * 70)
        print("Real-Time Velocity Calculation")
        print("=" * 70)

        return

    # --------------------------------------------------------
    # dt
    # --------------------------------------------------------

    dt = (
        current_time
        -
        previous_time
    )

    previous_time = current_time

    # Protect against abnormal dt
    if dt <= 0 or dt > 0.2:

        return

    # --------------------------------------------------------
    # Gravity compensation
    # --------------------------------------------------------

    corrected_acc = (
        acc - gravity
    )

    # --------------------------------------------------------
    # Low-pass filter
    # --------------------------------------------------------

    acc_filtered = low_pass_filter(
        corrected_acc
    )

    # --------------------------------------------------------
    # Trapezoidal integration
    # --------------------------------------------------------

    velocity = velocity + (
        previous_acc
        +
        acc_filtered
    ) * 0.5 * dt

    previous_acc = acc_filtered.copy()

    # --------------------------------------------------------
    # Zero Velocity Detection
    # --------------------------------------------------------

    stationary = is_stationary(acc_filtered,current_time)

    if stationary:

        velocity[:] = 0.0

    # --------------------------------------------------------
    # Safety protection
    # --------------------------------------------------------

    speed = np.linalg.norm(
        velocity
    )

    if speed > MAX_VELOCITY:

        velocity *= (
            MAX_VELOCITY / speed
        )

        speed = MAX_VELOCITY

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    elapsed = (
        current_time
        -
        measurement_start
    )

    sample_count += 1

    data_rows.append([

        f"{elapsed:.6f}",

        f"{dt:.6f}",

        f"{acc[0]:.6f}",
        f"{acc[1]:.6f}",
        f"{acc[2]:.6f}",

        f"{corrected_acc[0]:.6f}",
        f"{corrected_acc[1]:.6f}",
        f"{corrected_acc[2]:.6f}",

        f"{acc_filtered[0]:.6f}",
        f"{acc_filtered[1]:.6f}",
        f"{acc_filtered[2]:.6f}",

        f"{velocity[0]:.6f}",
        f"{velocity[1]:.6f}",
        f"{velocity[2]:.6f}",

        f"{speed:.6f}",

        int(stationary)
    ])

    # --------------------------------------------------------
    # Real-time display
    # --------------------------------------------------------

    print(
        "\r"
        f"ACC "
        f"X:{acc_filtered[0]:7.3f} "
        f"Y:{acc_filtered[1]:7.3f} "
        f"Z:{acc_filtered[2]:7.3f}"
        " | "
        f"VEL "
        f"X:{velocity[0]:7.3f} "
        f"Y:{velocity[1]:7.3f} "
        f"Z:{velocity[2]:7.3f}"
        " | "
        f"SPEED:{speed:7.3f} m/s"
        " | "
        f"N:{sample_count}",
        end="",
        flush=True
    )


# ============================================================
# BLE callback
# ============================================================

def notification_handler(
    sender,
    data
):

    packets = extract_packets(
        bytes(data)
    )

    for packet in packets:

        acc = parse_packet(
            packet
        )

        if acc is not None:

            process_sample(
                acc
            )


# ============================================================
# Main
# ============================================================

async def main():

    global running
    global calibration_start
    global measurement_start

    print()
    print("=" * 70)
    print("WT9011DCL-BT50 Real-Time Velocity Test")
    print("=" * 70)

    print()

    print(
        f"Device : {DEVICE_ADDRESS}"
    )

    print(
        f"Notify : {NOTIFY_UUID}"
    )

    print(
        f"Output : {OUTPUT_FILE}"
    )

    print()

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    try:

        async with BleakClient(
            DEVICE_ADDRESS
        ) as client:

            print(
                f"[BLE] Connected: "
                f"{client.is_connected}"
            )

            # ------------------------------------------------
            # Start notification
            # ------------------------------------------------

            await client.start_notify(
                NOTIFY_UUID,
                notification_handler
            )

            print()
            print("=" * 70)
            print("Calibration")
            print("=" * 70)

            print()
            print(
                "请保持 IMU 静止 "
                f"{CALIBRATION_TIME:.0f} 秒"
            )

            print(
                "不要移动 IMU..."
            )

            calibration_start = (
                time.perf_counter()
            )

            # Keyboard thread
            threading.Thread(
                target=keyboard_listener,
                daemon=True
            ).start()

            # ------------------------------------------------
            # Wait calibration
            # ------------------------------------------------

            while (
                running
                and
                time.perf_counter()
                -
                calibration_start
                <
                CALIBRATION_TIME
            ):

                await asyncio.sleep(
                    0.01
                )

            if not running:

                return

            calibration_phase()

            # ------------------------------------------------
            # Reset
            # ------------------------------------------------

            previous_time = None

            measurement_start = (
                time.perf_counter()
            )

            print()
            print("=" * 70)
            print("开始实时速度测试")
            print("=" * 70)

            print()
            print(
                "现在可以移动 IMU"
            )

            print(
                "输入 Q 后回车结束测试"
            )

            print()

            # ------------------------------------------------
            # Measurement
            # ------------------------------------------------

            while running:

                await asyncio.sleep(
                    0.01
                )

            # ------------------------------------------------
            # Stop notify
            # ------------------------------------------------

            try:

                await client.stop_notify(
                    NOTIFY_UUID
                )

            except Exception:

                pass

    except Exception as e:

        print()
        print(
            f"[BLE] Error: {e}"
        )

    finally:

        running = False

        print()
        print()
        print("=" * 70)
        print("保存测试数据")
        print("=" * 70)

        save_csv()

        print()
        print("=" * 70)
        print("程序结束")
        print("=" * 70)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except Exception as e:
        import traceback
        print()
        print(f"[BLE] Error: {type(e).__name__}: {e}")
        traceback.print_exc()