"""
record_reach_to_grasp.py

功能：
1. Camera 以约 30 FPS 采集
2. 每一帧直接保存为 JPG
3. 不进行实时推理
4. IMU 通过 BLE 持续读取原始数据
5. WT9011DCL-BT50 设置为 50 Hz
6. 正确解析 BLE notification 中的多个 IMU 数据包
7. Camera 和 IMU 使用同一个 time.time() 作为时间基准
8. 程序启动后直接开始录制
9. 按 Q 结束整个录制
10. 不使用 valid 字段
11. 录制阶段保存完整 IMU 原始数据
12. 后处理阶段再将 IMU 50 Hz 重采样到 30 Hz

输出：

TRIAL_NAME/
├── frames/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── camera.csv
└── imu.csv


camera.csv:

frame,t_camera


imu.csv:

imu_index,t_imu,ax,ay,az,mag
"""

import asyncio
import struct
import threading
import time
import csv
import os

import cv2
import numpy as np
from bleak import BleakClient


# ============================================================
# 配置
# ============================================================

TRIAL_NAME = "large_trial3"

BACKUP_DIR = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData"

OUTPUT_DIR = os.path.join(
    BACKUP_DIR,
    TRIAL_NAME
)

FRAME_DIR = os.path.join(
    OUTPUT_DIR,
    "frames"
)


# ============================================================
# IMU
# ============================================================

IMU_ADDRESS = "D8:76:A8:5D:CC:CF"

# WT9011DCL-BT50
#
# FFE9:
#   Write characteristic
#
# FFE4:
#   Notify characteristic
#

IMU_WRITE_UUID = (
    "0000ffe9-0000-1000-8000-00805f9a34fb"
)

IMU_NOTIFY_UUID = (
    "0000ffe4-0000-1000-8000-00805f9a34fb"
)

GRAVITY = 9.81


# ============================================================
# IMU Calibration
# ============================================================

CALIB_SECONDS = 2.0


# ============================================================
# Camera
# ============================================================

CAMERA_INDEX = 1

FRAME_W = 640
FRAME_H = 480

TARGET_FPS = 30.0

CAP_BACKEND = cv2.CAP_DSHOW


# ============================================================
# Shared State
# ============================================================

class SharedState:

    def __init__(self):

        self.lock = threading.Lock()

        # 整个程序是否运行
        self.running = True

        # IMU calibration
        self.calibrating = True

        self.calib_buffer = []

        self.acc_bias = [
            0.0,
            0.0,
            0.0
        ]

        # 最新 IMU 状态
        self.latest_acc = [
            0.0,
            0.0,
            0.0
        ]

        self.latest_mag = 0.0

        self.latest_imu_ts = None


state = SharedState()


# ============================================================
# IMU Packet Parser
# ============================================================

def parse_witmotion_packet(packet):

    """
    解析一个完整的 WT9011DCL 20-byte 数据包。

    packet:
        20 bytes

    格式：

        55 61
        ...
        ...

    返回：
        ax, ay, az
        单位：g
    """

    if len(packet) != 20:

        return None

    if packet[0] != 0x55:

        return None

    if packet[1] != 0x61:

        return None

    # 读取 9 个 int16
    vals = struct.unpack(
        "<9h",
        packet[2:20]
    )

    ax_raw = vals[0]
    ay_raw = vals[1]
    az_raw = vals[2]

    # WT9011DCL 加速度 ±16 g
    ax = (
        ax_raw /
        32768.0 *
        16.0
    )

    ay = (
        ay_raw /
        32768.0 *
        16.0
    )

    az = (
        az_raw /
        32768.0 *
        16.0
    )

    return [
        ax,
        ay,
        az
    ]


# ============================================================
# 从 BLE notification 中提取所有 IMU packets
# ============================================================

def extract_imu_packets(data):

    """
    一次 BLE notification 可能包含多个
    20-byte WT9011D 数据包。

    例如：

        40 bytes

        ┌───────────────┐
        │ packet 1 20 B │
        ├───────────────┤
        │ packet 2 20 B │
        └───────────────┘

    返回：

        [
            [ax, ay, az],
            [ax, ay, az]
        ]
    """

    packets = []

    data_len = len(data)

    i = 0

    while i <= data_len - 20:

        # 找 packet header
        if (
            data[i] == 0x55
            and data[i + 1] == 0x61
        ):

            packet = data[
                i:i + 20
            ]

            parsed = (
                parse_witmotion_packet(
                    packet
                )
            )

            if parsed is not None:

                packets.append(
                    parsed
                )

                i += 20

                continue

        i += 1

    return packets


# ============================================================
# Main
# ============================================================

def record():

    # ========================================================
    # 创建目录
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        FRAME_DIR,
        exist_ok=True
    )

    # ========================================================
    # Camera
    # ========================================================

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        CAP_BACKEND
    )

    if not cap.isOpened():

        print(
            "[Camera] 打不开摄像头"
        )

        state.running = False

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_W
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_H
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        TARGET_FPS
    )

    actual_w = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_h = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print()
    print(
        f"[Camera] "
        f"{actual_w} x {actual_h}"
    )

    print(
        f"[Camera] reported FPS = "
        f"{actual_fps:.2f}"
    )


    # ========================================================
    # CSV
    # ========================================================

    camera_csv_path = os.path.join(
        OUTPUT_DIR,
        "camera.csv"
    )

    imu_csv_path = os.path.join(
        OUTPUT_DIR,
        "imu.csv"
    )

    camera_file = open(
        camera_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    imu_file = open(
        imu_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    camera_writer = csv.writer(
        camera_file
    )

    imu_writer = csv.writer(
        imu_file
    )


    # ========================================================
    # CSV Header
    # ========================================================

    camera_writer.writerow([
        "frame",
        "t_camera"
    ])

    imu_writer.writerow([
        "imu_index",
        "t_imu",
        "ax",
        "ay",
        "az",
        "mag"
    ])


    # ========================================================
    # IMU records
    # ========================================================

    imu_records = []

    imu_records_lock = (
        threading.Lock()
    )

    imu_counter = 0


    # ========================================================
    # IMU Notification Handler
    # ========================================================

    def recording_imu_handler(
        sender,
        data
    ):

        nonlocal imu_counter

        # ----------------------------------------------------
        # 找出这次 notification 中的
        # 所有 20-byte packets
        # ----------------------------------------------------

        packets = (
            extract_imu_packets(
                data
            )
        )

        if len(packets) == 0:

            return


        # ----------------------------------------------------
        # 当前 notification 的接收时间
        #
        # 使用 time.time()
        #
        # 与 Camera 完全相同的时间基准
        # ----------------------------------------------------

        notification_time = (
            time.time()
        )


        # ----------------------------------------------------
        # WT9011DCL = 50 Hz
        #
        # 每个 sample 间隔：
        #
        # 1 / 50 = 0.02 s
        #
        # 如果一次 notification 包含两个 sample：
        #
        # sample 1 = notification_time
        # sample 2 = notification_time + 0.02
        #
        # 注意：
        # 这是对 BLE 合包情况下的时间重建。
        # ----------------------------------------------------

        sample_interval = (
            1.0 / 50.0
        )


        for packet_index, raw in enumerate(
            packets
        ):

            # ------------------------------------------------
            # 时间戳
            # ------------------------------------------------

            if len(packets) == 1:

                ts = notification_time

            else:

                # 第一个 packet
                # 使用当前 notification 时间
                #
                # 第二个 packet
                # 往后推 20 ms

                ts = (
                    notification_time
                    +
                    packet_index
                    *
                    sample_interval
                )


            # =================================================
            # Calibration
            # =================================================

            with state.lock:

                if state.calibrating:

                    state.calib_buffer.append(
                        raw
                    )

                    continue

                # ------------------------------------------------
                # Bias correction
                # ------------------------------------------------

                net = [
                    (
                        raw[i]
                        -
                        state.acc_bias[i]
                    )
                    *
                    GRAVITY

                    for i in range(3)
                ]

                mag = (
                    net[0] ** 2
                    +
                    net[1] ** 2
                    +
                    net[2] ** 2
                ) ** 0.5

                state.latest_acc = net

                state.latest_mag = mag

                state.latest_imu_ts = ts


            # =================================================
            # 保存 IMU
            # =================================================

            with imu_records_lock:

                imu_records.append([
                    imu_counter,
                    ts,
                    net[0],
                    net[1],
                    net[2],
                    mag
                ])

                imu_counter += 1


    # ========================================================
    # IMU Async Task
    # ========================================================

    async def imu_record_task():

        try:

            async with BleakClient(
                IMU_ADDRESS
            ) as client:

                print()
                print(
                    f"[IMU] Connected: "
                    f"{client.is_connected}"
                )

                # ------------------------------------------------
                # Start notify
                # ------------------------------------------------

                await client.start_notify(
                    IMU_NOTIFY_UUID,
                    recording_imu_handler
                )

                # ------------------------------------------------
                # Calibration
                # ------------------------------------------------

                print()
                print(
                    "[IMU] 静止校准中..."
                )

                print(
                    f"[IMU] 请保持设备静止 "
                    f"{CALIB_SECONDS:.1f} 秒"
                )

                with state.lock:

                    state.calibrating = True

                    state.calib_buffer = []


                await asyncio.sleep(
                    CALIB_SECONDS
                )


                # ------------------------------------------------
                # Calculate bias
                # ------------------------------------------------

                with state.lock:

                    if (
                        len(
                            state.calib_buffer
                        )
                        > 0
                    ):

                        buf = np.array(
                            state.calib_buffer
                        )

                        state.acc_bias = (
                            buf.mean(
                                axis=0
                            )
                            .tolist()
                        )

                    else:

                        state.acc_bias = [
                            0.0,
                            0.0,
                            0.0
                        ]

                    state.calibrating = False


                print()
                print(
                    "[IMU] 校准完成"
                )

                print(
                    "[IMU] bias = "
                    f"{[f'{v:.5f}' for v in state.acc_bias]}"
                    " g"
                )


                # ------------------------------------------------
                # Keep connection
                # ------------------------------------------------

                while state.running:

                    await asyncio.sleep(
                        0.05
                    )


                # ------------------------------------------------
                # Stop notification
                # ------------------------------------------------

                try:

                    await client.stop_notify(
                        IMU_NOTIFY_UUID
                    )

                except Exception:

                    pass


                print(
                    "[IMU] Disconnected"
                )


        except Exception as e:

            print()
            print(
                f"[IMU] Error: {e}"
            )

            state.running = False


    # ========================================================
    # Start IMU Thread
    # ========================================================

    state.running = True

    imu_thread = threading.Thread(
        target=lambda:
            asyncio.run(
                imu_record_task()
            ),
        daemon=True
    )

    imu_thread.start()


    # ========================================================
    # Wait for calibration
    # ========================================================

    while True:

        with state.lock:

            calibrating = (
                state.calibrating
            )

            running = (
                state.running
            )

        if not running:

            break

        if not calibrating:

            break

        time.sleep(
            0.05
        )


    # ========================================================
    # Check IMU
    # ========================================================

    if not state.running:

        cap.release()

        camera_file.close()

        imu_file.close()

        return


    # ========================================================
    # Recording starts automatically
    # ========================================================

    print()
    print("=" * 70)
    print("录制开始")
    print("=" * 70)

    print()
    print("Camera : ~30 FPS")
    print("IMU    : ~50 Hz")
    print()
    print("录制过程中不进行任何推理")
    print()
    print("按 Q 结束录制")
    print()


    # ========================================================
    # Camera Loop
    # ========================================================

    frame_idx = 0

    frame_interval = (
        1.0 /
        TARGET_FPS
    )

    next_frame_time = (
        time.perf_counter()
    )


    try:

        while state.running:

            # ------------------------------------------------
            # Frame timing
            # ------------------------------------------------

            now_perf = (
                time.perf_counter()
            )

            sleep_time = (
                next_frame_time
                -
                now_perf
            )

            if sleep_time > 0:

                time.sleep(
                    sleep_time
                )

            next_frame_time += (
                frame_interval
            )


            # ------------------------------------------------
            # Read Camera
            # ------------------------------------------------

            ret, frame = (
                cap.read()
            )

            if not ret:

                print(
                    "[Camera] 读取帧失败"
                )

                break


            # ------------------------------------------------
            # Camera timestamp
            #
            # 与 IMU 使用同一个：
            #
            # time.time()
            # ------------------------------------------------

            t_camera = (
                time.time()
            )


            # ------------------------------------------------
            # Save JPG
            # ------------------------------------------------

            frame_name = (
                f"frame_{frame_idx:06d}.jpg"
            )

            frame_path = os.path.join(
                FRAME_DIR,
                frame_name
            )

            success = cv2.imwrite(
                frame_path,
                frame
            )

            if not success:

                print(
                    f"[Camera] JPG 保存失败: "
                    f"{frame_name}"
                )


            # ------------------------------------------------
            # Camera CSV
            # ------------------------------------------------

            camera_writer.writerow([
                frame_idx,
                f"{t_camera:.9f}"
            ])

            camera_file.flush()


            # ------------------------------------------------
            # Display
            #
            # 这里只显示状态
            # 没有任何推理
            # ------------------------------------------------

            display = frame.copy()

            cv2.putText(
                display,
                "REC",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                display,
                f"frame={frame_idx}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                "Press Q to stop",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Reach-to-Grasp Recorder",
                display
            )


            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                print()
                print(
                    "[System] Q pressed"
                )

                print(
                    "[System] 结束录制"
                )

                state.running = False

                break


            frame_idx += 1


    except KeyboardInterrupt:

        print()
        print(
            "[System] Ctrl+C"
        )

        state.running = False


    finally:

        # ====================================================
        # Stop
        # ====================================================

        state.running = False


        # ====================================================
        # Camera cleanup
        # ====================================================

        cap.release()

        cv2.destroyAllWindows()


        # ====================================================
        # Wait IMU
        # ====================================================

        print()
        print(
            "[System] 等待 IMU 线程结束..."
        )

        imu_thread.join(
            timeout=3
        )


        # ====================================================
        # Save IMU CSV
        # ====================================================

        with imu_records_lock:

            total_imu = len(
                imu_records
            )

            print()
            print(
                f"[IMU] 保存 "
                f"{total_imu} 条数据..."
            )

            for row in imu_records:

                imu_writer.writerow(row)


        # ====================================================
        # Close CSV
        # ====================================================

        camera_file.close()

        imu_file.close()


        # ====================================================
        # Final statistics
        # ====================================================

        print()
        print("=" * 70)
        print("录制完成")
        print("=" * 70)

        print()

        print(
            f"Camera frames : "
            f"{frame_idx}"
        )

        print(
            f"IMU samples   : "
            f"{total_imu}"
        )

        print()

        print(
            f"Camera CSV    : "
            f"{camera_csv_path}"
        )

        print(
            f"IMU CSV       : "
            f"{imu_csv_path}"
        )

        print(
            f"Frames        : "
            f"{FRAME_DIR}"
        )

        print()

        print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("Reach-to-Grasp Recorder")
    print("=" * 70)

    print()
    print(
        f"Trial : {TRIAL_NAME}"
    )

    print(
        f"Camera target FPS : "
        f"{TARGET_FPS}"
    )

    print(
        "IMU hardware rate : 50 Hz"
    )

    print()

    print(
        "程序启动后将自动开始录制"
    )

    print(
        "按 Q 结束录制"
    )

    print()

    record()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()