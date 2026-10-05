"""
record_reach_to_grasp.py

功能：
1. Camera 以约 30 FPS 采集
2. 每一帧直接保存为 JPG
3. 实时检测两个蓝色指尖标记并计算中心点距离
4. IMU 通过 BLE 持续读取原始数据
5. WT9011DCL-BT50 设置为 50 Hz
6. 正确解析 BLE notification 中的多个 IMU 数据包
7. Camera 和 IMU 使用同一个 time.time() 作为时间基准
8. 程序启动后直接开始录制
9. 按 Q 结束整个录制
10. 不使用 valid 字段
11. 录制阶段保存完整 IMU 原始数据
12. 实时低通滤波并对三轴加速度做梯形积分
13. 后处理阶段再将 IMU 50 Hz 重采样到 30 Hz

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

frame,t_camera,finger_width


imu.csv:

imu_index,t_imu,ax,ay,az,mag,vx,vy,vz,velocity
"""

import asyncio
import struct
import threading
import time
import csv
import os

import cv2
import numpy as np
import mediapipe as mp
from bleak import BleakClient

from datetime import datetime


# ============================================================
# 配置
# ============================================================

TRIAL_NAME = f"TRIAL_Black_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 只改这个单词："red"、"black" 或 "blue"
OBJECT_COLOR = "black"

BACKUP_DIR = r"Data\BackupData\D"

OUTPUT_DIR = os.path.join(
    BACKUP_DIR,
    TRIAL_NAME
)

FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEO_PATH = os.path.join(OUTPUT_DIR, "recorded_video.mp4")


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

# 与 Analyze.py 相同的因果低通参数。实时积分没有终点漂移修正。
REALTIME_LPF_ALPHA = 0.15


# ============================================================
# Camera
# ============================================================

CAMERA_INDEX = 0

FRAME_W = 640
FRAME_H = 1000

TARGET_FPS = 30.0

CAP_BACKEND = cv2.CAP_DSHOW


# ============================================================
# MediaPipe Hand Detection
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
THUMB_TIP = 4
INDEX_TIP = 8


# ============================================================
# Object Color Detection
# ============================================================

# OpenCV HSV 范围。红色跨越色相轴首尾，因此需要两个范围。
OBJECT_HSV_RANGES = {
    "red": [
        (
            np.array([0, 160, 50], dtype=np.uint8),
            np.array([8, 255, 255], dtype=np.uint8),
        ),
        (
            np.array([172, 160, 50], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        ),
    ],
    "black": [
        (
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([180, 120, 60], dtype=np.uint8),
        ),
    ],
    "blue": [
        (
            np.array([90, 80, 50], dtype=np.uint8),
            np.array([135, 255, 255], dtype=np.uint8),
        ),
    ],
}

OBJECT_BOX_COLORS = {
    "red": (0, 0, 255),
    "black": (255, 255, 255),
    "blue": (255, 0, 0),
}

# 彩色物体在远处的轮廓较小；黑色需要较高阈值以过滤阴影。
OBJECT_MIN_CONTOUR_AREA = {
    "red": 500.0,
    "black": 3000.0,
    "blue": 800.0,
}

OBJECT_COLOR = OBJECT_COLOR.strip().lower()
if OBJECT_COLOR not in OBJECT_HSV_RANGES:
    raise ValueError(
        f"OBJECT_COLOR must be red, black, or blue; got {OBJECT_COLOR!r}"
    )

MIN_CUBE_AREA = OBJECT_MIN_CONTOUR_AREA[OBJECT_COLOR]
MIN_FILL_RATIO = 0.30
OBJECT_KERNEL = np.ones((5, 5), dtype=np.uint8)

# 红色容易与皮肤重叠，因此额外使用形状和亮度过滤。
RED_MIN_FILL_RATIO = 0.50
RED_MIN_SOLIDITY = 0.75
RED_MIN_ASPECT_RATIO = 0.65
RED_MAX_ASPECT_RATIO = 1.35
RED_MAX_MEAN_VALUE = 130.0


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

        # 实时速度积分状态
        self.filtered_acc = [
            0.0,
            0.0,
            0.0
        ]

        self.previous_filtered_acc = None

        self.realtime_velocity = [
            0.0,
            0.0,
            0.0
        ]

        self.latest_velocity = 0.0

        self.integration_ts = None


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
# MediaPipe Finger Width Detection
# ============================================================

def detect_finger_width(frame, hands):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return 0.0, None, None, None

    lm = result.multi_hand_landmarks[0]
    h, w = frame.shape[:2]
    thumb = lm.landmark[THUMB_TIP]
    index = lm.landmark[INDEX_TIP]
    thumb_point = (int(thumb.x * w), int(thumb.y * h))
    index_point = (int(index.x * w), int(index.y * h))
    finger_width = float(np.linalg.norm(np.array(thumb_point, dtype=float) - np.array(index_point, dtype=float)))
    return finger_width, thumb_point, index_point, lm


# ============================================================
# Selected Color Object Detection
# ============================================================

def detect_cube(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in OBJECT_HSV_RANGES[OBJECT_COLOR]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OBJECT_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, OBJECT_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < MIN_CUBE_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(w * h)
        if bbox_area <= 0:
            continue

        fill_ratio = contour_area / bbox_area
        if fill_ratio < MIN_FILL_RATIO:
            continue

        if OBJECT_COLOR == "red":
            aspect_ratio = w / float(h)
            if not (
                RED_MIN_ASPECT_RATIO
                <= aspect_ratio
                <= RED_MAX_ASPECT_RATIO
            ):
                continue

            if fill_ratio < RED_MIN_FILL_RATIO:
                continue

            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            if hull_area <= 0:
                continue

            solidity = contour_area / hull_area
            if solidity < RED_MIN_SOLIDITY:
                continue

            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            mean_value = cv2.mean(hsv, mask=contour_mask)[2]
            if mean_value > RED_MAX_MEAN_VALUE:
                continue

        candidates.append((bbox_area, x, y, w, h))

    if not candidates:
        return 0.0, None

    bbox_area, x, y, w, h = max(candidates, key=lambda item: item[0])
    return bbox_area, (x, y, w, h)


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

    cap = cv2.VideoCapture(CAMERA_INDEX, CAP_BACKEND)

    if not cap.isOpened():
        print("[Camera] 打不开摄像头")
        state.running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print()
    print(f"[Camera] {actual_w} x {actual_h}")
    print(f"[Camera] reported FPS = {actual_fps:.2f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, TARGET_FPS, (actual_w, actual_h))
    if not video_writer.isOpened():
        print("[Video] 无法创建视频文件")
        cap.release()
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)

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

    camera_writer.writerow(["frame", "t_camera", "finger_width", "object_area"])

    imu_writer.writerow([
        "imu_index",
        "t_imu",
        "ax",
        "ay",
        "az",
        "mag",
        "vx",
        "vy",
        "vz",
        "velocity"
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

                # ------------------------------------------------
                # 实时三轴速度积分
                #
                # 低通滤波和梯形积分与 Analyze.py 的积分前半段一致。
                # 实时阶段不知道动作终点，因此不做终点漂移修正。
                # ------------------------------------------------

                filtered_acc = [
                    REALTIME_LPF_ALPHA * net[i]
                    +
                    (1.0 - REALTIME_LPF_ALPHA) * state.filtered_acc[i]
                    for i in range(3)
                ]

                if (
                    state.integration_ts is not None
                    and
                    state.previous_filtered_acc is not None
                ):

                    dt = ts - state.integration_ts

                    if 0.0 < dt <= 0.2:

                        state.realtime_velocity = [
                            state.realtime_velocity[i]
                            +
                            0.5
                            *
                            (
                                state.previous_filtered_acc[i]
                                +
                                filtered_acc[i]
                            )
                            *
                            dt
                            for i in range(3)
                        ]

                state.filtered_acc = filtered_acc
                state.previous_filtered_acc = filtered_acc.copy()
                state.integration_ts = ts

                vx, vy, vz = state.realtime_velocity

                velocity = (
                    vx ** 2
                    +
                    vy ** 2
                    +
                    vz ** 2
                ) ** 0.5

                state.latest_acc = net

                state.latest_mag = mag

                state.latest_velocity = velocity

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
                    mag,
                    vx,
                    vy,
                    vz,
                    velocity
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
                print("[IMU] 静止校准中...")

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

                    # 从校准结束后的第一个 IMU 样本开始积分。
                    state.filtered_acc = [0.0, 0.0, 0.0]
                    state.previous_filtered_acc = None
                    state.realtime_velocity = [0.0, 0.0, 0.0]
                    state.latest_velocity = 0.0
                    state.integration_ts = None


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


                print("[IMU] Disconnected")


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
            # Blue finger marker detection
            #
            # 不足两个蓝色标记时 finger_width = 0
            # ------------------------------------------------

            finger_width, thumb_point, index_point, hand_landmarks = detect_finger_width(frame, hands)
            object_area, object_box = detect_cube(frame)


            # ------------------------------------------------
            # JPG path
            # ------------------------------------------------

            frame_name = f"frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(FRAME_DIR, frame_name)


            # ------------------------------------------------
            # Camera CSV
            # ------------------------------------------------

            camera_writer.writerow([frame_idx, f"{t_camera:.9f}", f"{finger_width:.3f}", f"{object_area:.3f}"])

            camera_file.flush()


            # ------------------------------------------------
            # Display
            #
            # 这里只显示状态
            # 没有任何推理
            # ------------------------------------------------

            display = frame.copy()

            with state.lock:
                ax, ay, az = state.latest_acc
                mag = state.latest_mag
                velocity = state.latest_velocity

            if hand_landmarks is not None:
                mp_draw.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.circle(display, thumb_point, 8, (255, 0, 0), -1)
                cv2.circle(display, index_point, 8, (0, 0, 255), -1)
                cv2.line(display, thumb_point, index_point, (0, 255, 255), 2)

            if object_box is not None:
                x, y, bw, bh = object_box
                cv2.rectangle(
                    display,
                    (x, y),
                    (x + bw, y + bh),
                    OBJECT_BOX_COLORS[OBJECT_COLOR],
                    2,
                )

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

            cv2.putText(
                display,
                f"ax={ax:.2f} m/s^2",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"ay={ay:.2f} m/s^2",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"az={az:.2f} m/s^2",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"mag={mag:.2f} m/s^2",
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"velocity={velocity:.3f} m/s",
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            finger_color = (0, 255, 255) if finger_width > 0 else (0, 0, 255)
            cv2.putText(display, f"finger_width={finger_width:.1f} px", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, finger_color, 2)

            object_color = (0, 255, 0) if object_area > 0 else (0, 0, 255)
            cv2.putText(
                display,
                f"{OBJECT_COLOR}_area={object_area:.0f} px^2",
                (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                object_color,
                2,
            )

            # 保存已经画好检测结果的 JPG 和视频
            success = cv2.imwrite(frame_path, display)
            if not success:
                print(f"[Camera] JPG 保存失败: {frame_name}")

            video_writer.write(display)
            cv2.imshow("Reach-to-Grasp Recorder", display)


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

        hands.close()
        video_writer.release()
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

        print(f"Frames        : {FRAME_DIR}")
        print(f"Video         : {VIDEO_PATH}")
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

    print("程序启动后将自动开始录制")

    print("按 Q 结束录制")

    print()

    record()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()