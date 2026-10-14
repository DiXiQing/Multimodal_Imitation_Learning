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
11. 姿态逐帧去重力，保存线性加速度、原始加速度与姿态
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
import ctypes

import cv2
import numpy as np
import mediapipe as mp
from bleak import BleakClient, BleakScanner
from imu_processing import gravity_in_sensor, parse_imu_packet, linear_acceleration, calibration_quality

from datetime import datetime


# ============================================================
# 配置
# ============================================================

TRIAL_NAME = f"TRIAL_Size_3.5cm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

# BLE 连接失败时自动重试，避免一次超时直接终止整次录制
IMU_MAX_CONNECT_ATTEMPTS = 3
IMU_RETRY_DELAY_SECONDS = 3.0
IMU_SCAN_TIMEOUT_SECONDS = 10.0
IMU_CONNECT_TIMEOUT_SECONDS = 30.0

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
CALIB_MAX_WAIT_SECONDS = 20.0

# 与 Analyze.py 相同的因果低通参数。实时积分没有终点漂移修正。
REALTIME_LPF_ALPHA = 0.15


# ============================================================
# Camera
# ============================================================

CAMERA_INDEX = 1

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

# 小尺寸物体在远处轮廓较小；黑色2cm方块也需要低于1000px²的阈值。
OBJECT_MIN_CONTOUR_AREA = {
    "red": 500.0,
    "black": 500.0,
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
        self.calib_linear_buffer = []
        self.calib_gyro_buffer = []
        self.calib_timestamps = []

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

    返回：原始加速度(g)、角速度(°/s)、姿态角(°)的字典。
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

    return parse_imu_packet(packet)


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

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
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
        "velocity",
        "ax_raw_g", "ay_raw_g", "az_raw_g",
        "gx_dps", "gy_dps", "gz_dps",
        "roll_deg", "pitch_deg", "yaw_deg",
        "gravity_x_g", "gravity_y_g", "gravity_z_g"
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


        for packet_index, packet in enumerate(
            packets
        ):
            raw = packet['acc']
            angles = packet['angles']
            gravity = gravity_in_sensor(angles)

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
                    state.calib_linear_buffer.append(raw - gravity)
                    state.calib_gyro_buffer.append(packet['gyro'].copy())
                    state.calib_timestamps.append(ts)
                    while len(state.calib_timestamps) > 1 and ts - state.calib_timestamps[0] > 2.1:
                        state.calib_timestamps.pop(0)
                        state.calib_buffer.pop(0)
                        state.calib_linear_buffer.pop(0)
                        state.calib_gyro_buffer.pop(0)

                    continue

                # ------------------------------------------------
                # Bias correction
                # ------------------------------------------------

                # 与 IMUTest 相同：逐帧重力投影 + 初始两秒残余传感器零偏。
                net = linear_acceleration(packet, state.acc_bias, GRAVITY).tolist()

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
                    velocity,
                    *raw.tolist(),
                    *packet['gyro'].tolist(),
                    *angles.tolist(),
                    *gravity.tolist()
                ])

                imu_counter += 1


    # ========================================================
    # IMU Async Task
    # ========================================================

    async def imu_record_once():

        try:
            print(f"[IMU] 扫描设备 {IMU_ADDRESS}（最多 {IMU_SCAN_TIMEOUT_SECONDS:g} 秒）...")
            # 每次重试重新扫描，传入当前广播对应的 BLEDevice，避免隐式扫描和旧句柄。
            device = await BleakScanner.find_device_by_address(
                IMU_ADDRESS, timeout=IMU_SCAN_TIMEOUT_SECONDS
            )
            if device is None:
                raise ConnectionError("扫描未发现 IMU；检查设备电源及是否被其他程序占用")
            print(f"[IMU] 已发现 {device.name or IMU_ADDRESS}，开始连接并读取服务...")

            async with BleakClient(
                device,
                timeout=IMU_CONNECT_TIMEOUT_SECONDS,
                winrt={"use_cached_services": False} if os.name == "nt" else {},
            ) as client:

                print()
                print(
                    f"[IMU] Connected: "
                    f"{client.is_connected}"
                )

                # ------------------------------------------------
                # Start notify
                # ------------------------------------------------
                characteristic = client.services.get_characteristic(IMU_NOTIFY_UUID)
                if characteristic is None:
                    available = [c.uuid for service in client.services for c in service.characteristics]
                    raise ConnectionError(
                        f"连接成功但未发现 IMU 通知特征 {IMU_NOTIFY_UUID}；"
                        f"实际服务特征：{available}"
                    )
                if not {"notify", "indicate"}.intersection(characteristic.properties):
                    raise ConnectionError("IMU 特征不支持数据通知，不能开始录制")

                await client.start_notify(
                    characteristic,
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
                    state.calib_linear_buffer = []
                    state.calib_gyro_buffer = []
                    state.calib_timestamps = []


                await asyncio.sleep(
                    CALIB_SECONDS
                )

                calibration_started = time.monotonic() - CALIB_SECONDS
                last_reason = None
                while state.running:
                    if not client.is_connected:
                        raise ConnectionError("IMU 在校准期间断开")
                    with state.lock:
                        good, reason = calibration_quality(
                            state.calib_buffer, state.calib_linear_buffer,
                            state.calib_gyro_buffer, state.calib_timestamps,
                        )
                        if good:
                            # 接受样本和释放录制必须在同一个锁内完成。
                            state.acc_bias = np.mean(state.calib_linear_buffer, axis=0).tolist()
                            state.filtered_acc = [0.0, 0.0, 0.0]
                            state.previous_filtered_acc = None
                            state.realtime_velocity = [0.0, 0.0, 0.0]
                            state.latest_velocity = 0.0
                            state.integration_ts = None
                            state.calibrating = False
                    if good:
                        break
                    if reason != last_reason:
                        print(f"[IMU] 校准尚未合格，继续保持静止：{reason}")
                        last_reason = reason
                    if time.monotonic() - calibration_started >= CALIB_MAX_WAIT_SECONDS:
                        raise RuntimeError(f"静置校准等待 20 秒仍未合格：{reason}")
                    await asyncio.sleep(0.05)
                if not state.running:
                    return


                # ------------------------------------------------
                # Calculate bias
                # ------------------------------------------------

                print()
                print(
                    "[IMU] 校准完成"
                )

                print(
                    "[IMU] 去重力后的传感器零偏 = "
                    f"{[f'{v:.5f}' for v in state.acc_bias]}"
                    " g"
                )



                # ------------------------------------------------
                # Keep connection
                # ------------------------------------------------

                while state.running:
                    if not client.is_connected:
                        with state.lock:
                            state.running = False
                        raise ConnectionError("录制期间 IMU 已断开；停止录制以避免保存缺失 IMU 的帧")

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

            print(
                f"[IMU] 本次连接失败: "
                f"type={type(e).__name__}, "
                f"repr={e!r}"
            )

            # 把异常交给外层重试逻辑处理。重试期间保持 calibrating=True，
            # 避免主线程在两次连接之间提前开始录制。
            raise


    # ========================================================
    # IMU Async Task with automatic retries
    # ========================================================

    async def imu_record_task():

        for attempt in range(1, IMU_MAX_CONNECT_ATTEMPTS + 1):

            if not state.running:
                return

            # 重试期间主线程等待校准；最终失败时终止整个录制。
            with state.lock:
                state.calibrating = True

            if attempt > 1:
                print()
                print(
                    f"[IMU] 第 {attempt}/{IMU_MAX_CONNECT_ATTEMPTS} 次重试连接..."
                )

            try:
                await imu_record_once()
                return

            except Exception as e:
                if not state.running:
                    return
                print(
                    f"[IMU] 连接/运行失败（第 {attempt}/"
                    f"{IMU_MAX_CONNECT_ATTEMPTS} 次）："
                    f"{type(e).__name__}: {e}"
                )

                if attempt < IMU_MAX_CONNECT_ATTEMPTS:
                    print(
                        f"[IMU] {IMU_RETRY_DELAY_SECONDS:.1f} 秒后自动重试"
                    )
                    await asyncio.sleep(IMU_RETRY_DELAY_SECONDS)
                else:
                    with state.lock:
                        state.running = False

                    print()
                    print(
                        f"[IMU] {IMU_MAX_CONNECT_ATTEMPTS} 次连接/运行均失败，"
                        "终止整个录制。"
                    )
                    return


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

        hands.close()

        video_writer.release()

        cap.release()

        cv2.destroyAllWindows()

        imu_thread.join(timeout=3)

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
    print("按 Q 结束录制（Windows 下无需点击画面窗口）")
    print()


    # ========================================================
    # Camera Loop
    # ========================================================

    frame_idx = 0

    # Windows global key state; retain OpenCV key handling on other systems.
    get_key_state = None
    if os.name == "nt":
        get_key_state = ctypes.WinDLL("user32", use_last_error=True).GetAsyncKeyState
        get_key_state.argtypes = [ctypes.c_int]
        get_key_state.restype = ctypes.c_short
    # Ignore a Q held before recording starts until it has been released.
    global_q_armed = get_key_state is None or not (get_key_state(0x51) & 0x8000)

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

            # Keep an untouched camera image for saved frame files.
            raw_frame = frame.copy()


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

            # JPG保存原始相机图像；视频和预览保留检测结果及状态信息。
            success = cv2.imwrite(frame_path, raw_frame)
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

            global_q_pressed = False
            if get_key_state is not None:
                q_down = bool(get_key_state(0x51) & 0x8000)
                if not q_down:
                    global_q_armed = True
                global_q_pressed = global_q_armed and q_down

            if key in (ord("q"), ord("Q")) or global_q_pressed:

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
