"""
analyze_grasp_frames.py

功能：
1. 从 frames/ 读取 Camera JPG
2. 从 camera.csv 读取 frame + t_camera
3. MediaPipe 检测手部
4. 使用拇指尖(4)和食指尖(8)计算 Finger Width
5. 使用黑色区域检测 Object Area
6. 生成带标注的视频
7. 生成 grasp_data.csv
8. 生成 Area + Finger Width 曲线
9. 生成 Object Area vs Finger Width 散点图

当前数据结构：

TRIAL_NAME/
├── frames/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── camera.csv
└── imu.csv

注意：
imu.csv 本版本暂不参与处理。
下一步再进行 IMU 50 Hz -> Camera 30 Hz 的时间同步/重采样。
"""

import os
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp


# ============================================================
# 配置
# ============================================================

TRIAL_DIR = (
    r"C:\MineApp\Code\Multimodal_Imitation_Learning"
    r"\Data\BackupData\large_trial3"
)

FRAME_DIR = os.path.join(
    TRIAL_DIR,
    "frames"
)

CAMERA_CSV = os.path.join(
    TRIAL_DIR,
    "camera.csv"
)


# ============================================================
# Camera FPS
# ============================================================

TARGET_FPS = 30.0


# ============================================================
# 黑色物体检测
# ============================================================

BLACK_LOWER = np.array([
    0,
    0,
    0
])

BLACK_UPPER = np.array([
    180,
    255,
    80
])

MIN_CUBE_AREA = 3000


# ============================================================
# MediaPipe
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

K = np.ones(
    (5, 5),
    np.uint8
)

THUMB_TIP = 4
INDEX_TIP = 8


# ============================================================
# Detect black object
# ============================================================

def detect_cube(frame):
    """
    检测黑色物体。

    返回：
        area : object area
        box  : (x, y, w, h)

    如果没有检测到：
        area = 0
        box = None
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    black = cv2.inRange(
        hsv,
        BLACK_LOWER,
        BLACK_UPPER
    )

    # 去除小噪声
    black = cv2.morphologyEx(
        black,
        cv2.MORPH_OPEN,
        K
    )

    # 填补区域
    black = cv2.morphologyEx(
        black,
        cv2.MORPH_CLOSE,
        K
    )

    contours, _ = cv2.findContours(
        black,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0, None

    # 找最大黑色区域
    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        contour
    )

    if area < MIN_CUBE_AREA:
        return 0, None

    box = cv2.boundingRect(
        contour
    )

    return area, box


# ============================================================
# Load Camera CSV
# ============================================================

def load_camera_csv():

    if not os.path.exists(
        CAMERA_CSV
    ):

        print(
            f"[Error] 找不到 camera.csv:"
        )

        print(
            CAMERA_CSV
        )

        return None


    camera_data = []


    with open(
        CAMERA_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:

                frame_idx = int(
                    row["frame"]
                )

                t_camera = float(
                    row["t_camera"]
                )

                camera_data.append({
                    "frame": frame_idx,
                    "t_camera": t_camera
                })

            except Exception as e:

                print(
                    f"[Warning] camera.csv "
                    f"数据读取失败: {row}"
                )

                print(e)


    return camera_data


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("Reach-to-Grasp Frame Post-Processing")
    print("=" * 70)

    print()

    print(
        f"Trial directory:"
    )

    print(
        TRIAL_DIR
    )

    print()

    print(
        f"Frames:"
    )

    print(
        FRAME_DIR
    )

    print()


    # ========================================================
    # Check folders
    # ========================================================

    if not os.path.exists(
        FRAME_DIR
    ):

        print(
            "[Error] 找不到 frames 文件夹"
        )

        return


    # ========================================================
    # Load camera CSV
    # ========================================================

    camera_data = load_camera_csv()

    if camera_data is None:

        return


    if len(camera_data) == 0:

        print(
            "[Error] camera.csv 没有数据"
        )

        return


    print(
        f"[Camera CSV] "
        f"{len(camera_data)} frames"
    )


    # ========================================================
    # Find first frame
    # ========================================================

    first_frame_idx = (
        camera_data[0]["frame"]
    )

    first_frame_path = os.path.join(
        FRAME_DIR,
        f"frame_{first_frame_idx:06d}.jpg"
    )

    first_frame = cv2.imread(
        first_frame_path
    )

    if first_frame is None:

        print(
            "[Error] 无法读取第一帧:"
        )

        print(
            first_frame_path
        )

        return


    h, w = first_frame.shape[:2]


    print(
        f"[Frame] "
        f"{w} x {h}"
    )


    # ========================================================
    # Output paths
    # ========================================================

    output_video = os.path.join(
        TRIAL_DIR,
        "analyzed_video.mp4"
    )

    output_csv = os.path.join(
        TRIAL_DIR,
        "grasp_data.csv"
    )

    output_plot = os.path.join(
        TRIAL_DIR,
        "grasp_curve.png"
    )

    output_scatter = os.path.join(
        TRIAL_DIR,
        "area_vs_finger_width.png"
    )


    # ========================================================
    # Video Writer
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    video_writer = cv2.VideoWriter(
        output_video,
        fourcc,
        TARGET_FPS,
        (w, h)
    )


    if not video_writer.isOpened():

        print(
            "[Error] 无法创建输出视频"
        )

        return


    # ========================================================
    # MediaPipe
    # ========================================================

    hands = mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=1,

        min_detection_confidence=0.3,

        min_tracking_confidence=0.3
    )


    # ========================================================
    # Result data
    # ========================================================

    result_data = []

    valid_frames = []

    areas = []

    finger_widths = []

    hand_detected_count = 0

    object_detected_count = 0


    # ========================================================
    # Processing
    # ========================================================

    total_frames = len(
        camera_data
    )


    print()
    print("=" * 70)
    print("开始后处理")
    print("=" * 70)
    print()


    for process_idx, row in enumerate(
        camera_data
    ):

        frame_idx = row["frame"]

        t_camera = row["t_camera"]


        # ----------------------------------------------------
        # Frame path
        # ----------------------------------------------------

        frame_path = os.path.join(
            FRAME_DIR,
            f"frame_{frame_idx:06d}.jpg"
        )


        frame = cv2.imread(
            frame_path
        )


        if frame is None:

            print(
                f"[Warning] 无法读取:"
                f" frame_{frame_idx:06d}.jpg"
            )

            continue


        vis = frame.copy()


        # ----------------------------------------------------
        # Object detection
        # ----------------------------------------------------

        area, box = detect_cube(
            frame
        )


        if area > 0:

            object_detected_count += 1


        # ----------------------------------------------------
        # Hand detection
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        result = hands.process(
            rgb
        )


        finger_width = 0.0


        if result.multi_hand_landmarks:

            hand_detected_count += 1


            lm = (
                result.multi_hand_landmarks[0]
            )


            thumb = (
                lm.landmark[
                    THUMB_TIP
                ]
            )


            index = (
                lm.landmark[
                    INDEX_TIP
                ]
            )


            thumb_point = np.array([
                thumb.x * w,
                thumb.y * h
            ])


            index_point = np.array([
                index.x * w,
                index.y * h
            ])


            finger_width = float(
                np.linalg.norm(
                    thumb_point
                    -
                    index_point
                )
            )


            # ------------------------------------------------
            # Draw hand landmarks
            # ------------------------------------------------

            mp_draw.draw_landmarks(

                vis,

                lm,

                mp_hands.HAND_CONNECTIONS
            )


            # ------------------------------------------------
            # Draw thumb
            # ------------------------------------------------

            cv2.circle(

                vis,

                tuple(
                    thumb_point.astype(int)
                ),

                8,

                (255, 0, 0),

                -1
            )


            # ------------------------------------------------
            # Draw index
            # ------------------------------------------------

            cv2.circle(

                vis,

                tuple(
                    index_point.astype(int)
                ),

                8,

                (0, 0, 255),

                -1
            )


            # ------------------------------------------------
            # Draw finger width line
            # ------------------------------------------------

            cv2.line(

                vis,

                tuple(
                    thumb_point.astype(int)
                ),

                tuple(
                    index_point.astype(int)
                ),

                (0, 255, 255),

                2
            )


        # ----------------------------------------------------
        # Draw object
        # ----------------------------------------------------

        if box is not None:

            x, y, bw, bh = box


            cv2.rectangle(

                vis,

                (x, y),

                (
                    x + bw,
                    y + bh
                ),

                (0, 0, 255),

                2
            )


            cv2.putText(

                vis,

                f"Object area: {area:.0f} px^2",

                (
                    x,
                    max(25, y - 10)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 0, 255),

                2
            )


        # ====================================================
        # Text information
        # ====================================================

        cv2.putText(

            vis,

            f"Frame: {frame_idx}",

            (10, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )

        # ----------------------------------------------------
        # Finger Width
        # ----------------------------------------------------

        if finger_width > 0:

            finger_text = (
                f"Finger width: "
                f"{finger_width:.1f} px"
            )

        else:

            finger_text = (
                "Finger width: N/A"
            )


        cv2.putText(

            vis,

            finger_text,

            (10, 85),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 200, 255),

            1
        )


        # ----------------------------------------------------
        # Object Area
        # ----------------------------------------------------

        if area > 0:

            area_text = (
                f"Object area: "
                f"{area:.0f} px^2"
            )

        else:

            area_text = (
                "Object area: N/A"
            )


        cv2.putText(

            vis,

            area_text,

            (10, 115),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 0),

            2
        )


        # ====================================================
        # Write video
        # ====================================================

        video_writer.write(
            vis
        )


        # ====================================================
        # Save data
        # ====================================================

        result_data.append({

            "frame":
                frame_idx,

            "t_camera":
                t_camera,

            "finger_width":
                finger_width,

            "object_area":
                area
        })


        # ----------------------------------------------------
        # Only use frames where both detected
        # ----------------------------------------------------

        if (
            finger_width > 0
            and
            area > 0
        ):

            valid_frames.append(
                frame_idx
            )

            areas.append(
                area
            )

            finger_widths.append(
                finger_width
            )


        # ====================================================
        # Progress
        # ====================================================

        if (
            process_idx % 30 == 0
            or
            process_idx == total_frames - 1
        ):

            progress = (
                process_idx + 1
            ) / total_frames * 100


            print(
                f"\rProcessing: "
                f"{process_idx + 1}/"
                f"{total_frames} "
                f"({progress:.1f}%)",
                end=""
            )


    print()
    print()


    # ========================================================
    # Release
    # ========================================================

    video_writer.release()

    hands.close()


    # ========================================================
    # Save CSV
    # ========================================================

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow([
            "frame",
            "t_camera",
            "finger_width",
            "object_area"
        ])


        for row in result_data:

            writer.writerow([

                row["frame"],

                f'{row["t_camera"]:.9f}',

                (
                    f'{row["finger_width"]:.3f}'
                    if row["finger_width"] > 0
                    else ""
                ),

                (
                    f'{row["object_area"]:.3f}'
                    if row["object_area"] > 0
                    else ""
                )
            ])


    # ========================================================
    # Print statistics
    # ========================================================

    print()
    print("=" * 70)
    print("处理完成")
    print("=" * 70)

    print()

    print(
        f"总 Camera frames       : "
        f"{total_frames}"
    )

    print(
        f"检测到手的 frames      : "
        f"{hand_detected_count}"
    )

    print(
        f"检测到物体的 frames    : "
        f"{object_detected_count}"
    )

    print(
        f"手 + 物体同时检测      : "
        f"{len(valid_frames)}"
    )

    if total_frames > 0:

        print()

        print(
            f"Hand detection rate    : "
            f"{hand_detected_count / total_frames * 100:.1f}%"
        )

        print(
            f"Object detection rate  : "
            f"{object_detected_count / total_frames * 100:.1f}%"
        )

        print(
            f"Both detection rate    : "
            f"{len(valid_frames) / total_frames * 100:.1f}%"
        )


    # ========================================================
    # Save curve
    # ========================================================

    if len(valid_frames) >= 2:

        valid_frames_np = np.array(
            valid_frames
        )

        areas_np = np.array(
            areas
        )

        finger_widths_np = np.array(
            finger_widths
        )


        # ====================================================
        # Figure 1
        # ====================================================

        fig, ax1 = plt.subplots(
            figsize=(14, 6)
        )


        ax1.plot(

            valid_frames_np,

            areas_np,

            "r-o",

            markersize=3,

            label="Object Area"
        )


        ax1.set_xlabel(
            "Frame"
        )

        ax1.set_ylabel(
            "Object Area (px²)"
        )


        ax2 = ax1.twinx()


        ax2.plot(

            valid_frames_np,

            finger_widths_np,

            "b-o",

            markersize=3,

            label="Finger Width"
        )


        ax2.set_ylabel(
            "Finger Width (px)"
        )


        ax1.set_title(
            "Object Area and Finger Width over Time"
        )


        ax1.grid(
            True,
            alpha=0.3
        )


        fig.tight_layout()


        plt.savefig(
            output_plot,
            dpi=150
        )


        plt.close(
            fig
        )


        # ====================================================
        # Figure 2
        # ====================================================

        fig2, ax = plt.subplots(
            figsize=(8, 6)
        )


        scatter = ax.scatter(

            areas_np,

            finger_widths_np,

            c=valid_frames_np,

            cmap="viridis",

            s=30
        )


        ax.set_xlabel(
            "Object Area (px²)"
        )

        ax.set_ylabel(
            "Finger Width (px)"
        )


        ax.set_title(
            "Object Area vs Finger Width"
        )


        ax.grid(
            True,
            alpha=0.3
        )


        fig2.colorbar(
            scatter,
            ax=ax,
            label="Frame"
        )


        fig2.tight_layout()


        plt.savefig(
            output_scatter,
            dpi=150
        )


        plt.close(
            fig2
        )


        print()

        print(
            f"[Saved] 曲线:"
        )

        print(
            output_plot
        )

        print()

        print(
            f"[Saved] Area vs Finger Width:"
        )

        print(
            output_scatter
        )


    else:

        print()

        print(
            "[Warning] 同时检测到手和物体的有效帧太少"
        )

        print(
            "无法生成可靠的曲线。"
        )


    # ========================================================
    # Final output
    # ========================================================

    print()

    print(
        f"[Saved] 视频:"
    )

    print(
        output_video
    )

    print()

    print(
        f"[Saved] 数据:"
    )

    print(
        output_csv
    )

    print()

    print("=" * 70)
    print("全部完成")
    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()