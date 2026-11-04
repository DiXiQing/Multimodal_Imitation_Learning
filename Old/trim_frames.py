"""
trim_frames.py

直接裁剪原始 Reach-to-Grasp 数据。

功能：
1. 删除指定 frame 之前的数据
2. 删除指定 frame 之后的数据
3. 直接删除原始 JPG
4. 直接更新原始 camera.csv
5. 根据保留下来的 Camera 时间范围裁剪 imu.csv
6. 不重新编号 frame
7. 原始数据直接修改

例如：

START_FRAME = 682
END_FRAME = 1210

删除：
    frame <= 682
    frame >= 1210

保留：
    683 ~ 1209
"""

import os
import csv


# ============================================================
# 配置
# ============================================================

TRIAL_DIR = (
    r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData\TRIAL_20260829_003924"
)

# 删除 frame <= START_FRAME
START_FRAME = 10

# 删除 frame >= END_FRAME
# 如果不删除后面的 frame，设置为 None
END_FRAME = 150


# ============================================================
# 文件路径
# ============================================================

FRAME_DIR = os.path.join(
    TRIAL_DIR,
    "frames"
)

CAMERA_CSV = os.path.join(
    TRIAL_DIR,
    "camera.csv"
)

IMU_CSV = os.path.join(
    TRIAL_DIR,
    "imu.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("Trim Original Reach-to-Grasp Data")
    print("=" * 70)

    print()
    print(f"Trial:")
    print(TRIAL_DIR)

    print()

    print(
        f"删除 frame <= {START_FRAME}"
    )

    if END_FRAME is not None:

        print(
            f"删除 frame >= {END_FRAME}"
        )

    else:

        print(
            "不删除后面的 frame"
        )

    print()


    # ========================================================
    # 检查文件
    # ========================================================

    if not os.path.exists(TRIAL_DIR):

        print(
            "[Error] Trial 文件夹不存在："
        )

        print(
            TRIAL_DIR
        )

        return


    if not os.path.exists(FRAME_DIR):

        print(
            "[Error] frames 文件夹不存在："
        )

        print(
            FRAME_DIR
        )

        return


    if not os.path.exists(CAMERA_CSV):

        print(
            "[Error] camera.csv 不存在："
        )

        print(
            CAMERA_CSV
        )

        return


    # ========================================================
    # 读取 camera.csv
    # ========================================================

    camera_data = []

    with open(
        CAMERA_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        if fieldnames is None:

            print(
                "[Error] camera.csv 没有表头。"
            )

            return


        if "frame" not in fieldnames:

            print(
                "[Error] camera.csv 中没有 frame 列。"
            )

            return


        if "t_camera" not in fieldnames:

            print(
                "[Error] camera.csv 中没有 t_camera 列。"
            )

            return


        for row in reader:

            try:

                frame = int(
                    row["frame"]
                )

                camera_data.append(row)

            except Exception:

                print(
                    "[Warning] 跳过异常 Camera 数据："
                )

                print(row)


    if len(camera_data) == 0:

        print(
            "[Error] camera.csv 没有有效数据。"
        )

        return


    # ========================================================
    # 显示实际 frame 范围
    # ========================================================

    all_frames = [
        int(row["frame"])
        for row in camera_data
    ]

    min_frame = min(all_frames)
    max_frame = max(all_frames)


    print()
    print("=" * 70)
    print("Camera CSV 实际范围")
    print("=" * 70)

    print()

    print(
        f"最小 frame : {min_frame}"
    )

    print(
        f"最大 frame : {max_frame}"
    )

    print(
        f"Camera 总行数 : {len(camera_data)}"
    )


    # ========================================================
    # 判断保留数据
    # ========================================================

    keep_camera = []
    delete_camera = []


    for row in camera_data:

        frame = int(
            row["frame"]
        )

        delete = False


        # 删除前面的 frame

        if frame <= START_FRAME:

            delete = True


        # 删除后面的 frame

        if (
            END_FRAME is not None
            and
            frame >= END_FRAME
        ):

            delete = True


        if delete:

            delete_camera.append(row)

        else:

            keep_camera.append(row)


    # ========================================================
    # 如果没有保留数据，停止
    # ========================================================

    if len(keep_camera) == 0:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print()

        print(
            "没有任何 Camera frame 会被保留。"
        )

        print()

        print(
            "你的设置是："
        )

        print(
            f"START_FRAME = {START_FRAME}"
        )

        print(
            f"END_FRAME   = {END_FRAME}"
        )

        print()

        print(
            "实际 Camera frame 范围："
        )

        print(
            f"{min_frame} ~ {max_frame}"
        )

        print()

        print(
            "为了安全，程序没有删除任何数据。"
        )

        return


    # ========================================================
    # 保留范围
    # ========================================================

    keep_frames = [
        int(row["frame"])
        for row in keep_camera
    ]

    first_keep_frame = min(
        keep_frames
    )

    last_keep_frame = max(
        keep_frames
    )


    print()
    print("=" * 70)
    print("将要进行的操作")
    print("=" * 70)

    print()

    print(
        f"删除 Camera rows : "
        f"{len(delete_camera)}"
    )

    print(
        f"保留 Camera rows : "
        f"{len(keep_camera)}"
    )

    print()

    print(
        f"保留 frame 范围 : "
        f"{first_keep_frame} ~ {last_keep_frame}"
    )


    # ========================================================
    # 获取 Camera 时间范围
    # ========================================================

    t_start = float(
        keep_camera[0]["t_camera"]
    )

    t_end = float(
        keep_camera[-1]["t_camera"]
    )


    print()

    print(
        f"保留 Camera 时间:"
    )

    print(
        f"{t_start:.9f}"
        f" -> "
        f"{t_end:.9f}"
    )


    # ========================================================
    # 读取 IMU
    # ========================================================

    imu_data = []

    imu_fieldnames = None


    if os.path.exists(IMU_CSV):

        with open(
            IMU_CSV,
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            imu_fieldnames = (
                reader.fieldnames
            )


            if (
                imu_fieldnames is not None
                and
                "t_imu" in imu_fieldnames
            ):

                for row in reader:

                    try:

                        t_imu = float(
                            row["t_imu"]
                        )

                        imu_data.append(
                            row
                        )

                    except Exception:

                        print(
                            "[Warning] "
                            "跳过异常 IMU 数据："
                        )

                        print(row)


    # ========================================================
    # 计算 IMU 保留范围
    # ========================================================

    keep_imu = []
    delete_imu = []


    for row in imu_data:

        t_imu = float(
            row["t_imu"]
        )


        if (
            t_imu >= t_start
            and
            t_imu <= t_end
        ):

            keep_imu.append(row)

        else:

            delete_imu.append(row)


    print()

    print(
        f"原始 IMU samples : "
        f"{len(imu_data)}"
    )

    print(
        f"删除 IMU samples : "
        f"{len(delete_imu)}"
    )

    print(
        f"保留 IMU samples : "
        f"{len(keep_imu)}"
    )


    # ========================================================
    # 最后一次确认
    # ========================================================

    print()
    print("=" * 70)
    print("⚠️ 即将直接删除原始数据")
    print("=" * 70)

    print()

    print(
        f"删除 JPG : {len(delete_camera)}"
    )

    print(
        f"保留 JPG : {len(keep_camera)}"
    )

    print(
        f"删除 IMU : {len(delete_imu)}"
    )

    print(
        f"保留 IMU : {len(keep_imu)}"
    )

    print()

    print(
        "camera.csv 和 imu.csv 也会被直接修改。"
    )

    print(
        "此操作无法自动恢复。"
    )

    print()

    # ========================================================
    # 删除 JPG
    # ========================================================

    print()
    print(
        "开始删除 JPG..."
    )


    deleted_jpg = 0


    for row in delete_camera:

        frame = int(
            row["frame"]
        )

        filename = (
            f"frame_{frame:06d}.jpg"
        )

        filepath = os.path.join(
            FRAME_DIR,
            filename
        )


        if os.path.exists(filepath):

            try:

                os.remove(filepath)

                deleted_jpg += 1

            except Exception as e:

                print(
                    f"[Error] 删除失败："
                    f"{filename}"
                )

                print(e)

        else:

            print(
                f"[Warning] JPG 不存在："
                f"{filename}"
            )


    print()

    print(
        f"JPG 删除完成："
        f"{deleted_jpg}"
    )


    # ========================================================
    # 更新 Camera CSV
    # ========================================================

    print()

    print(
        "更新 camera.csv..."
    )


    temp_camera_csv = (
        CAMERA_CSV + ".tmp"
    )


    with open(
        temp_camera_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in keep_camera:

            writer.writerow(row)


    os.replace(
        temp_camera_csv,
        CAMERA_CSV
    )


    print(
        "camera.csv 更新完成。"
    )


    # ========================================================
    # 更新 IMU CSV
    # ========================================================

    if (
        imu_fieldnames is not None
        and
        len(imu_data) > 0
    ):

        print()

        print(
            "更新 imu.csv..."
        )


        temp_imu_csv = (
            IMU_CSV + ".tmp"
        )


        with open(
            temp_imu_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=imu_fieldnames
            )

            writer.writeheader()

            for row in keep_imu:

                writer.writerow(row)


        os.replace(
            temp_imu_csv,
            IMU_CSV
        )


        print(
            "imu.csv 更新完成。"
        )


    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 70)
    print("裁剪完成")
    print("=" * 70)

    print()

    print(
        f"Camera frame:"
    )

    print(
        f"  删除 : {deleted_jpg}"
    )

    print(
        f"  保留 : {len(keep_camera)}"
    )

    print()

    print(
        f"IMU:"
    )

    print(
        f"  删除 : {len(delete_imu)}"
    )

    print(
        f"  保留 : {len(keep_imu)}"
    )

    print()

    print(
        f"最终 Camera frame:"
    )

    print(
        f"{first_keep_frame} ~ {last_keep_frame}"
    )

    print()

    print(
        "原始数据已经直接修改。"
    )

    print(
        "Frame 编号没有重新排序。"
    )

    print()


if __name__ == "__main__":

    main()