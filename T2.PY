import asyncio
import time
from bleak import BleakClient


# ============================================================
# WT9011DCL-BT50
# ============================================================

IMU_ADDRESS = "D8:76:A8:5D:CC:CF"

# BLE characteristic
WRITE_UUID = "0000ffe9-0000-1000-8000-00805f9a34fb"
NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"


# ============================================================
# WT9011DCL commands
# ============================================================

# Unlock configuration
UNLOCK_CMD = bytes([
    0xFF,
    0xAA,
    0x69,
    0x88,
    0xB5
])

# Set output rate
#
# Register:
#   0x03 = RRATE
#
# Value:
#   0x08 = 50 Hz
#
SET_50HZ_CMD = bytes([
    0xFF,
    0xAA,
    0x03,
    0x08,
    0x00
])

# Save configuration
SAVE_CMD = bytes([
    0xFF,
    0xAA,
    0x00,
    0x00,
    0x00
])


# ============================================================
# Helper
# ============================================================

def print_hex(name, data):

    print(
        f"{name}: "
        + " ".join(
            f"{x:02X}" for x in data
        )
    )


# ============================================================
# Step 1
# 设置 50 Hz
# ============================================================

async def set_50hz():

    print()
    print("=" * 70)
    print("WT9011DCL-BT50 - Set Output Rate")
    print("=" * 70)

    print()
    print(f"Device : {IMU_ADDRESS}")
    print(f"WRITE  : {WRITE_UUID}")
    print()

    try:

        async with BleakClient(
            IMU_ADDRESS
        ) as client:

            print(
                f"[BLE] Connected: "
                f"{client.is_connected}"
            )

            # ------------------------------------------------
            # 1. Unlock
            # ------------------------------------------------

            print()
            print("[1/3] Unlock configuration")

            print_hex(
                "TX",
                UNLOCK_CMD
            )

            await client.write_gatt_char(
                WRITE_UUID,
                UNLOCK_CMD,
                response=True
            )

            await asyncio.sleep(0.1)

            # ------------------------------------------------
            # 2. Set 50 Hz
            # ------------------------------------------------

            print()
            print("[2/3] Set output rate = 50 Hz")

            print_hex(
                "TX",
                SET_50HZ_CMD
            )

            await client.write_gatt_char(
                WRITE_UUID,
                SET_50HZ_CMD,
                response=True
            )

            await asyncio.sleep(0.1)

            # ------------------------------------------------
            # 3. Save
            # ------------------------------------------------

            print()
            print("[3/3] Save configuration")

            print_hex(
                "TX",
                SAVE_CMD
            )

            await client.write_gatt_char(
                WRITE_UUID,
                SAVE_CMD,
                response=True
            )

            await asyncio.sleep(0.5)

            print()
            print("=" * 70)
            print("设置命令发送完成")
            print("=" * 70)

            print()
            print("Output rate:")
            print("    50 Hz")

            print()
            print(
                "设备配置已经保存。"
            )

            print()
            print(
                "接下来将重新连接设备，"
                "实际测量 BLE 输出频率。"
            )

            return True

    except Exception as e:

        print()
        print("=" * 70)
        print("设置失败")
        print("=" * 70)

        print(
            f"Error: {e}"
        )

        return False


# ============================================================
# Step 2
# 实际测试 IMU 输出频率
# ============================================================

async def test_frequency():

    print()
    print("=" * 70)
    print("Testing actual IMU output frequency")
    print("=" * 70)

    print()

    sample_count = 0

    timestamps = []

    start_time = None

    test_duration = 5.0

    # --------------------------------------------------------
    # Notification callback
    # --------------------------------------------------------

    def notification_handler(
        sender,
        data
    ):

        nonlocal sample_count
        nonlocal start_time

        # WT9011DCL 普通数据包
        #
        # 你的原程序使用 0x55 0x61
        #

        if len(data) < 2:
            return

        if data[0] != 0x55:
            return

        if data[1] != 0x61:
            return

        ts = time.perf_counter()

        if start_time is None:

            start_time = ts

        timestamps.append(ts)

        sample_count += 1

    try:

        async with BleakClient(
            IMU_ADDRESS
        ) as client:

            print(
                f"[BLE] Connected: "
                f"{client.is_connected}"
            )

            await client.start_notify(
                NOTIFY_UUID,
                notification_handler
            )

            print()
            print(
                f"开始统计 {test_duration:.1f} 秒..."
            )

            # ------------------------------------------------
            # 等待数据
            # ------------------------------------------------

            await asyncio.sleep(
                test_duration
            )

            await client.stop_notify(
                NOTIFY_UUID
            )

            print()
            print("=" * 70)
            print("Frequency Test Result")
            print("=" * 70)

            print()
            print(
                f"Samples received : "
                f"{sample_count}"
            )

            print(
                f"Test duration    : "
                f"{test_duration:.2f} s"
            )

            measured_fps = (
                sample_count /
                test_duration
            )

            print(
                f"Measured rate    : "
                f"{measured_fps:.2f} Hz"
            )

            # ------------------------------------------------
            # 更准确的时间间隔
            # ------------------------------------------------

            if len(timestamps) >= 2:

                intervals = [
                    timestamps[i + 1]
                    - timestamps[i]
                    for i in range(
                        len(timestamps) - 1
                    )
                ]

                mean_interval = (
                    sum(intervals) /
                    len(intervals)
                )

                actual_rate = (
                    1.0 /
                    mean_interval
                )

                print(
                    f"Mean interval    : "
                    f"{mean_interval * 1000:.2f} ms"
                )

                print(
                    f"Interval rate    : "
                    f"{actual_rate:.2f} Hz"
                )

            print()

            # ------------------------------------------------
            # 判断
            # ------------------------------------------------

            if (
                45.0
                <= measured_fps
                <= 55.0
            ):

                print(
                    "✓ IMU 输出频率确认约为 50 Hz"
                )

            else:

                print(
                    "⚠ 实际频率不是 50 Hz"
                )

                print(
                    "请把上面的结果发给我。"
                )

    except Exception as e:

        print()
        print("=" * 70)
        print("频率测试失败")
        print("=" * 70)

        print(
            f"Error: {e}"
        )


# ============================================================
# Main
# ============================================================

async def main():

    print()
    print("=" * 70)
    print("WT9011DCL-BT50 IMU Rate Configuration")
    print("=" * 70)

    print()
    print("目标输出频率: 50 Hz")
    print()

    # --------------------------------------------------------
    # 设置
    # --------------------------------------------------------

    success = await set_50hz()

    if not success:

        print()
        print(
            "设置失败，程序结束。"
        )

        return

    # --------------------------------------------------------
    # 等待设备保存配置
    # --------------------------------------------------------

    print()
    print(
        "等待设备保存配置..."
    )

    await asyncio.sleep(2.0)

    # --------------------------------------------------------
    # 测试
    # --------------------------------------------------------

    await test_frequency()

    print()
    print("=" * 70)
    print("程序结束")
    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())