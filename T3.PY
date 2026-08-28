import asyncio
import time
from bleak import BleakClient


# ============================================================
# WT9011DCL-BT50
# ============================================================

IMU_ADDRESS = "D8:76:A8:5D:CC:CF"

NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"


# ============================================================
# Statistics
# ============================================================

notification_count = 0
packet_count = 0

start_time = None


# ============================================================
# Notification handler
# ============================================================

def notification_handler(sender, data):

    global notification_count
    global packet_count
    global start_time

    notification_count += 1

    if start_time is None:
        start_time = time.perf_counter()

    # --------------------------------------------------------
    # 打印前 20 条 notification
    # --------------------------------------------------------

    if notification_count <= 20:

        print()
        print(
            f"Notification #{notification_count}"
        )

        print(
            f"Length = {len(data)} bytes"
        )

        print(
            "HEX:"
        )

        print(
            " ".join(
                f"{x:02X}"
                for x in data
            )
        )

    # --------------------------------------------------------
    # 查找所有 0x55 61
    # --------------------------------------------------------

    i = 0

    packets_in_this_notification = 0

    while i <= len(data) - 2:

        if (
            data[i] == 0x55
            and data[i + 1] == 0x61
        ):

            packets_in_this_notification += 1

            packet_count += 1

            i += 20

        else:

            i += 1

    # --------------------------------------------------------
    # 前 20 条显示解析结果
    # --------------------------------------------------------

    if notification_count <= 20:

        print(
            f"Detected 0x55 0x61 packets: "
            f"{packets_in_this_notification}"
        )


# ============================================================
# Main
# ============================================================

async def main():

    global notification_count
    global packet_count

    print()
    print("=" * 70)
    print("WT9011DCL-BT50 BLE Packet Diagnostic")
    print("=" * 70)

    print()
    print(
        f"Device : {IMU_ADDRESS}"
    )

    print(
        f"Notify : {NOTIFY_UUID}"
    )

    print()

    try:

        async with BleakClient(
            IMU_ADDRESS
        ) as client:

            print(
                f"Connected: "
                f"{client.is_connected}"
            )

            await client.start_notify(
                NOTIFY_UUID,
                notification_handler
            )

            print()
            print(
                "开始接收原始 BLE 数据..."
            )

            print(
                "测试时间: 5 秒"
            )

            print()

            await asyncio.sleep(5.0)

            await client.stop_notify(
                NOTIFY_UUID
            )

            print()
            print("=" * 70)
            print("RESULT")
            print("=" * 70)

            print()

            print(
                f"BLE notifications : "
                f"{notification_count}"
            )

            print(
                f"Detected IMU packets : "
                f"{packet_count}"
            )

            print()

            print(
                f"Notification rate : "
                f"{notification_count / 5.0:.2f} Hz"
            )

            print(
                f"IMU packet rate : "
                f"{packet_count / 5.0:.2f} Hz"
            )

            print()

            if packet_count > 0:

                ratio = (
                    packet_count /
                    notification_count
                )

                print(
                    f"Packets / notification : "
                    f"{ratio:.2f}"
                )

            print()

            print("=" * 70)
            print("Diagnostic finished")
            print("=" * 70)

    except Exception as e:

        print()
        print(
            f"ERROR: {e}"
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())