import asyncio
import struct
import time
import csv
import keyboard  # pip install keyboard
from bleak import BleakClient
import matplotlib.pyplot as plt        # pip install matplotlib
from datetime import datetime

ADDRESS = "D8:76:A8:5D:CC:CF"
WRITE_UUID = "0000ffe9-0000-1000-8000-00805f9a34fb"
NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"

GRAVITY = 9.81

# ---------- 状态 ----------
measuring = False
calibrating = False
CALIB_FRAMES = 100
calib_buffer = []
stopped = False          # 用于结束 main() 的 while 循环

acc_bias = [0.0, 0.0, 0.0]

# ---------- 数据记录 ----------
records = []          # 每项: (相对时间s, ax, ay, az, |a|)
record_start_time = None


def parse_witmotion(data: bytearray):
    if len(data) < 20 or data[0] != 0x55 or data[1] != 0x61:
        return None
    vals = struct.unpack("<9h", data[2:20])
    ax, ay, az, gx, gy, gz, roll, pitch, yaw = vals
    acc  = [v / 32768 * 16 for v in (ax, ay, az)]
    gyro = [v / 32768 * 2000 for v in (gx, gy, gz)]
    return {"acc": acc, "gyro": gyro}


def finish_calibration():
    global acc_bias, calibrating, calib_buffer, record_start_time, records, measuring

    n = len(calib_buffer)
    ax_list = [s["acc"][0] for s in calib_buffer]
    ay_list = [s["acc"][1] for s in calib_buffer]
    az_list = [s["acc"][2] for s in calib_buffer]
    acc_bias = [sum(ax_list) / n, sum(ay_list) / n, sum(az_list) / n]

    calibrating = False
    calib_buffer = []
    print(">>> 校准完成 <<<")
    print(f"加速度零偏(含重力分量): {[f'{v:.4f}' for v in acc_bias]} g")
    print(">>> 开始测量加速度 <<<")

    records = []
    record_start_time = time.time()
    measuring = True


def notification_handler(sender, data):
    global calib_buffer

    parsed = parse_witmotion(data)
    if parsed is None:
        return

    if calibrating:
        calib_buffer.append(parsed)
        if len(calib_buffer) >= CALIB_FRAMES:
            finish_calibration()
        return

    if not measuring:
        return

    raw = parsed["acc"]
    net = [(raw[i] - acc_bias[i]) * GRAVITY for i in range(3)]
    mag = (net[0] ** 2 + net[1] ** 2 + net[2] ** 2) ** 0.5

    t_rel = time.time() - record_start_time
    records.append((t_rel, net[0], net[1], net[2], mag))

    print(f"Acc(net): ax={net[0]:7.3f} ay={net[1]:7.3f} az={net[2]:7.3f} m/s²   "
          f"|a|={mag:6.3f} m/s²")


def save_csv_and_chart():
    if not records:
        print("没有数据,跳过保存")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = f"acc_data_{ts}.csv"
    chart_path = f"acc_chart_{ts}.png"

    # ---------- 写 CSV(内置库,不需要额外安装) ----------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "ax_m/s2", "ay_m/s2", "az_m/s2", "mag_m/s2"])
        writer.writerows(records)
    print(f"[已保存] CSV → {csv_path}  (可直接用 Excel/WPS 打开)")

    # ---------- 画图 ----------
    t    = [r[0] for r in records]
    ax_  = [r[1] for r in records]
    ay_  = [r[2] for r in records]
    az_  = [r[3] for r in records]
    mag_ = [r[4] for r in records]

    plt.figure(figsize=(12, 6))
    plt.plot(t, ax_, label="ax", linewidth=1)
    plt.plot(t, ay_, label="ay", linewidth=1)
    plt.plot(t, az_, label="az", linewidth=1)
    plt.plot(t, mag_, label="|a|", linewidth=1.5, linestyle="--", color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("Net Acceleration over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[已保存] 曲线图 → {chart_path}")


def stop_and_save():
    """按 's' 结束测量并保存(单向,只负责停止)。"""
    global measuring, calibrating, stopped

    if calibrating:
        print("校准中,请稍候...")
        return
    if not measuring:
        print("尚未在测量中")
        return

    measuring = False
    print(">>> 测量停止,正在保存数据... <<<")
    save_csv_and_chart()
    stopped = True


def start_calibration():
    """连接成功后自动调用,开始校准 → 自动进入测量。"""
    global calibrating, calib_buffer
    calib_buffer = []
    calibrating = True
    print(">>> 已连接,自动校准中,请保持设备静止约1秒... <<<")


async def main():
    global stopped

    keyboard.add_hotkey("s", stop_and_save)
    print("程序启动后自动校准并开始记录数据")
    print("按 's' 键 停止测量并导出 CSV + 曲线图, 也可 Ctrl+C 退出(会自动保存)")

    async with BleakClient(ADDRESS) as client:
        print("Connected:", client.is_connected)
        await client.start_notify(NOTIFY_UUID, notification_handler)

        # 连接成功后立即自动开始校准 → 测量,不再需要按键触发
        start_calibration()

        try:
            while not stopped:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            if measuring and not stopped:
                print(">>> 收到退出信号,正在保存数据... <<<")
                save_csv_and_chart()


asyncio.run(main())