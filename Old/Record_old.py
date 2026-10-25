"""
Record.py
Synchronized recording: Camera frames + EMG signal + Servo angle

Dependencies:
    pip install pyserial opencv-python bitalino numpy

Usage:
    python Record.py
    python Record.py --port COM3
    python Record.py --no-save-frames
"""

import cv2
import serial
import serial.tools.list_ports
import threading
import time
import csv
import os
import argparse
import numpy as np
from datetime import datetime
from collections import deque
from bitalino import BITalino

from emg_interface.funcs import adc_to_mV, setup_realtime_envelop_filter, realtime_filter

# ─────────────────────── Config ───────────────────────
# BITALINO_MAC    = "88:6B:0F:D9:AF:C6"
# BITALINO_MAC    = "00:21:06:BE:18:CB"
BITALINO_MAC = "COM4"
EMG_SAMPLING    = 1000          # Hz
EMG_CHANNELS    = 4
EMG_CHUNK       = 10            # samples per read
EMG_WINDOW_MS   = 100           # ms window saved per frame → 100 samples
EMG_ACQ_CH      = [1, 2, 3, 4]

VCC               = 3.3
SAMPLING_RES      = 2 ** 10
SENSOR_GAIN       = 1009
# ──────────────────────────────────────────────────────


# ─────────────────────── Shared State ───────────────────────

class SharedState:
    def __init__(self, initial_angle: int = 10):
        # Servo angle
        self.latest_angle    = initial_angle
        self.latest_angle_ts = None
        # EMG buffer: keeps last 100 samples × 4 channels
        self.emg_window      = np.zeros((EMG_WINDOW_MS, EMG_CHANNELS), dtype=np.float32)
        self.lock            = threading.Lock()
        self.running         = True
        self.is_valid        = False

state = SharedState()


# ─────────────────────── Serial Reader Thread (Servo) ───────────────────────

def serial_reader(port: str, baud: int):
    try:
        ser = serial.Serial(port, baud, timeout=1)

        global arduino_ser
        arduino_ser = ser

        print(f"[Serial] Connected {port} @ {baud}")
    except serial.SerialException as e:
        print(f"[Serial] Failed: {e}")
        # state.running = False
        return

    while state.running:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            angle_val = None
            if line.startswith("ANGLE:"):
                try:
                    angle_val = int(line.split(":")[1])
                except ValueError:
                    pass
            else:
                try:
                    angle_val = int(line)
                except ValueError:
                    pass

            if angle_val is not None:
                ts = time.time()
                with state.lock:
                    state.latest_angle    = angle_val
                    state.latest_angle_ts = ts

        except serial.SerialException:
            print("[Serial] Disconnected")
            break

    ser.close()

# ─────────────────────── 发送函数 ───────────────────────
arduino_ser = None
def serial_send(cmd):
    global arduino_ser

    if arduino_ser is not None:
        arduino_ser.write((cmd+"\n").encode())


# ─────────────────────── EMG Reader Thread (BITalino) ───────────────────────

def emg_reader():
    """
    Continuously read EMG from BITalino and keep a rolling 100-sample window.
    """
    # Setup envelope filter
    b, a, z_list = setup_realtime_envelop_filter(1, fs=EMG_SAMPLING)
    z = [z_list.copy() for _ in range(EMG_CHANNELS)]

    device = None
    try:
        device = BITalino(BITALINO_MAC)
        device.start(EMG_SAMPLING, EMG_ACQ_CH)
        print(f"[EMG] BITalino connected: {BITALINO_MAC}")
    except Exception as e:
        print(f"[EMG] BITalino connection failed: {e}")
        # state.running = False
        return

    while state.running:
        try:
            raw = device.read(EMG_CHUNK)[:, 5:]           # shape (10, 4)
            raw_mv = adc_to_mV(raw)                        # ADC → mV

            # Rectify + envelope filter per channel
            processed = np.abs(raw_mv)
            for ch in range(EMG_CHANNELS):
                processed[:, ch], z[ch] = realtime_filter(
                    processed[:, ch], b, z[ch], a
                )

            # Roll window and append new chunk
            with state.lock:
                state.emg_window = np.roll(state.emg_window, -EMG_CHUNK, axis=0)
                state.emg_window[-EMG_CHUNK:, :] = processed.astype(np.float32)

        except Exception as e:
            print(f"[EMG] Read error: {e}")
            break

    if device is not None:
        try:
            device.stop()
            device.close()
        except Exception:
            pass
    print("[EMG] BITalino disconnected")


# ─────────────────────── Camera + Recording Main Loop ───────────────────────

def run_camera(save_frames: bool, output_dir: str, csv_path: str):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[Camera] Cannot open camera")
        state.running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[Camera] Opened, {int(cap.get(3))}x{int(cap.get(4))}, FPS~{fps:.1f}")

    # 视频录制
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
    print(f"[Recording] Video → {video_path}")

    frames_dir = os.path.join(output_dir, "frames")
    emg_dir    = os.path.join(output_dir, "emg")
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(emg_dir, exist_ok=True)

    # CSV header
    # emg_ch1_t0 ... emg_ch4_t99 = 100 timesteps × 4 channels = 400 columns
    emg_cols = [f"emg_ch{ch+1}_t{t}" for t in range(EMG_WINDOW_MS) for ch in range(EMG_CHANNELS)]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer   = csv.writer(csv_file)
    writer.writerow(["timestamp", "frame_index", "angle", "angle_timestamp",
                     "frame_file", "emg_file", "valid"])

    frame_idx = 0
    action_history = deque(maxlen=5)
    print("\n[System] Recording started. Press 'q' to quit.\n")

    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame read failed")
            break

        frame_ts = time.time()

        with state.lock:
            angle     = state.latest_angle
            angle_ts  = state.latest_angle_ts
            emg_snap  = state.emg_window.copy()   # (100, 4)

            # 四通道平均
            ch_mean = np.mean(emg_snap, axis=0)

            ch1, ch2, ch3, ch4 = ch_mean

            activity = ch1 + ch2 + ch3 + ch4

            # 判断动作
            if activity < 0.18:
                action = "KEEP"

            elif ch2 > ch3:
                action = "OPEN"

            else:
                action = "CLOSE"

            # ---------- 防抖 ----------
            action_history.append(action)

            if len(action_history) == action_history.maxlen:

                if all(a == "OPEN" for a in action_history):
                    serial_send("O")

                elif all(a == "CLOSE"  for a in action_history):
                    serial_send("C")

        # Overlay info on frame
        ts_str    = datetime.fromtimestamp(frame_ts).strftime("%H:%M:%S.%f")[:-3]
        angle_str = f"Angle: {angle} deg"

        valid_str = "● REC" if state.is_valid else "○ STANDBY"
        valid_color = (0, 0, 255) if state.is_valid else (150, 150, 150)
        cv2.putText(frame, valid_str, (500, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, valid_color, 2)

        emg_rms   = float(np.sqrt(np.mean(emg_snap[-10:, :] ** 2)))  # RMS of last 10ms
        emg_str   = f"EMG RMS: {emg_rms:.4f} mV"

        cv2.putText(frame, angle_str, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        for ch in range(4):
            ch_rms = float(np.sqrt(np.mean(emg_snap[-10:, ch] ** 2)))
            cv2.putText(frame, f"EMG CH{ch+1}: {ch_rms:.4f} mV",
                        (10, 75 + ch * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                    
        cv2.putText(frame, ts_str, (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Frame #{frame_idx}", (480, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, f"Activity: {activity:.4f}",
                    (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    
        video_writer.write(frame)
        cv2.imshow("Multimodal Recorder", frame)

        # Save frame image
        frame_file = ""
        if save_frames:
            frame_file = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, frame_file), frame)

        # Save EMG snapshot as .npy (100×4 float32)
        emg_file = f"emg_{frame_idx:06d}.npy"
        np.save(os.path.join(emg_dir, emg_file), emg_snap)

        # Write CSV row (flatten EMG window inline too for easy inspection)
        writer.writerow([
            round(frame_ts, 4),
            frame_idx,
            angle if angle is not None else "",
            round(angle_ts, 4) if angle_ts else "",
            frame_file,
            emg_file,
            1 if state.is_valid else 0,
        ])

        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[System] Quit")
            state.running = False
        elif key == ord("s"):                          # ← 新增
            state.is_valid = True
            print(f"[Record] Valid segment START - Frame #{frame_idx}")
        elif key == ord("e"):                          # ← 新增
            state.is_valid = False
            print(f"[Record] Valid segment END - Frame #{frame_idx}")

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\n[Done] CSV     → {csv_path}")
    print(f"[Done] Frames  → {frames_dir}/")
    print(f"[Done] EMG     → {emg_dir}/")


# ─────────────────────── Auto Detect Serial Port ───────────────────────

def auto_detect_port() -> str | None:
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        if any(kw in desc for kw in ["arduino", "ch340", "cp210", "ftdi", "usb serial"]):
            return p.device
    if ports:
        print("[Serial] Arduino not found. Available ports:")
        for p in ports:
            print(f"  {p.device}  {p.description}")
    return None


# ─────────────────────── Main ───────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multimodal recorder: Camera + EMG + Servo")
    parser.add_argument("--port",          type=str, default=None)
    parser.add_argument("--baud",          type=int, default=9600)
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--output",        type=str, default=None)
    parser.add_argument("--initial-angle", type=int, default=10)
    args = parser.parse_args()

    global state
    state = SharedState(initial_angle=args.initial_angle)

    port = args.port or auto_detect_port()
    if port is None:
        print("[Error] Serial port not found. Use --port COM3")
        return

    run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData"
    os.makedirs(backup_dir, exist_ok=True)
    output_dir = args.output or os.path.join(backup_dir, f"recording_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    csv_path   = os.path.join(output_dir, f"data_{run_id}.csv")

    # Start servo serial thread
    t_serial = threading.Thread(
        target=serial_reader, args=(port, args.baud), daemon=True)
    t_serial.start()

    # Start EMG thread
    t_emg = threading.Thread(target=emg_reader, daemon=True)
    t_emg.start()

    # Wait for EMG to initialize
    print("[System] Waiting for EMG to initialize...")
    time.sleep(2)

    # Main camera loop
    try:
        run_camera(
            save_frames=not args.no_save_frames,
            output_dir=output_dir,
            csv_path=csv_path,
        )
    except KeyboardInterrupt:
        print("\n[System] Ctrl+C")
        state.running = False


if __name__ == "__main__":
    main()