"""
inference.py
实时推理：摄像头读帧 → 模型预测角度 → 发串口给Arduino执行

使用方法：
    python inference.py
    python inference.py --port COM3
"""

import cv2
import serial
import serial.tools.list_ports
import threading
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ─────────────────────── 配置 ───────────────────────
BASE_DIR   = r"C:\MineApp\Code\Multimodal_Imitation_Learning"
MODEL_PATH = BASE_DIR + r"\gripper_model.pth"
ANGLE_MAX  = 70
ANGLE_MIN  = 0

CAM_INDEX  = 0          # 外接摄像头编号
BAUD       = 9600
INTERVAL   = 0.05       # 推理间隔（秒），约20fps
# ────────────────────────────────────────────────────


# ─────────────────────── 模型 ───────────────────────

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return model


def load_model(model_path, device):
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded: {model_path}")
    return model


# ─────────────────────── 串口 ───────────────────────

def auto_detect_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        if any(kw in desc for kw in ["arduino", "ch340", "cp210", "ftdi", "usb serial"]):
            return p.device
    if ports:
        print("Available ports:")
        for p in ports:
            print(f"  {p.device}  {p.description}")
    return None


def send_angle(ser, angle: int):
    """发送角度指令给Arduino，格式: CMD:45\n"""
    cmd = f"CMD:{angle}\n"
    ser.write(cmd.encode("utf-8"))


# ─────────────────────── 推理主循环 ───────────────────────

def run_inference(model, device, tf, ser):
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera {CAM_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera opened. Press 'q' to quit.\n")

    last_angle = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Frame read failed")
            break

        # ── 推理 ──
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t).squeeze().item()

        pred_angle = int(round(out * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN))
        pred_angle = max(ANGLE_MIN, min(ANGLE_MAX, pred_angle))

        # ── 只在角度变化时发串口（减少噪声）──
        print(f"pred: {pred_angle}°")
        if pred_angle != last_angle:
            send_angle(ser, pred_angle)
            last_angle = pred_angle

        # ── 画面叠加信息 ──
        cv2.putText(frame, f"Predicted Angle: {pred_angle} deg",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "AUTO MODE",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Gripper Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit.")
            break

        time.sleep(INTERVAL)

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────── 主入口 ───────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--baud", type=int, default=BAUD)
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 模型
    model = load_model(MODEL_PATH, device)

    # 图像预处理（和训练时一致）
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # 串口
    port = args.port or auto_detect_port()
    if port is None:
        print("[Error] No serial port found. Use --port COM3")
        return

    try:
        ser = serial.Serial(port, args.baud, timeout=1)
        print(f"Serial connected: {port} @ {args.baud}\n")
    except serial.SerialException as e:
        print(f"[Error] Serial connection failed: {e}")
        return

    # 推理
    try:
        run_inference(model, device, tf, ser)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
