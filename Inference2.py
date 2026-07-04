"""
inference.py
多模态实时推理：Camera + EMG → 模型预测角度 → 发串口给Arduino

Usage:
    python inference.py
    python inference.py --port COM6
"""

import cv2
import serial
import serial.tools.list_ports
import threading
import time
import argparse
import numpy as np
from PIL import Image
from bitalino import BITalino

import torch
import torch.nn as nn
from torchvision import models, transforms

from emg_interface.funcs import adc_to_mV, setup_realtime_envelop_filter, realtime_filter

# ─────────────────────── Config ───────────────────────
BASE_DIR   = r"C:\MineApp\Code\Multimodal_Imitation_Learning"
MODEL_PATH = BASE_DIR + r"\Data\TrainingData\gripper_model_multimodal.pth"

ANGLE_MAX  = 70
ANGLE_MIN  = 0

CAM_INDEX  = 0
BAUD       = 9600

BITALINO_MAC = "COM4"
EMG_SAMPLING = 1000
EMG_CHANNELS = 4
EMG_CHUNK    = 10
EMG_WINDOW   = 100
EMG_ACQ_CH   = [1, 2, 3, 4]
# ──────────────────────────────────────────────────────


# ─────────────────────── Shared EMG State ───────────────────────

class EMGState:
    def __init__(self):
        self.window  = np.zeros((EMG_WINDOW, EMG_CHANNELS), dtype=np.float32)
        self.lock    = threading.Lock()
        self.running = True

emg_state = EMGState()


# ─────────────────────── EMG Reader Thread ───────────────────────

def emg_reader():
    b, a, z_list = setup_realtime_envelop_filter(1, fs=EMG_SAMPLING)
    z = [z_list.copy() for _ in range(EMG_CHANNELS)]

    device = None
    try:
        device = BITalino(BITALINO_MAC)
        device.start(EMG_SAMPLING, EMG_ACQ_CH)
        print(f"[EMG] BITalino connected: {BITALINO_MAC}")
    except Exception as e:
        print(f"[EMG] BITalino connection failed: {e} — inference will continue without EMG")
        return

    while emg_state.running:
        try:
            raw      = device.read(EMG_CHUNK)[:, 5:]
            raw_mv   = adc_to_mV(raw)
            processed = np.abs(raw_mv)
            for ch in range(EMG_CHANNELS):
                processed[:, ch], z[ch] = realtime_filter(
                    processed[:, ch], b, z[ch], a)

            with emg_state.lock:
                emg_state.window = np.roll(emg_state.window, -EMG_CHUNK, axis=0)
                emg_state.window[-EMG_CHUNK:, :] = processed.astype(np.float32)

        except Exception as e:
            print(f"[EMG] Read error: {e}")
            break

    if device is not None:
        try:
            device.stop()
            device.close()
        except Exception:
            pass
    print("[EMG] Disconnected")


# ─────────────────────── Model ───────────────────────

class EMGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class MultimodalGripper(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.emg_encoder   = EMGEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img, emg):
        img_feat = self.image_encoder(img).view(img.size(0), -1)
        emg_feat = self.emg_encoder(emg)
        fused    = torch.cat([img_feat, emg_feat], dim=1)
        return self.fusion(fused).squeeze(1)


def load_model(model_path, device):
    model = MultimodalGripper().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Model] Loaded: {model_path}")
    return model


# ─────────────────────── Serial ───────────────────────

def auto_detect_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        if any(kw in desc for kw in ["arduino", "ch340", "cp210", "ftdi", "usb serial"]):
            return p.device
    return None


def send_angle(ser, angle: int):
    cmd = f"CMD:{angle}\n"
    ser.write(cmd.encode("utf-8"))


# ─────────────────────── Inference Loop ───────────────────────

def run_inference(model, device, tf, ser):
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera {CAM_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera opened. Press 'q' to quit.\n")

    last_angle = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Frame read failed")
            break

        # ── 获取EMG快照 ──
        with emg_state.lock:
            emg_snap = emg_state.window.copy()  # (100, 4)

        # ── 推理 ──
        img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = tf(img).unsqueeze(0).to(device)
        emg_t = torch.tensor(emg_snap, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # 分别打印两个分支的输出
            img_feat = model.image_encoder(img_t).view(1, -1)
            emg_feat = model.emg_encoder(emg_t)
            print(f"img_feat norm: {img_feat.norm().item():.3f}  emg_feat norm: {emg_feat.norm().item():.3f}")
            out = model(img_t, emg_t).item()

        pred_angle = int(round(out * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN))
        pred_angle = max(ANGLE_MIN, min(ANGLE_MAX, pred_angle))

        print(f"pred: {pred_angle} deg")

        if pred_angle != last_angle:
            send_angle(ser, pred_angle)
            print(f"→ SEND CMD:{pred_angle}")
            last_angle = pred_angle

        # ── 画面叠加 ──
        cv2.putText(frame, f"Predicted: {pred_angle} deg",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "AUTO MODE",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # EMG各通道RMS
        for ch in range(4):
            ch_rms = float(np.sqrt(np.mean(emg_snap[-10:, ch] ** 2)))
            cv2.putText(frame, f"EMG CH{ch+1}: {ch_rms:.4f} mV",
                        (10, 120 + ch * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        cv2.imshow("Gripper Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit.")
            break

    emg_state.running = False
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────── Main ───────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--baud", type=int, default=BAUD)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(MODEL_PATH, device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # 启动EMG线程
    t_emg = threading.Thread(target=emg_reader, daemon=True)
    t_emg.start()
    print("[System] Waiting for EMG to initialize...")
    time.sleep(2)

    # 串口
    port = args.port or auto_detect_port()
    if port is None:
        print("[Error] No serial port found. Use --port COM6")
        return

    try:
        ser = serial.Serial(port, args.baud, timeout=1)
        print(f"[Serial] Connected: {port} @ {args.baud}\n")
    except serial.SerialException as e:
        print(f"[Error] Serial failed: {e}")
        return

    try:
        run_inference(model, device, tf, ser)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        emg_state.running = False
        ser.close()


if __name__ == "__main__":
    main()
