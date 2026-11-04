"""
inference.py  (视角一：意图选择目标端点)

流程:
  Camera → 回归 open_target, close_target
  EMG    → 分类 意图(张开/夹紧/保持)
  融合   → 张开→去open_target, 夹紧→去close_target, 保持→维持当前
  发送角度给 Arduino (CMD:xx)

保持状态用模型自己记忆的 current_angle(不读Arduino)

Usage:
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

# ─────────── Config ───────────
BASE_DIR   = r"C:\MineApp\Code\Multimodal_Imitation_Learning"
MODEL_PATH = BASE_DIR + r"\Data\TrainingData\gripper_model_intent.pth"

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

INTENT_NAMES = ["OPEN", "CLOSE", "STAY"]
# ──────────────────────────────


# ─────────── EMG State ───────────
class EMGState:
    def __init__(self):
        self.window  = np.zeros((EMG_WINDOW, EMG_CHANNELS), dtype=np.float32)
        self.lock    = threading.Lock()
        self.running = True

emg_state = EMGState()


def emg_reader():
    b, a, z_list = setup_realtime_envelop_filter(1, fs=EMG_SAMPLING)
    z = [z_list.copy() for _ in range(EMG_CHANNELS)]
    device = None
    try:
        device = BITalino(BITALINO_MAC)
        device.start(EMG_SAMPLING, EMG_ACQ_CH)
        print(f"[EMG] connected: {BITALINO_MAC}")
    except Exception as e:
        print(f"[EMG] failed: {e} — continue without EMG")
        return

    while emg_state.running:
        try:
            raw = device.read(EMG_CHUNK)[:, 5:]
            raw_mv = adc_to_mV(raw)
            processed = np.abs(raw_mv)
            for ch in range(EMG_CHANNELS):
                processed[:, ch], z[ch] = realtime_filter(processed[:, ch], b, z[ch], a)
            with emg_state.lock:
                emg_state.window = np.roll(emg_state.window, -EMG_CHUNK, axis=0)
                emg_state.window[-EMG_CHUNK:, :] = processed.astype(np.float32)
        except Exception as e:
            print(f"[EMG] read error: {e}")
            break

    if device is not None:
        try:
            device.stop(); device.close()
        except Exception:
            pass
    print("[EMG] disconnected")


# ─────────── Model (和Train.py一致) ───────────
class EMGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 32)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class IntentGripper(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.target_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2), nn.Sigmoid()
        )
        self.emg_encoder = EMGEncoder()
        self.intent_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3)
        )
    def forward(self, img, emg):
        img_feat = self.image_encoder(img).view(img.size(0), -1)
        targets  = self.target_head(img_feat)
        open_pred, close_pred = targets[:, 0], targets[:, 1]
        emg_feat = self.emg_encoder(emg)
        intent_logit = self.intent_head(emg_feat)
        return open_pred, close_pred, intent_logit


def load_model(path, device):
    model = IntentGripper().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[Model] loaded: {path}")
    return model


# ─────────── Serial ───────────
def auto_detect_port():
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if any(k in desc for k in ["arduino","ch340","cp210","ftdi","usb serial"]):
            return p.device
    return None

def send_angle(ser, angle):
    ser.write(f"CMD:{int(angle)}\n".encode())


# ─────────── Inference Loop ───────────
def run_inference(model, device, tf, ser):
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[Error] cannot open camera {CAM_INDEX}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera opened. Press 'q' to quit.\n")

    # 视频录制
    from datetime import datetime
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
    print(f"[Recording] Video -> {video_path}")

    # 模型自己记忆的当前角度(保持用)
    current_angle = 10.0     # 初始角度
    last_sent = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with emg_state.lock:
            emg_snap = emg_state.window.copy()

        img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = tf(img).unsqueeze(0).to(device)
        emg_t = torch.tensor(emg_snap, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            open_p, close_p, intent_logit = model(img_t, emg_t)
            open_deg  = open_p.item()  * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
            close_deg = close_p.item() * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
            intent = int(intent_logit.argmax(1).item())

        # ── 融合:意图选择目标 ──
        if intent == 0:      # 张开
            current_angle = open_deg
        elif intent == 1:    # 夹紧
            current_angle = close_deg
        # intent==2 保持:current_angle 不变

        target_angle = int(round(max(ANGLE_MIN, min(ANGLE_MAX, current_angle))))

        # 发送(变化才发)
        if target_angle != last_sent:
            send_angle(ser, target_angle)
            last_sent = target_angle

        print(f"intent={INTENT_NAMES[intent]:5s}  "
              f"open={open_deg:.0f}  close={close_deg:.0f}  → angle={target_angle}")

        # ── 画面叠加 ──
        cv2.putText(frame, f"Intent: {INTENT_NAMES[intent]}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"open={open_deg:.0f}  close={close_deg:.0f}", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
        cv2.putText(frame, f"Angle: {target_angle} deg", (10,115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        for ch in range(4):
            rms = float(np.sqrt(np.mean(emg_snap[-10:, ch]**2)))
            cv2.putText(frame, f"CH{ch+1}: {rms:.3f}", (10,150+ch*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,180,255), 1)

        video_writer.write(frame)
        cv2.imshow("Intent Gripper Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    emg_state.running = False
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--baud", type=int, default=BAUD)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(MODEL_PATH, device)

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    t_emg = threading.Thread(target=emg_reader, daemon=True)
    t_emg.start()
    print("[System] waiting EMG init...")
    time.sleep(2)

    port = args.port or auto_detect_port()
    if port is None:
        print("[Error] no serial port. Use --port COM6")
        return
    try:
        ser = serial.Serial(port, args.baud, timeout=1)
        print(f"[Serial] connected: {port}\n")
    except serial.SerialException as e:
        print(f"[Error] serial failed: {e}")
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