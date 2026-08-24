"""
Train.py  (视角一：Camera回归目标端点 + EMG意图分类)

结构:
  Camera(ResNet18) → 回归 open_target, close_target
  EMG(1D CNN)      → 分类 意图(张开/夹紧/保持)

三损失联合训练:
  loss = open回归 + close回归 + 意图分类

意图标签由角度变化反推:
  angle 变大 → 张开(0)
  angle 变小 → 夹紧(1)
  angle 不变 → 保持(2)

读 *_labeled.csv (需先跑 label_targets.py 标注端点)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ─────────── Config ───────────
BASE_DIR   = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TrainingData"
MODEL_SAVE = os.path.join(BASE_DIR, "gripper_model_intent.pth")

ANGLE_MAX  = 70
ANGLE_MIN  = 0

BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4
VAL_SPLIT  = 0.2

# 意图分类损失的权重
LAMBDA_INTENT = 1.0


# ─────────── Dataset ───────────

class GripperDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]

        # Camera
        img = Image.open(row["frame_path"]).convert("RGB")
        img = self.transform(img)

        # EMG (100,4)
        emg = np.load(row["emg_path"]).astype(np.float32)
        emg = torch.tensor(emg, dtype=torch.float32)

        # 目标端点(归一化)
        open_t  = (float(row["open_target"])  - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
        close_t = (float(row["close_target"]) - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
        open_t  = float(np.clip(open_t, 0, 1))
        close_t = float(np.clip(close_t, 0, 1))

        # 意图标签(0张开/1夹紧/2保持)
        intent = int(row["intent"])

        return (img, emg,
                torch.tensor(open_t,  dtype=torch.float32),
                torch.tensor(close_t, dtype=torch.float32),
                torch.tensor(intent,  dtype=torch.long))


def load_all_data(base_dir):
    all_dfs = []
    for rec_dir in sorted(Path(base_dir).glob("recording_*")):
        # 读标注后的CSV
        csv_files = list(rec_dir.glob("*_labeled.csv"))
        if not csv_files:
            print(f"[Skip] {rec_dir.name}: 没有 _labeled.csv")
            continue
        df = pd.read_csv(csv_files[0]).reset_index(drop=True)

        # 只用valid=1
        if "valid" in df.columns:
            df = df[df["valid"] == 1].reset_index(drop=True)

        # 需要 open_target / close_target
        df = df[df["open_target"].notna() & df["close_target"].notna()]
        df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
        df = df[df["angle"].notna()].reset_index(drop=True)

        # ── 反推意图标签(基于相邻帧角度变化)──
        # 注意:在每个valid段内按frame_index顺序算差分
        intents = []
        prev_angle = None
        for _, r in df.iterrows():
            a = r["angle"]
            if prev_angle is None:
                intents.append(2)  # 第一帧默认保持
            elif a > prev_angle:
                intents.append(0)  # 张开
            elif a < prev_angle:
                intents.append(1)  # 夹紧
            else:
                intents.append(2)  # 保持
            prev_angle = a
        df["intent"] = intents

        # 路径
        df["frame_path"] = df["frame_file"].apply(
            lambda f: str(rec_dir / "frames" / f) if pd.notna(f) and f != "" else "")
        df = df[df["frame_path"].apply(os.path.exists)]
        df["emg_path"] = df["emg_file"].apply(
            lambda f: str(rec_dir / "emg" / f) if pd.notna(f) and f != "" else "")
        df = df[df["emg_path"].apply(os.path.exists)]

        all_dfs.append(df)
        print(f"[Load] {rec_dir.name}: {len(df)} frames")

    combined = pd.concat(all_dfs, ignore_index=True)

    # 意图分布
    print(f"\nTotal frames: {len(combined)}")
    print(f"意图分布: 张开={sum(combined['intent']==0)} "
          f"夹紧={sum(combined['intent']==1)} "
          f"保持={sum(combined['intent']==2)}")
    return combined


# ─────────── Model ───────────

class EMGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class IntentGripper(nn.Module):
    """Camera回归两个目标端点 + EMG分类意图"""
    def __init__(self):
        super().__init__()
        # Camera分支
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # 回归两个端点
        self.target_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2), nn.Sigmoid()   # [open_target, close_target] 各[0,1]
        )

        # EMG分支
        self.emg_encoder = EMGEncoder()
        # 意图三分类
        self.intent_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3)   # 张开/夹紧/保持
        )

    def forward(self, img, emg):
        img_feat = self.image_encoder(img).view(img.size(0), -1)
        targets  = self.target_head(img_feat)      # (B,2)
        open_pred  = targets[:, 0]                  # (B,)
        close_pred = targets[:, 1]                  # (B,)

        emg_feat    = self.emg_encoder(emg)
        intent_logit = self.intent_head(emg_feat)   # (B,3)

        return open_pred, close_pred, intent_logit


# ─────────── Train ───────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    df = load_all_data(BASE_DIR)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(df) * (1 - VAL_SPLIT))
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}\n")

    train_loader = DataLoader(GripperDataset(train_df, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(GripperDataset(val_df, val_tf),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = IntentGripper().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        model.train()
        for img, emg, open_t, close_t, intent in train_loader:
            img, emg = img.to(device), emg.to(device)
            open_t, close_t, intent = open_t.to(device), close_t.to(device), intent.to(device)

            optimizer.zero_grad()
            open_p, close_p, intent_logit = model(img, emg)
            loss = mse(open_p, open_t) + mse(close_p, close_t) \
                   + LAMBDA_INTENT * ce(intent_logit, intent)
            loss.backward()
            optimizer.step()

        # Val
        model.eval()
        v_open, v_close, v_correct, v_total = 0,0,0,0
        with torch.no_grad():
            for img, emg, open_t, close_t, intent in val_loader:
                img, emg = img.to(device), emg.to(device)
                open_t, close_t, intent = open_t.to(device), close_t.to(device), intent.to(device)
                open_p, close_p, intent_logit = model(img, emg)
                v_open  += mse(open_p, open_t).item()*len(img)
                v_close += mse(close_p, close_t).item()*len(img)
                v_correct += (intent_logit.argmax(1)==intent).sum().item()
                v_total += len(img)
        v_open/=len(val_df); v_close/=len(val_df)
        open_deg  = (v_open**0.5)*(ANGLE_MAX-ANGLE_MIN)
        close_deg = (v_close**0.5)*(ANGLE_MAX-ANGLE_MIN)
        intent_acc = v_correct/v_total*100
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"open={open_deg:.1f}deg  close={close_deg:.1f}deg  "
              f"intent_acc={intent_acc:.1f}%")

        combined_val = v_open + v_close
        if combined_val < best_val:
            best_val = combined_val
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  → saved (open={open_deg:.1f} close={close_deg:.1f} acc={intent_acc:.1f}%)")

    print(f"\nDone. Model: {MODEL_SAVE}")


if __name__ == "__main__":
    train()