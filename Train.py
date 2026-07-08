"""
Train.py
多模态训练：Camera图像 + EMG信号 → 舵机角度回归

Usage:
    python Train.py
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

# ─────────────────────── Config ───────────────────────
BASE_DIR   = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TrainingData"
MODEL_SAVE = os.path.join(BASE_DIR, "gripper_model_multimodal.pth")

ANGLE_MAX  = 70
ANGLE_MIN  = 0

EMG_WINDOW = 100   # 时间步
EMG_CH     = 4     # 通道数

BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4
VAL_SPLIT  = 0.2
# ──────────────────────────────────────────────────────


# ─────────────────────── Dataset ───────────────────────

class GripperDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]

        # ── Camera图像 ──
        img = Image.open(row["frame_path"]).convert("RGB")
        img = self.transform(img)

        # ── EMG信号 (100, 4) → flatten → (400,) ──
        emg = np.load(row["emg_path"]).astype(np.float32)  # (100, 4)
        emg = torch.tensor(emg, dtype=torch.float32)        # (100, 4)

        # ── 角度标签 归一化到[0,1] ──
        angle_norm = (float(row["angle"]) - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
        angle_norm = float(np.clip(angle_norm, 0.0, 1.0))

        return img, emg, torch.tensor(angle_norm, dtype=torch.float32)


def load_all_data(base_dir):
    all_dfs = []
    for rec_dir in sorted(Path(base_dir).glob("recording_*")):
        csv_files = list(rec_dir.glob("*_clean.csv"))
        if not csv_files:
            continue
        df = pd.read_csv(csv_files[0])

        # 只用valid=1的帧
        if "valid" in df.columns:
            df = df[df["valid"] == 1]
        
        # 过滤angle为空
        df = df[df["angle"].notna() & (df["angle"] != "")]
        df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
        df = df[df["angle"].notna()]

        # frame路径
        df["frame_path"] = df["frame_file"].apply(
            lambda f: str(rec_dir / "frames" / f) if pd.notna(f) and f != "" else ""
        )
        df = df[df["frame_path"].apply(os.path.exists)]

        # emg路径
        df["emg_path"] = df["emg_file"].apply(
            lambda f: str(rec_dir / "emg" / f) if pd.notna(f) and f != "" else ""
        )
        df = df[df["emg_path"].apply(os.path.exists)]

        all_dfs.append(df)
        print(f"[Load] {rec_dir.name}: {len(df)} frames")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal valid frames: {len(combined)}")
    return combined


# ─────────────────────── Model ───────────────────────

class EMGEncoder(nn.Module):
    """1D CNN 处理 EMG 时间序列 (100, 4) → 32维特征"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 4, 100)
            nn.Conv1d(4, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),   # → (B, 16, 50)

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),   # → (B, 32, 25)

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (B, 64, 1)
        )
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        # x: (B, 100, 4) → (B, 4, 100)
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)  # (B, 64)
        return self.fc(x)             # (B, 32)


class MultimodalGripper(nn.Module):
    """残差融合：Camera预测基础角度，EMG预测调整量"""
    def __init__(self):
        super().__init__()

        # Camera分支：预测基础角度
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.camera_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()          # 基础角度 [0,1]
        )

        # EMG分支：预测调整量
        self.emg_encoder = EMGEncoder()
        self.emg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()             # 调整量 [-1,1]
        )

    def forward(self, img, emg):
        # Camera → 基础角度
        img_feat   = self.image_encoder(img).view(img.size(0), -1)
        base_angle = self.camera_head(img_feat).squeeze(1)      # (B,)

        # EMG → 调整量，限制在±0.3范围内
        emg_feat   = self.emg_encoder(emg)
        adjustment = self.emg_head(emg_feat).squeeze(1) * 0.3  # (B,)

        # 残差融合
        output = torch.clamp(base_angle + adjustment, 0.0, 1.0)
        return output


# ─────────────────────── Train ───────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    df = load_all_data(BASE_DIR)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split    = int(len(df) * (1 - VAL_SPLIT))
    train_df = df.iloc[:split]
    val_df   = df.iloc[split:]
    print(f"Train: {len(train_df)}  Val: {len(val_df)}\n")

    train_loader = DataLoader(GripperDataset(train_df, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(GripperDataset(val_df, val_tf),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = MultimodalGripper().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        running_loss = 0.0
        for imgs, emgs, labels in train_loader:
            imgs, emgs, labels = imgs.to(device), emgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs, emgs)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(imgs)
        train_loss = running_loss / len(train_df)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, emgs, labels in val_loader:
                imgs, emgs, labels = imgs.to(device), emgs.to(device), labels.to(device)
                preds    = model(imgs, emgs)
                val_loss += criterion(preds, labels).item() * len(imgs)
        val_loss /= len(val_df)

        train_deg = (train_loss ** 0.5) * (ANGLE_MAX - ANGLE_MIN)
        val_deg   = (val_loss   ** 0.5) * (ANGLE_MAX - ANGLE_MIN)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train={train_loss:.5f}({train_deg:.1f} deg)  "
              f"val={val_loss:.5f}({val_deg:.1f} deg)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  → Saved best model (val={val_loss:.5f})")

    print(f"\nDone. Best model: {MODEL_SAVE}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve (Multimodal)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "training_curve_multimodal.png")
    plt.savefig(plot_path)
    print(f"Loss curve saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    train()
