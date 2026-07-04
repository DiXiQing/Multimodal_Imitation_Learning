"""
train.py
用摄像头帧图像训练舵机角度回归模型

使用方法：
    python train.py
"""

import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ─────────────────────── 配置 ───────────────────────
BASE_DIR     = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TrainingData"
MODEL_SAVE   = os.path.join(BASE_DIR, "gripper_model.pth")
ANGLE_MAX    = 70       # 你的 maxAngle
ANGLE_MIN    = 0        # 你的 minAngle

BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-4
VAL_SPLIT    = 0.2      # 20% 做验证集
# ────────────────────────────────────────────────────


# ─────────────────────── 数据集 ───────────────────────

class GripperDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img = Image.open(row["frame_path"]).convert("RGB")
        img = self.transform(img)

        # 归一化角度到 [0, 1]
        angle_norm = (float(row["angle"]) - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
        angle_norm = float(np.clip(angle_norm, 0.0, 1.0))

        return img, torch.tensor(angle_norm, dtype=torch.float32)


def load_all_data(base_dir):
    """读取所有 recording_* 文件夹的 CSV，拼成一个 DataFrame"""
    all_dfs = []
    for rec_dir in sorted(Path(base_dir).glob("recording_*")):
        csv_files = list(rec_dir.glob("data_*.csv"))
        if not csv_files:
            continue
        df = pd.read_csv(csv_files[0])
        df = df[df["angle"].notna() & (df["angle"] != "")]
        df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
        df = df[df["angle"].notna()]
        df["frame_path"] = df["frame_file"].apply(
            lambda f: str(rec_dir / "frames" / f) if pd.notna(f) and f != "" else ""
        )
        df = df[df["frame_path"].apply(os.path.exists)]
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"有效帧数：{len(combined)}")
    return combined


# ─────────────────────── 模型 ───────────────────────

def build_model():
    """ResNet18 + 回归输出"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()          # 输出 [0,1]，对应归一化角度
    )
    return model


# ─────────────────────── 训练 ───────────────────────

def train():
    device = torch.device("cuda")
    print(f"使用设备：{torch.cuda.get_device_name(0)}\n")

    # 数据增强（训练集）
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    # 验证集不做增强
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # 加载数据
    df = load_all_data(BASE_DIR)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱

    split = int(len(df) * (1 - VAL_SPLIT))
    train_df = df.iloc[:split]
    val_df   = df.iloc[split:]
    print(f"训练集：{len(train_df)} 帧  验证集：{len(val_df)} 帧\n")

    train_loader = DataLoader(GripperDataset(train_df, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(GripperDataset(val_df, val_tf),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── 训练 ──
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs).squeeze(1)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(imgs)
        train_loss = running_loss / len(train_df)

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).squeeze(1)
                val_loss += criterion(preds, labels).item() * len(imgs)
        val_loss /= len(val_df)

        # 换算成角度误差（°）
        train_deg = (train_loss ** 0.5) * (ANGLE_MAX - ANGLE_MIN)
        val_deg   = (val_loss   ** 0.5) * (ANGLE_MAX - ANGLE_MIN)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.5f}({train_deg:.1f}°)  "
              f"val_loss={val_loss:.5f}({val_deg:.1f}°)")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  → 已保存最优模型（val_loss={val_loss:.5f}）")

    print(f"\n训练完成！最优模型：{MODEL_SAVE}")

    # 画 loss 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "training_curve.png")
    plt.savefig(plot_path)
    print(f"Loss曲线已保存：{plot_path}")
    plt.show()


if __name__ == "__main__":
    train()