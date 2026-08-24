"""
test_offline.py
离线测试：用训练好的模型预测测试视频的角度，和真实角度对比

使用方法：
    python test_offline.py
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ─────────────────────── 配置（改这里）───────────────────────
BASE_DIR   = r"C:\MineApp\Code\Multimodal_Imitation_Learning"
Testing_DIR   = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TestingData"
MODEL_PATH = os.path.join(BASE_DIR, "gripper_model.pth")
ANGLE_MAX  = 70
ANGLE_MIN  = 0

# ↓↓↓ 把你的测试文件夹名字填在这里 ↓↓↓
TEST_FOLDERS = [
    "recording_20260606_214809",
    ]
# ─────────────────────────────────────────────────────────────


# ─────────────────────── 模型（和train.py一致）───────────────────────

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


# ─────────────────────── 推理 ───────────────────────

def predict_folder(model, device, tf, rec_dir: Path):
    """对一个recording文件夹做推理，返回 DataFrame（含真实角度和预测角度）"""
    csv_files = list(rec_dir.glob("data_*.csv"))
    if not csv_files:
        print(f"[跳过] {rec_dir.name}：找不到CSV")
        return None

    df = pd.read_csv(csv_files[0])
    df = df[df["angle"].notna() & (df["angle"] != "")]
    df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
    df = df[df["angle"].notna()]
    df["frame_path"] = df["frame_file"].apply(
        lambda f: str(rec_dir / "frames" / f) if pd.notna(f) and f != "" else ""
    )
    df = df[df["frame_path"].apply(os.path.exists)].reset_index(drop=True)

    pred_angles = []
    model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():
            img = Image.open(row["frame_path"]).convert("RGB")
            img_t = tf(img).unsqueeze(0).to(device)
            out = model(img_t).squeeze().item()
            pred_angle = out * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
            pred_angles.append(round(pred_angle, 2))

    df["pred_angle"] = pred_angles
    df["error_deg"]  = (df["pred_angle"] - df["angle"]).abs()
    return df


# ─────────────────────── 统计 & 画图 ───────────────────────

def show_results(df, folder_name, save_dir):
    mae  = df["error_deg"].mean()
    rmse = (df["error_deg"] ** 2).mean() ** 0.5
    max_err = df["error_deg"].max()

    print(f"\n─── {folder_name} ───")
    print(f"  帧数：{len(df)}")
    print(f"  MAE  : {mae:.2f}°")
    print(f"  RMSE : {rmse:.2f}°")
    print(f"  最大误差：{max_err:.2f}°")

    # 画预测 vs 真实角度
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(df.index, df["angle"],      label="Ground Truth", color="steelblue")
    axes[0].plot(df.index, df["pred_angle"], label="Predicted", color="orange", linestyle="--")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title(f"{folder_name}  |  MAE={mae:.2f}°  RMSE={rmse:.2f}°")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df["error_deg"], color="red", alpha=0.7)
    axes[1].axhline(mae, color="red", linestyle="--", label=f"MAE={mae:.2f}°")
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("Error (deg)")
    axes[1].set_title("Per-frame Prediction Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"test_result_{folder_name}.png")
    plt.savefig(save_path)
    print(f"  图表已保存：{save_path}")
    plt.show()

    return {"folder": folder_name, "MAE": mae, "RMSE": rmse, "max_error": max_err}


# ─────────────────────── 主入口 ───────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 加载模型
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"模型已加载：{MODEL_PATH}\n")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    all_stats = []
    for folder_name in TEST_FOLDERS:
        rec_dir = Path(Testing_DIR) / folder_name
        if not rec_dir.exists():
            print(f"[错误] 文件夹不存在：{rec_dir}")
            continue

        df = predict_folder(model, device, tf, rec_dir)
        if df is None:
            continue

        stats = show_results(df, folder_name, Testing_DIR)
        all_stats.append(stats)

    # 汇总
    if len(all_stats) > 1:
        print("\n─── 整体汇总 ───")
        for s in all_stats:
            print(f"  {s['folder']}：MAE={s['MAE']:.2f}°  RMSE={s['RMSE']:.2f}°")
        overall_mae = np.mean([s["MAE"] for s in all_stats])
        print(f"\n  平均 MAE：{overall_mae:.2f}°")


if __name__ == "__main__":
    main()