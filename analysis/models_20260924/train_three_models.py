"""Train three small PyTorch regressors and save them as .pth files.

Models:
  1. area_only:         object_area -> finger_width
  2. area_velocity:     object_area + velocity -> finger_width
  3. area_acceleration: object_area + ax + ay + az -> finger_width

Validation keeps complete trials separate from training so adjacent frames from the
same trial cannot appear in both sets.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DEFAULT_DATA_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "pth_models"

MODEL_SPECS = {
    "area_only": ["object_area"],
    "area_velocity": ["object_area", "velocity"],
    "area_acceleration": ["object_area", "ax", "ay", "az"],
}


class FingerWidthRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(data_root.glob("TRIAL_*/grasp_data.csv")):
        frame = pd.read_csv(csv_path)
        frame["trial"] = csv_path.parent.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No grasp_data.csv found under {data_root}")

    data = pd.concat(frames, ignore_index=True)
    required = ["finger_width", "object_area", "velocity", "ax", "ay", "az"]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    for name in required:
        data[name] = pd.to_numeric(data[name], errors="coerce")

    finite = np.isfinite(data[required]).all(axis=1)
    valid = finite & (data["finger_width"] > 0) & (data["object_area"] > 0)
    return data.loc[valid].reset_index(drop=True)


def split_by_trial(data: pd.DataFrame, val_trials: int, seed: int):
    trials = sorted(data["trial"].unique())
    if len(trials) < 2:
        raise ValueError("At least two trials are required for train/validation split")

    count = min(max(1, val_trials), len(trials) - 1)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(trials))
    val_names = set(shuffled[:count])
    val_mask = data["trial"].isin(val_names)
    return data.loc[~val_mask].copy(), data.loc[val_mask].copy(), sorted(val_names)


def mean_and_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def regression_metrics(target: torch.Tensor, prediction: torch.Tensor) -> tuple[float, float]:
    error = prediction - target
    rmse = torch.sqrt(torch.mean(error.square())).item()
    mae = torch.mean(torch.abs(error)).item()
    return rmse, mae


def train_one_model(
    model_name: str,
    columns: list[str],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
) -> Path:
    x_train_raw = train_data[columns].to_numpy(dtype=np.float32)
    x_val_raw = val_data[columns].to_numpy(dtype=np.float32)
    y_train_raw = train_data["finger_width"].to_numpy(dtype=np.float32)
    y_val_raw = val_data["finger_width"].to_numpy(dtype=np.float32)

    x_mean, x_scale = mean_and_scale(x_train_raw)
    y_mean = np.float32(y_train_raw.mean())
    y_scale = np.float32(y_train_raw.std())
    if y_scale < 1e-8:
        y_scale = np.float32(1.0)

    x_train = torch.from_numpy((x_train_raw - x_mean) / x_scale).to(device)
    x_val = torch.from_numpy((x_val_raw - x_mean) / x_scale).to(device)
    y_train = torch.from_numpy((y_train_raw - y_mean) / y_scale).to(device)
    y_val_px = torch.from_numpy(y_val_raw).to(device)

    model = FingerWidthRegressor(len(columns), hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = loss_fn(prediction, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction_px = model(x_val) * float(y_scale) + float(y_mean)
            val_rmse, val_mae = regression_metrics(y_val_px, val_prediction_px)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(
                f"[{model_name}] epoch {epoch:4d}/{epochs} "
                f"train_loss={loss.item():.5f} val_RMSE={val_rmse:.2f}px val_MAE={val_mae:.2f}px"
            )

    if best_state is None:
        raise RuntimeError(f"Training failed for {model_name}")

    checkpoint = {
        "model_name": model_name,
        "model_class": "FingerWidthRegressor",
        "input_columns": columns,
        "input_size": len(columns),
        "hidden_size": hidden_size,
        "state_dict": best_state,
        "x_mean": torch.tensor(x_mean, dtype=torch.float32),
        "x_scale": torch.tensor(x_scale, dtype=torch.float32),
        "y_mean": float(y_mean),
        "y_scale": float(y_scale),
        "target_column": "finger_width",
        "best_val_rmse_px": best_val_rmse,
    }

    save_path = output_dir / f"{model_name}.pth"
    torch.save(checkpoint, save_path)
    print(f"[{model_name}] saved: {save_path} (best val RMSE={best_val_rmse:.2f}px)")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train three finger-width regression models")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--val-trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_root)
    train_data, val_data, val_names = split_by_trial(data, args.val_trials, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Valid rows: {len(data)} | train: {len(train_data)} | validation: {len(val_data)}")
    print(f"Validation trials: {', '.join(val_names)}")

    for model_name, columns in MODEL_SPECS.items():
        train_one_model(
            model_name=model_name,
            columns=columns,
            train_data=train_data,
            val_data=val_data,
            output_dir=args.output_dir,
            device=device,
            epochs=args.epochs,
            learning_rate=args.lr,
            hidden_size=args.hidden_size,
        )


if __name__ == "__main__":
    main()
