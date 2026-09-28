"""Train three 15-frame window MLP models and save three .pth files."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData")
OUTPUT_DIR = Path(__file__).resolve().parent / "pth_models"

MODEL_SPECS = {
    "window_area_only": ["object_area"],
    "window_area_velocity": ["object_area", "velocity"],
    "window_area_acceleration": ["object_area", "ax", "ay", "az"],
}

TARGET_COLUMN = "finger_width"
WINDOW_SIZE = 15
EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 32
VAL_TRIALS = 2
EARLY_STOPPING_PATIENCE = 150
SEED = 42

# Keep this trial unseen for comparison with the old single-frame MLP.
EXCLUDED_TRIALS = {

}


class WindowMLP(nn.Module):
    def __init__(self, window_size: int, feature_count: int, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * feature_count, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.10),
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


def load_trials(data_root: Path) -> dict[str, pd.DataFrame]:
    all_features = sorted({column for columns in MODEL_SPECS.values() for column in columns})
    required = all_features + [TARGET_COLUMN]
    trials = {}

    for csv_path in sorted(data_root.glob("TRIAL_*/grasp_data.csv")):
        trial_name = csv_path.parent.name
        if trial_name in EXCLUDED_TRIALS:
            print(f"[Hold-out] {trial_name}: excluded from training")
            continue

        frame = pd.read_csv(csv_path)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            print(f"[Skip] {trial_name}: missing {missing}")
            continue

        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.reset_index(drop=True)

        if len(frame) < WINDOW_SIZE:
            print(f"[Skip] {trial_name}: fewer than {WINDOW_SIZE} rows")
            continue
        trials[trial_name] = frame

    if len(trials) < 2:
        raise RuntimeError("At least two usable trials are required")
    return trials


def split_trial_names(trial_names: list[str], val_count: int, seed: int):
    count = min(max(1, val_count), len(trial_names) - 1)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(sorted(trial_names)))
    return sorted(shuffled[count:]), sorted(shuffled[:count])


def make_windows(
    trials: dict[str, pd.DataFrame],
    trial_names: list[str],
    feature_columns: list[str],
    window_size: int,
):
    windows = []
    targets = []

    for trial_name in trial_names:
        frame = trials[trial_name]
        used_columns = feature_columns + [TARGET_COLUMN]
        values = frame[used_columns].to_numpy(dtype=np.float32)
        valid = np.isfinite(values).all(axis=1)
        valid &= frame[TARGET_COLUMN].to_numpy(dtype=np.float32) > 0
        valid &= frame["object_area"].to_numpy(dtype=np.float32) > 0

        features = frame[feature_columns].to_numpy(dtype=np.float32)
        target = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)

        for end in range(window_size - 1, len(frame)):
            start = end - window_size + 1
            if not valid[start:end + 1].all():
                continue
            windows.append(features[start:end + 1])
            targets.append(target[end])

    if not windows:
        raise RuntimeError(
            f"No complete {window_size}-frame windows for features {feature_columns}"
        )
    return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def normalize(train_x, val_x, train_y, val_y):
    feature_count = train_x.shape[-1]
    flattened = train_x.reshape(-1, feature_count)
    x_mean = flattened.mean(axis=0).astype(np.float32)
    x_scale = flattened.std(axis=0).astype(np.float32)
    x_scale[x_scale < 1e-8] = 1.0

    y_mean = np.float32(train_y.mean())
    y_scale = np.float32(train_y.std())
    if y_scale < 1e-8:
        y_scale = np.float32(1.0)

    return (
        (train_x - x_mean) / x_scale,
        (val_x - x_mean) / x_scale,
        (train_y - y_mean) / y_scale,
        (val_y - y_mean) / y_scale,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    )


def calculate_metrics(target: torch.Tensor, prediction: torch.Tensor):
    error = prediction - target
    rmse = torch.sqrt(torch.mean(error.square())).item()
    mae = torch.mean(torch.abs(error)).item()
    return rmse, mae


def train_one_model(
    model_name: str,
    feature_columns: list[str],
    trials: dict[str, pd.DataFrame],
    train_names: list[str],
    val_names: list[str],
    args,
    device: torch.device,
) -> Path:
    train_x, train_y = make_windows(
        trials, train_names, feature_columns, args.window_size
    )
    val_x, val_y = make_windows(
        trials, val_names, feature_columns, args.window_size
    )

    raw_val_y = val_y.copy()
    (
        train_x,
        val_x,
        train_y,
        _val_y_normalized,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    ) = normalize(train_x, val_x, train_y, val_y)

    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_x_tensor = torch.from_numpy(val_x).to(device)
    val_y_tensor = torch.from_numpy(raw_val_y).to(device)

    model = WindowMLP(
        args.window_size, len(feature_columns), args.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    print(f"\n[{model_name}] inputs={feature_columns}")
    print(f"train windows={len(train_x)} | validation windows={len(val_x)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        model.eval()
        with torch.inference_mode():
            val_prediction = model(val_x_tensor) * float(y_scale) + float(y_mean)
            val_rmse, val_mae = calculate_metrics(val_y_tensor, val_prediction)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 50 == 0:
            average_loss = total_loss / len(dataset)
            print(
                f"[{model_name}] epoch {epoch:4d}/{args.epochs} "
                f"train_loss={average_loss:.5f} "
                f"val_RMSE={val_rmse:.2f}px val_MAE={val_mae:.2f}px"
            )

        if epochs_without_improvement >= args.patience:
            print(f"[{model_name}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"Training failed for {model_name}")

    checkpoint = {
        "model_name": model_name,
        "model_class": "WindowMLP",
        "state_dict": {name: value.cpu() for name, value in best_state.items()},
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "window_size": args.window_size,
        "feature_count": len(feature_columns),
        "hidden_size": args.hidden_size,
        "x_mean": torch.tensor(x_mean, dtype=torch.float32),
        "x_scale": torch.tensor(x_scale, dtype=torch.float32),
        "y_mean": float(y_mean),
        "y_scale": float(y_scale),
        "train_trials": train_names,
        "validation_trials": val_names,
        "excluded_trials": sorted(EXCLUDED_TRIALS),
        "train_windows": len(train_x),
        "validation_windows": len(val_x),
        "best_epoch": best_epoch,
        "best_val_rmse_px": best_rmse,
        "best_val_mae_px": best_mae,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f"{model_name}.pth"
    torch.save(checkpoint, save_path)
    print(
        f"[{model_name}] saved: {save_path} | best epoch={best_epoch} "
        f"RMSE={best_rmse:.2f}px MAE={best_mae:.2f}px"
    )
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train three time-window MLP models")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--val-trials", type=int, default=VAL_TRIALS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.window_size < 2:
        parser.error("--window-size must be at least 2")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trials = load_trials(args.data_root)
    train_names, val_names = split_trial_names(
        list(trials), args.val_trials, args.seed
    )

    print(f"Device: {device}")
    print(f"Window: {args.window_size} frames")
    print(f"Train trials: {train_names}")
    print(f"Validation trials: {val_names}")

    for model_name, feature_columns in MODEL_SPECS.items():
        set_seed(args.seed)
        train_one_model(
            model_name,
            feature_columns,
            trials,
            train_names,
            val_names,
            args,
            device,
        )


if __name__ == "__main__":
    main()
