"""Offline inference for both single-frame and time-window MLP .pth models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Edit these paths, then run this script without arguments.
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "pth_models"
    / "window_area_velocity.pth"
)

CSV_PATH = Path(
    r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData"
    r"\TRIAL_20260924_151054\grasp_data.csv"
)

OUTPUT_PATH = None  # None: automatically save beside the input CSV


class FingerWidthRegressor(nn.Module):
    """Original single-frame MLP."""

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


class WindowMLP(nn.Module):
    """15-frame window MLP used by train_window_mlp.py."""

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


class FingerWidthPredictor:
    def __init__(self, model_path: str | Path, device: str = "auto"):
        self.model_path = Path(model_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # This checkpoint is produced locally by train_window_mlp.py.
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model_name = str(checkpoint["model_name"])
        self.model_class = str(checkpoint.get("model_class", "FingerWidthRegressor"))
        self.best_val_rmse_px = float(
            checkpoint.get("best_val_rmse_px", float("nan"))
        )
        self.best_val_mae_px = float(
            checkpoint.get("best_val_mae_px", float("nan"))
        )

        if self.model_class == "WindowMLP":
            self.is_window_model = True
            self.feature_columns = list(checkpoint["feature_columns"])
            self.window_size = int(checkpoint["window_size"])
            self.model = WindowMLP(
                window_size=self.window_size,
                feature_count=int(checkpoint["feature_count"]),
                hidden_size=int(checkpoint["hidden_size"]),
            ).to(self.device)
        else:
            self.is_window_model = False
            self.feature_columns = list(checkpoint["input_columns"])
            self.window_size = 1
            self.model = FingerWidthRegressor(
                input_size=int(checkpoint["input_size"]),
                hidden_size=int(checkpoint["hidden_size"]),
            ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.x_mean = checkpoint["x_mean"].to(self.device)
        self.x_scale = checkpoint["x_scale"].to(self.device)
        self.y_mean = float(checkpoint["y_mean"])
        self.y_scale = float(checkpoint["y_scale"])

    def predict_batch(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)

        if self.is_window_model:
            expected = (self.window_size, len(self.feature_columns))
            if array.ndim == 2:
                array = array[None, :, :]
            if array.ndim != 3 or tuple(array.shape[1:]) != expected:
                raise ValueError(
                    f"Expected [batch, {expected[0]}, {expected[1]}], got {array.shape}"
                )
        else:
            if array.ndim == 1:
                array = array[None, :]
            if array.ndim != 2 or array.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"Expected [batch, {len(self.feature_columns)}], got {array.shape}"
                )

        tensor = torch.from_numpy(array).to(self.device)
        normalized = (tensor - self.x_mean) / self.x_scale
        with torch.inference_mode():
            prediction = self.model(normalized) * self.y_scale + self.y_mean
        return prediction.detach().cpu().numpy()


def predict_csv(
    predictor: FingerWidthPredictor,
    csv_path: Path,
    output_path: Path | None,
) -> Path:
    data = pd.read_csv(csv_path)
    missing = [
        column for column in predictor.feature_columns if column not in data.columns
    ]
    if missing:
        raise KeyError(f"CSV is missing model inputs: {missing}")

    numeric = data[predictor.feature_columns].apply(pd.to_numeric, errors="coerce")
    raw = numeric.to_numpy(dtype=np.float32)
    row_valid = np.isfinite(raw).all(axis=1)

    if "object_area" in predictor.feature_columns:
        area_index = predictor.feature_columns.index("object_area")
        row_valid &= raw[:, area_index] > 0

    predictions = np.full(len(data), np.nan, dtype=np.float32)

    if predictor.is_window_model:
        windows = []
        end_rows = []
        for end in range(predictor.window_size - 1, len(data)):
            start = end - predictor.window_size + 1
            if not row_valid[start:end + 1].all():
                continue
            windows.append(raw[start:end + 1])
            end_rows.append(end)
        if windows:
            predicted = predictor.predict_batch(np.asarray(windows, dtype=np.float32))
            predictions[np.asarray(end_rows)] = predicted
    else:
        if row_valid.any():
            predictions[row_valid] = predictor.predict_batch(raw[row_valid])

    data["predicted_finger_width"] = predictions

    if output_path is None:
        output_path = csv_path.with_name(
            f"{csv_path.stem}_{predictor.model_name}_predictions.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    predicted_valid = np.isfinite(predictions)
    print(f"Predicted rows: {int(predicted_valid.sum())}/{len(data)}")

    if "finger_width" in data.columns:
        target = pd.to_numeric(
            data["finger_width"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        scored = predicted_valid & np.isfinite(target) & (target > 0)
        if scored.any():
            error = predictions[scored] - target[scored]
            rmse = float(np.sqrt(np.mean(error**2)))
            mae = float(np.mean(np.abs(error)))
            print(
                f"CSV RMSE={rmse:.2f}px | MAE={mae:.2f}px "
                f"| scored rows={int(scored.sum())}"
            )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline inference for single-frame and window MLP models"
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    predictor = FingerWidthPredictor(args.model, args.device)
    print(
        f"Loaded {predictor.model_name} | inputs={predictor.feature_columns} "
        f"| window={predictor.window_size} frames | device={predictor.device}"
    )
    print(
        f"Saved validation RMSE={predictor.best_val_rmse_px:.2f}px "
        f"| MAE={predictor.best_val_mae_px:.2f}px"
    )

    saved_path = predict_csv(predictor, args.csv, args.output)
    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()