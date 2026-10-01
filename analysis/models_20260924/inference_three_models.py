"""
Offline test for three finger-width prediction models.

Models:
1. object_area
2. object_area + velocity
3. object_area + acceleration

The script automatically tests all grasp_data.csv files
inside TEST_ROOT and reports RMSE / MAE for every trial.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent / "pth_models"

MODEL_PATHS = [
    MODEL_DIR / "window_area_only.pth",
    MODEL_DIR / "window_area_velocity.pth",
    MODEL_DIR / "window_area_acceleration.pth",
]

TEST_ROOT = Path(
    r"D:\Code\Multimodal_Imitation_Learning\Data\Validation\D"
)

OUTPUT_PATH = Path(__file__).resolve().parent / "offline_test_results.csv"
PLOT_DIR = Path(__file__).resolve().parent / "offline_test_plots"

MODEL_LABELS = {
    "window_area_only": "Area only",
    "window_area_velocity": "Area + velocity",
    "window_area_acceleration": "Area + acceleration",
}

MODEL_COLORS = {
    "window_area_only": "#1f77b4",
    "window_area_velocity": "#ff7f0e",
    "window_area_acceleration": "#2ca02c",
}

DEVICE = "auto"


# ============================================================
# Models
# ============================================================

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


# ============================================================
# Predictor
# ============================================================

class FingerWidthPredictor:
    def __init__(self, model_path: str | Path, device: str = "auto"):

        self.model_path = Path(model_path)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

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

            expected = (
                self.window_size,
                len(self.feature_columns),
            )

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
            prediction = self.model(normalized)
            prediction = prediction * self.y_scale + self.y_mean

        return prediction.detach().cpu().numpy()


# ============================================================
# Evaluate one CSV
# ============================================================

def evaluate_csv(predictor: FingerWidthPredictor, csv_path: Path):

    data = pd.read_csv(csv_path)

    missing = [
        column
        for column in predictor.feature_columns
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            f"CSV is missing model inputs: {missing}"
        )

    if "finger_width" not in data.columns:
        raise KeyError(
            "CSV is missing target column: finger_width"
        )

    numeric = data[
        predictor.feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    raw = numeric.to_numpy(
        dtype=np.float32
    )

    row_valid = np.isfinite(
        raw
    ).all(
        axis=1
    )

    if "object_area" in predictor.feature_columns:

        area_index = predictor.feature_columns.index(
            "object_area"
        )

        row_valid &= raw[:, area_index] > 0

    predictions = np.full(
        len(data),
        np.nan,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Window model
    # --------------------------------------------------------

    if predictor.is_window_model:

        windows = []
        end_rows = []

        for end in range(
            predictor.window_size - 1,
            len(data),
        ):

            start = (
                end
                - predictor.window_size
                + 1
            )

            if not row_valid[
                start:end + 1
            ].all():

                continue

            windows.append(
                raw[start:end + 1]
            )

            end_rows.append(
                end
            )

        if windows:

            predicted = predictor.predict_batch(
                np.asarray(
                    windows,
                    dtype=np.float32,
                )
            )

            predictions[
                np.asarray(end_rows)
            ] = predicted

    # --------------------------------------------------------
    # Single-frame model
    # --------------------------------------------------------

    else:

        if row_valid.any():

            predictions[
                row_valid
            ] = predictor.predict_batch(
                raw[row_valid]
            )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    target = pd.to_numeric(
        data["finger_width"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float32
    )

    valid = (
        np.isfinite(predictions)
        & np.isfinite(target)
        & (target > 0)
    )

    if not valid.any():
        return None

    error = (
        predictions[valid]
        - target[valid]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    if "frame" in data.columns:
        frames = pd.to_numeric(
            data["frame"],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
    else:
        frames = np.arange(len(data), dtype=np.float32)

    return {
        "rmse": rmse,
        "mae": mae,
        "samples": int(
            valid.sum()
        ),
        "frames": frames,
        "target": target,
        "predictions": predictions,
    }


# ============================================================
# Plots
# ============================================================

def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def create_plots(
    result_df: pd.DataFrame,
    curve_results: dict[str, dict[str, dict]],
) -> None:

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    model_order = [
        name
        for name in MODEL_LABELS
        if name in set(result_df["model"])
    ]

    # --------------------------------------------------------
    # Overall RMSE / MAE comparison
    # --------------------------------------------------------

    summary = result_df.groupby("model").agg(
        mean_rmse=("rmse_px", "mean"),
        mean_mae=("mae_px", "mean"),
        std_rmse=("rmse_px", "std"),
        std_mae=("mae_px", "std"),
    ).reindex(model_order)

    x = np.arange(len(model_order))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 6))
    rmse_bars = axis.bar(
        x - width / 2,
        summary["mean_rmse"],
        width,
        yerr=summary["std_rmse"],
        capsize=5,
        label="RMSE",
        color="#d95f02",
        alpha=0.88,
    )
    mae_bars = axis.bar(
        x + width / 2,
        summary["mean_mae"],
        width,
        yerr=summary["std_mae"],
        capsize=5,
        label="MAE",
        color="#1b9e77",
        alpha=0.88,
    )
    axis.bar_label(rmse_bars, fmt="%.1f", padding=3)
    axis.bar_label(mae_bars, fmt="%.1f", padding=3)
    axis.set_xticks(x, [model_label(name) for name in model_order])
    axis.set_ylabel("Error (px)")
    axis.set_title("Overall test error by model")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "overall_model_comparison.png", dpi=180)
    plt.close(fig)

    # --------------------------------------------------------
    # RMSE heatmap: every trial x every model
    # --------------------------------------------------------

    pivot = result_df.pivot(
        index="trial",
        columns="model",
        values="rmse_px",
    ).reindex(columns=model_order)

    values = pivot.to_numpy(dtype=float)
    figure_height = max(5.5, 0.55 * len(pivot.index) + 2.0)
    fig, axis = plt.subplots(figsize=(10, figure_height))
    image = axis.imshow(values, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(
        np.arange(len(model_order)),
        [model_label(name) for name in model_order],
    )
    axis.set_yticks(
        np.arange(len(pivot.index)),
        [name.replace("TRIAL_", "") for name in pivot.index],
    )
    axis.set_title("RMSE by trial and model (px)")

    threshold = np.nanmean(values)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=9,
                )

    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("RMSE (px)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rmse_heatmap.png", dpi=180)
    plt.close(fig)

    # --------------------------------------------------------
    # Ground truth and all three model predictions per trial
    # --------------------------------------------------------

    curve_dir = PLOT_DIR / "prediction_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    for trial_name, by_model in sorted(curve_results.items()):
        if not by_model:
            continue

        first_result = next(iter(by_model.values()))
        frames = first_result["frames"]
        target = first_result["target"]

        fig, axis = plt.subplots(figsize=(12, 6))
        axis.plot(
            frames,
            target,
            color="black",
            linewidth=2.6,
            label="Measured finger width",
            zorder=5,
        )

        for model_name in model_order:
            result = by_model.get(model_name)
            if result is None:
                continue
            axis.plot(
                result["frames"],
                result["predictions"],
                color=MODEL_COLORS.get(model_name),
                linewidth=2.0,
                alpha=0.9,
                label=(
                    f"{model_label(model_name)} "
                    f"(RMSE {result['rmse']:.1f}px)"
                ),
            )

        axis.set_title(trial_name)
        axis.set_xlabel("Frame")
        axis.set_ylabel("Finger width (px)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(curve_dir / f"{trial_name}.png", dpi=170)
        plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    for model_path in MODEL_PATHS:

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

    # --------------------------------------------------------
    # Find all test CSV files
    # --------------------------------------------------------

    csv_files = sorted(
        TEST_ROOT.glob(
            "*/grasp_data.csv"
        )
    )

    if not csv_files:

        raise FileNotFoundError(
            f"No grasp_data.csv found in: {TEST_ROOT}"
        )

    print(
        f"\nTest root: {TEST_ROOT}"
    )

    print(
        f"Test trials: {len(csv_files)}"
    )

    for csv_path in csv_files:

        print(
            f"  {csv_path.parent.name}"
        )

    results = []
    curve_results: dict[str, dict[str, dict]] = {}

    # --------------------------------------------------------
    # Test each model
    # --------------------------------------------------------

    for model_path in MODEL_PATHS:

        predictor = FingerWidthPredictor(
            model_path,
            DEVICE,
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Model: {predictor.model_name}"
        )

        print(
            f"Inputs: {predictor.feature_columns}"
        )

        print(
            f"Window: {predictor.window_size}"
        )

        print(
            f"Device: {predictor.device}"
        )

        print(
            f"Validation RMSE: {predictor.best_val_rmse_px:.2f}px"
        )

        print(
            f"Validation MAE: {predictor.best_val_mae_px:.2f}px"
        )

        print(
            "=" * 80
        )

        # ----------------------------------------------------
        # Test every trial
        # ----------------------------------------------------

        for csv_path in csv_files:

            trial_name = csv_path.parent.name

            try:

                result = evaluate_csv(
                    predictor,
                    csv_path,
                )

                if result is None:

                    print(
                        f"{trial_name}: no valid samples"
                    )

                    continue

                print(
                    f"{trial_name}: "
                    f"RMSE={result['rmse']:.2f}px | "
                    f"MAE={result['mae']:.2f}px | "
                    f"N={result['samples']}"
                )

                results.append(
                    {
                        "trial": trial_name,
                        "model": predictor.model_name,
                        "rmse_px": result["rmse"],
                        "mae_px": result["mae"],
                        "samples": result["samples"],
                    }
                )

                curve_results.setdefault(
                    trial_name,
                    {},
                )[predictor.model_name] = result

            except Exception as error:

                print(
                    f"{trial_name}: ERROR: {error}"
                )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not results:

        print(
            "\nNo valid test results."
        )

        return

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OVERALL TEST RESULTS"
    )

    print(
        "=" * 80
    )

    summary = result_df.groupby(
        "model"
    ).agg(
        mean_rmse_px=(
            "rmse_px",
            "mean",
        ),
        mean_mae_px=(
            "mae_px",
            "mean",
        ),
        std_rmse_px=(
            "rmse_px",
            "std",
        ),
        std_mae_px=(
            "mae_px",
            "std",
        ),
        trials=(
            "trial",
            "count",
        ),
        samples=(
            "samples",
            "sum",
        ),
    )

    print(
        summary.round(2)
    )

    create_plots(
        result_df,
        curve_results,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        f"Plots saved: {PLOT_DIR}"
    )


if __name__ == "__main__":
    main()
