"""Train two causal window MLP models with single-size or generalization splits.

Compare object_area against object_area + ax + ay + az.
Velocity is neither required nor used. Read grasp_data_smoothed.csv without
additional smoothing. Keep complete test trials out of training and normalize
using training data only. Defaults: 15-frame windows and 1000 epochs.
Select RUN_MODE in the configuration. Generalization trains on 3/4.5 cm,
holds out trials within each size, and evaluates all 4 cm trials separately.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\临时数据\D10\3")
OUTPUT_DIR = Path(__file__).resolve().parent / "window_mlp_d9_no_velocity"

# Choose in code: "single_size" or "generalization".
# single_size uses DATA_ROOT, TEST_TRIALS and TEST_TRIAL_NAMES below.
RUN_MODE = "generalization"
GENERALIZATION_ROOT = Path(r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\临时数据\D10")
GENERALIZATION_OUTPUT_DIR = Path(__file__).resolve().parent / "window_mlp_d10_generalization"
TRAIN_SIZE_FOLDERS = ("3", "4.5")
GENERALIZATION_SIZE_FOLDER = "4"
# Hold out complete trials separately within each training size.
SELF_TEST_TRIALS_BY_FOLDER = {"3": 5, "4.5": 4}
# Optional exact folder names; [] uses the seeded split for that size.
SELF_TEST_NAMES_BY_FOLDER = {"3": [], "4.5": []}

MODEL_SPECS = {
    "window_area_only": ["object_area"],
    "window_area_acceleration": ["object_area", "ax", "ay", "az"],
}

MODEL_LABELS = {
    "window_area_only": "Area only",
    "window_area_acceleration": "Area + acceleration",
}

TARGET_COLUMN = "finger_width"
WINDOW_SIZE = 15
EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 32
SEED = 42
INPUT_CSV = "grasp_data_smoothed.csv"
TEST_TRIALS = 4

# Edit this list to choose fixed test trials. All remaining trials train.
# Set to None or [] to use the random split controlled by TEST_TRIALS and SEED.
TEST_TRIAL_NAMES = []

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


def trial_size(trial_name: str) -> float:
    size_tokens = {
        "_2cm_": 2.0,
        "_2p5cm_": 2.5,
        "_3cm_": 3.0,
        "_3p5cm_": 3.5,
        "_4cm_": 4.0,
        "_4p5cm_": 4.5,
    }

    for token, size in size_tokens.items():
        if token in trial_name:
            return size

    raise ValueError(f"Cannot determine object size from trial name: {trial_name}")


def load_trials(data_root: Path) -> dict[str, pd.DataFrame]:
    all_features = sorted({column for columns in MODEL_SPECS.values() for column in columns})
    required = ["frame"] + all_features + [TARGET_COLUMN]
    trials = {}

    for csv_path in sorted(data_root.glob(f"TRIAL_*/{INPUT_CSV}")):
        trial_name = csv_path.parent.name
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

    if not trials:
        raise RuntimeError(f"No usable {INPUT_CSV} trials were found in {data_root} (expected TRIAL_*/{INPUT_CSV})")

    return trials


def split_train_test(trials: dict[str, pd.DataFrame], test_trials: int, seed: int):
    names = sorted(trials)
    if len(names) <= test_trials:
        raise ValueError(
            f"Found {len(names)} trial(s), but test_trials={test_trials}."
        )

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(names))
    test_names = sorted(shuffled[:test_trials])
    train_names = sorted(shuffled[test_trials:])
    return train_names, test_names


def choose_split(trials, test_names, test_count, seed):
    if not test_names:
        return split_train_test(trials, test_count, seed)
    test_names = sorted(set(test_names))
    missing = sorted(set(test_names) - set(trials))
    if missing:
        raise ValueError(f"Test trials not found among usable input trials: {missing}")
    train_names = sorted(set(trials) - set(test_names))
    if not train_names:
        raise ValueError("At least one trial must remain for training")
    return train_names, test_names


def prepare_split(args):
    if args.mode == "single_size":
        trials = load_trials(args.data_root)
        if len({trial_size(name) for name in trials}) != 1:
            raise ValueError("single_size requires exactly one object size")
        train, test = choose_split(trials, args.test_names, args.test_trials, args.seed)
        return trials, train, test, []
    if len(set(TRAIN_SIZE_FOLDERS)) != len(TRAIN_SIZE_FOLDERS):
        raise ValueError("Training size folders must be unique")
    if GENERALIZATION_SIZE_FOLDER in TRAIN_SIZE_FOLDERS:
        raise ValueError("Generalization size must not be used for training")
    trials, train, test = {}, [], []
    for folder in (*TRAIN_SIZE_FOLDERS, GENERALIZATION_SIZE_FOLDER):
        part = load_trials(args.generalization_root / folder)
        if any(trial_size(name) != float(folder) for name in part):
            raise ValueError(f"Trial size does not match folder {folder}")
        if set(part) & set(trials):
            raise ValueError("Duplicate trial names across size folders")
        trials.update(part)
        if folder == GENERALIZATION_SIZE_FOLDER:
            generalization = sorted(part)
        else:
            size_train, size_test = choose_split(
                part, SELF_TEST_NAMES_BY_FOLDER.get(folder),
                SELF_TEST_TRIALS_BY_FOLDER[folder], args.seed,
            )
            train.extend(size_train)
            test.extend(size_test)
    return trials, sorted(train), sorted(test), generalization


def make_windows(trials: dict[str, pd.DataFrame], trial_names: list[str], feature_columns: list[str], window_size: int):
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
        raise RuntimeError(f"No complete {window_size}-frame windows for features {feature_columns}")

    return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def make_trial_windows(frame: pd.DataFrame, feature_columns: list[str], window_size: int):
    used_columns = feature_columns + [TARGET_COLUMN]
    values = frame[used_columns].to_numpy(dtype=np.float32)

    valid = np.isfinite(values).all(axis=1)
    valid &= frame[TARGET_COLUMN].to_numpy(dtype=np.float32) > 0
    valid &= frame["object_area"].to_numpy(dtype=np.float32) > 0

    features = frame[feature_columns].to_numpy(dtype=np.float32)
    target = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    frame_numbers = frame["frame"].to_numpy(dtype=float)

    windows = []
    targets = []
    predicted_frames = []

    for end in range(window_size - 1, len(frame)):
        start = end - window_size + 1
        if not valid[start:end + 1].all():
            continue

        windows.append(features[start:end + 1])
        targets.append(target[end])
        predicted_frames.append(frame_numbers[end])

    if not windows:
        raise RuntimeError("No complete windows in this test trial")

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(predicted_frames, dtype=float),
    )


def normalization_from_training(train_x: np.ndarray, train_y: np.ndarray):
    feature_count = train_x.shape[-1]
    flattened = train_x.reshape(-1, feature_count)

    x_mean = flattened.mean(axis=0).astype(np.float32)
    x_scale = flattened.std(axis=0).astype(np.float32)
    x_scale[x_scale < 1e-8] = 1.0

    y_mean = np.float32(train_y.mean())
    y_scale = np.float32(train_y.std())
    if y_scale < 1e-8:
        y_scale = np.float32(1.0)

    return x_mean, x_scale, y_mean, y_scale


def regression_metrics(target: np.ndarray, prediction: np.ndarray):
    error = prediction - target
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))

    ss_res = float(np.sum((target - prediction) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    return rmse, mae, r2


def train_one_model(model_name: str, feature_columns: list[str], trials: dict[str, pd.DataFrame], train_names: list[str], args, device: torch.device) -> Path:
    train_x, train_y = make_windows(trials, train_names, feature_columns, args.window_size)

    x_mean, x_scale, y_mean, y_scale = normalization_from_training(train_x, train_y)

    train_x = (train_x - x_mean) / x_scale
    train_y = (train_y - y_mean) / y_scale

    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))

    model = WindowMLP(args.window_size, len(feature_columns), args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    print(f"\n[{model_name}] inputs={feature_columns}")
    print(f"train trials={len(train_names)} | train windows={len(train_x)}")

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

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            average_loss = total_loss / len(dataset)
            print(f"[{model_name}] epoch {epoch:4d}/{args.epochs} train_loss={average_loss:.5f}")

    checkpoint = {
        "model_name": model_name,
        "model_class": "WindowMLP",
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
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
        "train_windows": len(train_x),
        "epochs": args.epochs,
    }

    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    save_path = model_dir / f"{model_name}.pth"
    torch.save(checkpoint, save_path)

    print(f"[{model_name}] saved: {save_path}")
    return save_path


def predict_trial(checkpoint_path: Path, frame: pd.DataFrame, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    feature_columns = checkpoint["feature_columns"]
    window_size = checkpoint["window_size"]

    x_raw, target, predicted_frames = make_trial_windows(frame, feature_columns, window_size)

    x_mean = checkpoint["x_mean"].cpu().numpy()
    x_scale = checkpoint["x_scale"].cpu().numpy()

    x = torch.from_numpy((x_raw - x_mean) / x_scale).to(device)

    model = WindowMLP(window_size, checkpoint["feature_count"], checkpoint["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.inference_mode():
        prediction = model(x).cpu().numpy() * checkpoint["y_scale"] + checkpoint["y_mean"]

    return target.astype(float), prediction.astype(float), predicted_frames


def evaluate_and_plot(trials: dict[str, pd.DataFrame], test_names: list[str], model_paths: dict[str, Path], args, device: torch.device, prediction_folder="test_predictions"):
    prediction_dir = args.output_dir / prediction_folder
    prediction_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for trial_name in test_names:
        frame = trials[trial_name].copy()
        full_frames = frame["frame"].to_numpy(dtype=float)
        full_target = frame[TARGET_COLUMN].to_numpy(dtype=float)

        predictions = {}
        metrics = {}
        prediction_frames_by_model = {}

        for model_name, model_path in model_paths.items():
            target, prediction, prediction_frames = predict_trial(model_path, frame, device)
            rmse, mae, r2 = regression_metrics(target, prediction)

            predictions[model_name] = prediction
            metrics[model_name] = (rmse, mae, r2)

            prediction_frames_by_model[model_name] = prediction_frames

            rows.append({
                "trial": trial_name,
                "object_size_cm": trial_size(trial_name),
                "model": model_name,
                "rmse_px": rmse,
                "mae_px": mae,
                "r2": r2,
            })

        # Plot the full measured trajectory.
        # Window-MLP predictions start only when 15 frames are available.
        plt.figure(figsize=(14, 6))
        plt.plot(full_frames, full_target, linewidth=3, label="Measured finger width")

        for model_name in MODEL_SPECS:
            rmse = metrics[model_name][0]
            plt.plot(prediction_frames_by_model[model_name], predictions[model_name], linewidth=2, label=f"{MODEL_LABELS[model_name]} (RMSE {rmse:.1f}px)")

        plt.title(trial_name)
        plt.xlabel("Frame")
        plt.ylabel("Finger width (px)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(prediction_dir / f"{trial_name}.png", dpi=200)
        plt.close()

    return pd.DataFrame(rows)


def make_overall_summary(metrics: pd.DataFrame):
    rows = []

    for model_name in MODEL_SPECS:
        part = metrics.loc[metrics["model"] == model_name]

        rows.append({
            "model": model_name,
            "rmse_mean_px": part["rmse_px"].mean(),
            "rmse_sd_px": part["rmse_px"].std(ddof=1),
            "mae_mean_px": part["mae_px"].mean(),
            "mae_sd_px": part["mae_px"].std(ddof=1),
            "r2_mean": part["r2"].mean(),
            "r2_sd": part["r2"].std(ddof=1),
        })

    return pd.DataFrame(rows)


def plot_overall_comparison(summary: pd.DataFrame, output_dir: Path):
    order = list(MODEL_SPECS.keys())
    x = np.arange(len(order))
    width = 0.36

    summary = summary.set_index("model").loc[order].reset_index()

    rmse_mean = summary["rmse_mean_px"].to_numpy()
    rmse_sd = summary["rmse_sd_px"].to_numpy()
    mae_mean = summary["mae_mean_px"].to_numpy()
    mae_sd = summary["mae_sd_px"].to_numpy()

    plt.figure(figsize=(12, 7))

    bars_rmse = plt.bar(x - width / 2, rmse_mean, width, yerr=rmse_sd, capsize=7, label="RMSE")
    bars_mae = plt.bar(x + width / 2, mae_mean, width, yerr=mae_sd, capsize=7, label="MAE")

    for bar, value in zip(bars_rmse, rmse_mean):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}", ha="center", va="bottom")

    for bar, value in zip(bars_mae, mae_mean):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}", ha="center", va="bottom")

    plt.xticks(x, [MODEL_LABELS[name] for name in order])
    plt.ylabel("Error (px)")
    plt.title("Overall test error by model")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overall_model_comparison.png", dpi=200)
    plt.close()


def make_size_summary(metrics: pd.DataFrame):
    rows = []

    for size in sorted(metrics["object_size_cm"].dropna().unique()):
        for model_name in MODEL_SPECS:
            part = metrics.loc[(metrics["object_size_cm"] == size) & (metrics["model"] == model_name)]

            rows.append({
                "object_size_cm": size,
                "model": model_name,
                "rmse_mean_px": part["rmse_px"].mean(),
                "rmse_sd_px": part["rmse_px"].std(ddof=1),
                "mae_mean_px": part["mae_px"].mean(),
                "mae_sd_px": part["mae_px"].std(ddof=1),
                "r2_mean": part["r2"].mean(),
                "r2_sd": part["r2"].std(ddof=1),
            })

    return pd.DataFrame(rows)


def save_split(train_names: list[str], test_names: list[str], output_dir: Path, generalization_names=()):
    rows = []

    for name in train_names:
        rows.append({"trial": name, "object_size_cm": trial_size(name), "split": "train"})

    for name in test_names:
        rows.append({"trial": name, "object_size_cm": trial_size(name), "split": "test"})

    for name in generalization_names:
        rows.append({"trial": name, "object_size_cm": trial_size(name), "split": "generalization_test"})

    pd.DataFrame(rows).sort_values(["object_size_cm", "split", "trial"]).to_csv(output_dir / "data_split.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare area-only and area-plus-acceleration Window MLP models")
    parser.add_argument("--mode", choices=["single_size", "generalization"], default=RUN_MODE)
    parser.add_argument("--generalization-root", type=Path, default=GENERALIZATION_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-trials", type=int, default=TEST_TRIALS)
    parser.add_argument(
        "--test-names", nargs="+", default=TEST_TRIAL_NAMES,
        help="Exact trial folder names to hold out; overrides --test-trials.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = GENERALIZATION_OUTPUT_DIR if args.mode == "generalization" else OUTPUT_DIR

    if args.window_size < 2:
        parser.error("--window-size must be at least 2")

    if args.test_trials < 1:
        parser.error("--test-trials must be at least 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        trials, train_names, test_names, generalization_names = prepare_split(args)
    except (ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))

    if not test_names:
        parser.error("At least one test trial is required; set TEST_TRIAL_NAMES or TEST_TRIALS")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_split(train_names, test_names, args.output_dir, generalization_names)

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Window: {args.window_size} frames")
    print(f"Input CSV: {INPUT_CSV}")
    print("Additional smoothing in training: OFF")

    print(f"\nTraining trials: {len(train_names)}")
    for size in sorted({trial_size(name) for name in train_names}):
        names = [name for name in train_names if trial_size(name) == size]
        print(f"  {size:g} cm: {len(names)} -> {', '.join(names)}")

    print(f"\nTest trials: {len(test_names)}")
    for size in sorted({trial_size(name) for name in test_names}):
        names = [name for name in test_names if trial_size(name) == size]
        print(f"  {size:g} cm: {len(names)} -> {', '.join(names)}")

    if generalization_names:
        print(f"\nGeneralization trials (never used in training): {len(generalization_names)}")
        print("  " + ", ".join(generalization_names))

    model_paths = {}

    for model_name, feature_columns in MODEL_SPECS.items():
        set_seed(args.seed)
        model_paths[model_name] = train_one_model(model_name, feature_columns, trials, train_names, args, device)

    metrics = evaluate_and_plot(trials, test_names, model_paths, args, device)
    metrics.to_csv(args.output_dir / "test_trial_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    overall_summary = make_overall_summary(metrics)
    overall_summary.to_csv(args.output_dir / "overall_model_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    plot_overall_comparison(overall_summary, args.output_dir)

    size_summary = make_size_summary(metrics)
    size_summary.to_csv(args.output_dir / "model_summary_by_size.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    if generalization_names:
        generalization_metrics = evaluate_and_plot(
            trials, generalization_names, model_paths, args, device,
            prediction_folder="generalization_predictions",
        )
        generalization_metrics.to_csv(args.output_dir / "generalization_trial_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
        generalization_summary = make_overall_summary(generalization_metrics)
        generalization_summary.to_csv(args.output_dir / "generalization_model_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
        comparison_dir = args.output_dir / "generalization_comparison"
        comparison_dir.mkdir(exist_ok=True)
        plot_overall_comparison(generalization_summary, comparison_dir)
        print("\nGeneralization results (4 cm):")
        for _, row in generalization_summary.iterrows():
            print(f"{MODEL_LABELS[row['model']]}: RMSE {row['rmse_mean_px']:.2f}px | MAE {row['mae_mean_px']:.2f}px | R2 {row['r2_mean']:.3f}")

    print(f"\nTest results across {len(test_names)} held-out trial(s)")
    for _, row in overall_summary.iterrows():
        print(
            f"{MODEL_LABELS[row['model']]}: "
            f"RMSE {row['rmse_mean_px']:.2f} +/- {row['rmse_sd_px']:.2f}px | "
            f"MAE {row['mae_mean_px']:.2f} +/- {row['mae_sd_px']:.2f}px | "
            f"R2 {row['r2_mean']:.3f} +/- {row['r2_sd']:.3f}"
        )

    print(f"\nSaved results to: {args.output_dir}")
    print("  data_split.csv")
    print("  test_trial_metrics.csv")
    print("  overall_model_summary.csv")
    print("  model_summary_by_size.csv")
    print("  overall_model_comparison.png")
    print("  models/*.pth")
    print("  test_predictions/*.png")
    if generalization_names:
        print("  generalization_trial_metrics.csv")
        print("  generalization_model_summary.csv")
        print("  generalization_predictions/*.png")


if __name__ == "__main__":
    main()
