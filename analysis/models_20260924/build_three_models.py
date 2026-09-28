from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData")
OUTPUT_DIR = Path(__file__).resolve().parent
RIDGE_ALPHA = 1.0

MODEL_SPECS = {
    "area_only": ["object_area"],
    "area_velocity": ["object_area", "velocity"],
    "area_acceleration": ["object_area", "ax", "ay", "az"],
}


def load_data() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_ROOT.glob("TRIAL_*/grasp_data.csv")):
        df = pd.read_csv(path)
        df["trial"] = path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No grasp_data.csv found under {DATA_ROOT}")
    data = pd.concat(frames, ignore_index=True)
    required = ["finger_width", "object_area", "velocity", "ax", "ay", "az"]
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    finite = np.isfinite(data[required]).all(axis=1)
    valid = finite & (data["finger_width"] > 0) & (data["object_area"] > 0)
    return data.loc[valid].reset_index(drop=True)


def transform(raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (raw - mean) / scale


def fit_model(raw: np.ndarray, y: np.ndarray) -> dict:
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    phi = transform(raw, mean, scale)
    design = np.column_stack([np.ones(len(phi)), phi])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "intercept": float(coef[0]), "coef": coef[1:]}


def predict(model: dict, raw: np.ndarray) -> np.ndarray:
    return model["intercept"] + transform(raw, model["mean"], model["scale"]) @ model["coef"]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "rmse_px": float(np.sqrt(np.mean(err**2))),
        "mae_px": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def serializable_model(model: dict, columns: list[str]) -> dict:
    return {
        "input_columns": columns,
        "basis": columns,
        "raw_mean": model["mean"].tolist(),
        "raw_scale": model["scale"].tolist(),
        "intercept": model["intercept"],
        "coefficients": model["coef"].tolist(),
        "ridge_alpha": RIDGE_ALPHA,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    y = data["finger_width"].to_numpy(float)
    trials = sorted(data["trial"].unique())
    all_predictions = data[["trial", "frame", "finger_width", "object_area", "velocity", "ax", "ay", "az"]].copy()
    summary_rows = []
    fold_rows = []
    final_models = {}

    for model_name, columns in MODEL_SPECS.items():
        oof = np.full(len(data), np.nan)
        for trial in trials:
            test = data["trial"].eq(trial).to_numpy()
            train = ~test
            fitted = fit_model(data.loc[train, columns].to_numpy(float), y[train])
            oof[test] = predict(fitted, data.loc[test, columns].to_numpy(float))
            fold_rows.append({"model": model_name, "test_trial": trial, "n_test": int(test.sum()), **metrics(y[test], oof[test])})
        all_predictions[f"pred_{model_name}"] = oof
        summary_rows.append({"model": model_name, "features": "+".join(columns), "n_rows": len(data), "n_trials": len(trials), **metrics(y, oof)})
        final_models[model_name] = serializable_model(fit_model(data[columns].to_numpy(float), y), columns)

    summary = pd.DataFrame(summary_rows).sort_values("rmse_px")
    folds = pd.DataFrame(fold_rows)
    summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "leave_one_trial_out_metrics.csv", index=False)
    all_predictions.to_csv(OUTPUT_DIR / "out_of_trial_predictions.csv", index=False)
    (OUTPUT_DIR / "trained_models.json").write_text(json.dumps(final_models, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Three regression models",
        "",
        f"Data: {len(data)} valid frames from {len(trials)} trials.",
        "Validation: leave one complete trial out. Models are linear ridge regressions.",
        "",
        "| Model | Inputs | RMSE (px) | MAE (px) | R² |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.model} | {row.features} | {row.rmse_px:.3f} | {row.mae_px:.3f} | {row.r2:.3f} |")
    lines += [
        "",
        "The velocity values are the existing single-axis integrated values and inherit their known drift limitations.",
        "The acceleration model uses ax, ay, and az together.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

