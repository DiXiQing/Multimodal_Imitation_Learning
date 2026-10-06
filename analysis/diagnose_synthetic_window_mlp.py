"""Diagnostic analysis for the D3 synthetic reach-to-grasp dataset.

This script is read-only with respect to trial data. It writes diagnostic CSVs,
plots, and a text report under analysis/synthetic_diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning")
DATA_ROOT = PROJECT_ROOT / "Data" / "临时数据" / "D3"
TRAIN_OUTPUT = DATA_ROOT / "window_mlp_6train_2test"
OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "synthetic_diagnostics"
WINDOW_SIZE = 15
RESERVED_FRAMES = 15
MAX_LAG = 15
COLORS = ("Black", "Blue", "Red")
REQUIRED = ["frame", "t_camera", "finger_width", "object_area", "ax", "ay", "az", "velocity"]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return safe_corr(pd.Series(x).rank(method="average").to_numpy(), pd.Series(y).rank(method="average").to_numpy())


def lag_correlation(reference: np.ndarray, signal: np.ndarray, max_lag: int = MAX_LAG):
    """Return max correlation and lag for signal(t + lag) vs reference(t).

    Positive lag means the signal is delayed relative to the reference.
    """
    lags = []
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            ref = reference[-lag:]
            sig = signal[: len(signal) + lag]
        elif lag > 0:
            ref = reference[:-lag]
            sig = signal[lag:]
        else:
            ref = reference
            sig = signal
        lags.append(lag)
        correlations.append(safe_corr(ref, sig))
    correlations = np.asarray(correlations, dtype=float)
    if not np.isfinite(correlations).any():
        return float("nan"), float("nan")
    index = int(np.nanargmax(correlations))
    return float(lags[index]), float(correlations[index])


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def load_trials() -> dict[str, pd.DataFrame]:
    trials = {}
    for csv_path in sorted(DATA_ROOT.glob("TRIAL_*/grasp_data.csv")):
        frame = pd.read_csv(csv_path)
        missing = [column for column in REQUIRED if column not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path}: missing {missing}")
        for column in REQUIRED:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=REQUIRED).reset_index(drop=True)
        frame["acceleration_mag"] = np.sqrt(frame["ax"] ** 2 + frame["ay"] ** 2 + frame["az"] ** 2)
        trials[csv_path.parent.name] = frame
    if len(trials) != 24:
        raise RuntimeError(f"Expected 24 D3 trials, found {len(trials)}")
    return trials


def calculate_trial_diagnostics(trial_name: str, frame: pd.DataFrame) -> dict:
    part = frame.iloc[RESERVED_FRAMES:].reset_index(drop=True)
    aperture = part["finger_width"].to_numpy(float)
    area = part["object_area"].to_numpy(float)
    velocity = part["velocity"].to_numpy(float)
    acceleration_mag = part["acceleration_mag"].to_numpy(float)
    time = part["t_camera"].to_numpy(float)

    result = {"trial": trial_name, "object_color": trial_name.split("_")[1], "n_movement_frames": len(part)}
    signals = {
        "object_area": area,
        "velocity": velocity,
        "ax": part["ax"].to_numpy(float),
        "ay": part["ay"].to_numpy(float),
        "az": part["az"].to_numpy(float),
        "acceleration_mag": acceleration_mag,
    }
    for name, signal in signals.items():
        result[f"{name}_pearson"] = safe_corr(aperture, signal)
        result[f"{name}_spearman"] = spearman_corr(aperture, signal)
        lag, lag_corr = lag_correlation(aperture, signal)
        result[f"{name}_best_lag_frames"] = lag
        result[f"{name}_best_lag_corr"] = lag_corr

    aperture_peak_index = int(np.argmax(aperture))
    speed_peak_index = int(np.argmax(velocity))
    result["max_aperture_frame"] = int(part.iloc[aperture_peak_index]["frame"])
    result["peak_speed_frame"] = int(part.iloc[speed_peak_index]["frame"])
    result["max_aperture_minus_peak_speed_s"] = float(time[aperture_peak_index] - time[speed_peak_index])
    result["object_area_at_max_aperture_over_final"] = float(area[aperture_peak_index] / area[-1])

    # Partial correlations ask whether each additional signal still relates to
    # aperture after removing its linear relationship with object area.
    area_residual = residualize(aperture, area)
    for name, signal in {"velocity": velocity, "ax": signals["ax"], "ay": signals["ay"], "az": signals["az"], "acceleration_mag": acceleration_mag}.items():
        result[f"{name}_partial_pearson_given_area"] = safe_corr(area_residual, residualize(signal, area))

    # A robust noise diagnostic: residual around a 5-frame rolling median.
    for name, signal in signals.items():
        smooth = pd.Series(signal).rolling(5, center=True, min_periods=1).median().to_numpy()
        noise = signal - smooth
        signal_std = float(np.std(smooth))
        noise_std = float(np.std(noise))
        result[f"{name}_smooth_std"] = signal_std
        result[f"{name}_noise_std"] = noise_std
        result[f"{name}_snr"] = signal_std / noise_std if noise_std > 1e-12 else float("inf")

    return result


def normalized_profiles(trials: dict[str, pd.DataFrame], points: int = 101) -> pd.DataFrame:
    grid = np.linspace(0.0, 100.0, points)
    rows = []
    for trial_name, frame in trials.items():
        part = frame.iloc[RESERVED_FRAMES:].reset_index(drop=True)
        x = np.linspace(0.0, 100.0, len(part))
        derived = part.copy()
        derived["acceleration_mag"] = np.sqrt(derived["ax"] ** 2 + derived["ay"] ** 2 + derived["az"] ** 2)
        for variable in ["object_area", "finger_width", "velocity", "acceleration_mag", "ax", "ay", "az"]:
            rows.append(
                pd.DataFrame(
                    {
                        "trial": trial_name,
                        "object_color": trial_name.split("_")[1],
                        "progress_percent": grid,
                        "variable": variable,
                        "value": np.interp(grid, x, derived[variable].to_numpy(float)),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def make_relationship_plot(trials: dict[str, pd.DataFrame]) -> None:
    variables = ["object_area", "velocity", "ax", "ay", "az", "acceleration_mag"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    color_map = {"Black": "#222222", "Blue": "#377eb8", "Red": "#e41a1c"}
    for ax, variable in zip(axes.ravel(), variables):
        all_x = []
        all_y = []
        for trial_name, frame in trials.items():
            part = frame.iloc[RESERVED_FRAMES:]
            x = part[variable].to_numpy(float) if variable in part else np.sqrt((part[["ax", "ay", "az"]] ** 2).sum(axis=1).to_numpy(float))
            y = part["finger_width"].to_numpy(float)
            all_x.extend(x)
            all_y.extend(y)
            ax.scatter(x, y, s=8, alpha=0.23, color=color_map[trial_name.split("_")[1]])
        pearson = safe_corr(np.asarray(all_x), np.asarray(all_y))
        spearman = spearman_corr(np.asarray(all_x), np.asarray(all_y))
        ax.set_title(f"{variable}\nPearson={pearson:.3f}, Spearman={spearman:.3f}")
        ax.set_xlabel(variable)
        ax.set_ylabel("finger_width (px)")
        ax.grid(alpha=0.2)
    fig.suptitle("Synthetic D3: instantaneous relationships with finger width", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_ROOT / "relationships_with_finger_width.png", dpi=180)
    plt.close(fig)


def make_normalized_overlay(profiles: pd.DataFrame) -> None:
    variables = ["object_area", "finger_width", "velocity", "acceleration_mag"]
    color_map = {"Black": "#222222", "Blue": "#377eb8", "Red": "#e41a1c"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for ax, variable in zip(axes.ravel(), variables):
        subset = profiles[profiles["variable"] == variable]
        for (trial, color), part in subset.groupby(["trial", "object_color"]):
            ax.plot(part["progress_percent"], part["value"], color=color_map[color], alpha=0.17, linewidth=0.8)
        for color in COLORS:
            part = subset[subset["object_color"] == color]
            mean = part.groupby("progress_percent", as_index=False)["value"].mean()
            ax.plot(mean["progress_percent"], mean["value"], color=color_map[color], linewidth=2.6, label=f"{color} mean")
        ax.set_title(variable)
        ax.set_ylabel("value")
        ax.grid(alpha=0.22)
    axes[1, 0].set_xlabel("Movement progress (%) after 15-frame reserve")
    axes[1, 1].set_xlabel("Movement progress (%) after 15-frame reserve")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Synthetic D3: normalized trial trajectories", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_ROOT / "normalized_trajectory_overlays.png", dpi=180)
    plt.close(fig)


def make_lag_plot(summary: pd.DataFrame) -> None:
    variables = ["object_area", "velocity", "acceleration_mag"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    color_map = {"Black": "#222222", "Blue": "#377eb8", "Red": "#e41a1c"}
    for ax, variable in zip(axes, variables):
        part = summary[["trial", "object_color", f"{variable}_best_lag_frames", f"{variable}_best_lag_corr"]].copy()
        for color in COLORS:
            p = part[part["object_color"] == color]
            ax.scatter(p[f"{variable}_best_lag_frames"], p[f"{variable}_best_lag_corr"], color=color_map[color], label=color, s=35, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{variable} vs aperture")
        ax.set_xlabel("Best lag (frames)\npositive = signal delayed")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Correlation at best lag")
    axes[0].legend()
    fig.suptitle("Synthetic D3: cross-correlation lag diagnostics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT_ROOT / "cross_correlation_lags.png", dpi=180)
    plt.close(fig)


def make_model_summary_plot() -> None:
    summary_path = TRAIN_OUTPUT / "overall_model_summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    labels = {"window_area_only": "Area only", "window_area_velocity": "Area + velocity", "window_area_acceleration": "Area + acceleration"}
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, title in zip(axes, ["rmse_mean_px", "mae_mean_px", "r2_mean"], ["RMSE", "MAE", "R²"]):
        bars = ax.bar(x, summary[metric], color=["#4c78a8", "#f58518", "#54a24b"], yerr=summary[metric.replace("mean", "sd")], capsize=4)
        ax.set_xticks(x, [labels[name] for name in summary["model"]], rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Existing D3 Window MLP results", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_ROOT / "model_comparison.png", dpi=180)
    plt.close(fig)


def make_history_information_diagnostic(trials: dict[str, pd.DataFrame], split: pd.DataFrame) -> pd.DataFrame:
    """Compare simple ridge predictions from the same 15-frame histories.

    This is diagnostic only; it does not alter or retrain the requested MLP.
    """
    train_names = split.loc[split["split"] == "train", "trial"].tolist()
    test_names = split.loc[split["split"] == "test", "trial"].tolist()
    specs = {
        "area_history": ["object_area"],
        "area_velocity_history": ["object_area", "velocity"],
        "area_acceleration_history": ["object_area", "ax", "ay", "az"],
    }
    rows = []
    alpha = 1.0
    for model, features in specs.items():
        train_x, train_y = [], []
        test_x, test_y = [], []
        for name in train_names:
            frame = trials[name]
            values = frame[features + ["finger_width"]].to_numpy(float)
            for end in range(WINDOW_SIZE - 1, len(frame)):
                window = values[end - WINDOW_SIZE + 1 : end + 1]
                if np.isfinite(window).all():
                    train_x.append(window[:, : len(features)].reshape(-1))
                    train_y.append(window[-1, -1])
        for name in test_names:
            frame = trials[name]
            values = frame[features + ["finger_width"]].to_numpy(float)
            for end in range(WINDOW_SIZE - 1, len(frame)):
                window = values[end - WINDOW_SIZE + 1 : end + 1]
                if np.isfinite(window).all():
                    test_x.append(window[:, : len(features)].reshape(-1))
                    test_y.append(window[-1, -1])
        train_x = np.asarray(train_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float)
        test_x = np.asarray(test_x, dtype=float)
        test_y = np.asarray(test_y, dtype=float)
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        x_train = (train_x - mean) / scale
        x_test = (test_x - mean) / scale
        y_mean = train_y.mean()
        y_scale = train_y.std() or 1.0
        y_train = (train_y - y_mean) / y_scale
        identity = np.eye(x_train.shape[1])
        beta = np.linalg.solve(x_train.T @ x_train + alpha * identity, x_train.T @ y_train)
        pred = x_test @ beta * y_scale + y_mean
        error = pred - test_y
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        ss_tot = float(np.sum((test_y - test_y.mean()) ** 2))
        r2 = float(1.0 - np.sum(error**2) / ss_tot) if ss_tot > 1e-12 else float("nan")
        rows.append({"diagnostic_model": model, "rmse_px": rmse, "mae_px": mae, "r2": r2, "train_windows": len(train_x), "test_windows": len(test_x)})
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, profiles: pd.DataFrame, history_info: pd.DataFrame) -> None:
    model_summary = pd.read_csv(TRAIN_OUTPUT / "overall_model_summary.csv")
    lines = []
    lines.append("Synthetic D3 Window MLP diagnostic report")
    lines.append("")
    lines.append("Primary conclusion")
    lines.append("The current generator makes object_area a near-direct phase variable for finger_width.")
    lines.append("The 15-frame area history therefore identifies both movement phase and aperture state, leaving little independent information for velocity.")
    lines.append("Acceleration is derived from a synthetic velocity trajectory and contains high-frequency noise relative to its useful signal, so it can hurt a finite-capacity MLP.")
    lines.append("")
    lines.append("Existing MLP results")
    lines.append(model_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    lines.append("")
    lines.append("Simple 15-frame history diagnostic (not an MLP replacement)")
    lines.append(history_info.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    lines.append("")
    lines.append("Per-trial timing and area ratios")
    lines.append(summary[["trial", "object_color", "max_aperture_minus_peak_speed_s", "object_area_at_max_aperture_over_final", "velocity_best_lag_frames", "acceleration_mag_best_lag_frames"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    lines.append("")
    lines.append("Recommended next-generation changes")
    lines.append("1. Keep area-to-aperture coupling, but add trial-specific aperture offsets or closure timing that are not recoverable from area alone.")
    lines.append("2. Vary speed-peak timing independently within a physically plausible range and make the same area history correspond to slightly different aperture trajectories.")
    lines.append("3. Generate acceleration from a full 3-D hand velocity trajectory, then add low-amplitude sensor noise separately; avoid making acceleration a noisy copy of a derivative that the target does not use.")
    lines.append("4. Add controlled variation in approach speed and closure speed while preserving a single-peaked reach and smooth start/end behavior.")
    lines.append("5. Keep the train/test split by complete trial and report per-color results, not only pooled results.")
    (OUTPUT_ROOT / "diagnostic_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    trials = load_trials()
    summary = pd.DataFrame([calculate_trial_diagnostics(name, frame) for name, frame in trials.items()]).sort_values("trial")
    profiles = normalized_profiles(trials)
    split = pd.read_csv(TRAIN_OUTPUT / "data_split.csv")
    history_info = make_history_information_diagnostic(trials, split)

    summary.to_csv(OUTPUT_ROOT / "per_trial_diagnostics.csv", index=False, float_format="%.6f", encoding="utf-8-sig")
    profiles.to_csv(OUTPUT_ROOT / "normalized_profiles.csv", index=False, float_format="%.6f", encoding="utf-8-sig")
    history_info.to_csv(OUTPUT_ROOT / "history_information_diagnostic.csv", index=False, float_format="%.6f", encoding="utf-8-sig")

    variable_rows = []
    for variable in ["object_area", "finger_width", "velocity", "acceleration_mag", "ax", "ay", "az"]:
        part = profiles[profiles["variable"] == variable]
        grouped = part.groupby("progress_percent")["value"]
        temporal_sd = grouped.std(ddof=1).mean()
        temporal_mean_abs = grouped.mean().abs().mean()
        variable_rows.append({"variable": variable, "mean_temporal_sd": temporal_sd, "mean_absolute_level": temporal_mean_abs, "relative_temporal_sd": temporal_sd / temporal_mean_abs if temporal_mean_abs > 1e-12 else np.nan})
    pd.DataFrame(variable_rows).to_csv(OUTPUT_ROOT / "normalized_variable_stability.csv", index=False, float_format="%.6f", encoding="utf-8-sig")

    make_relationship_plot(trials)
    make_normalized_overlay(profiles)
    make_lag_plot(summary)
    make_model_summary_plot()
    write_report(summary, profiles, history_info)

    print(summary[["trial", "object_color", "object_area_pearson", "velocity_partial_pearson_given_area", "acceleration_mag_partial_pearson_given_area", "velocity_best_lag_frames", "acceleration_mag_best_lag_frames", "max_aperture_minus_peak_speed_s", "object_area_at_max_aperture_over_final"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved diagnostics to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
