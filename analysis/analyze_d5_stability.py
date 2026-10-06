"""Analyze trial-to-trial stability in D5 without modifying source data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\临时数据\D5")
OUT = ROOT / "stability_analysis"
GRID = np.linspace(0.0, 1.0, 101)
VARIABLES = ["object_area", "finger_width", "velocity", "acceleration_mag"]


def resample(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] > x[0] else np.linspace(0, 1, len(x))
    return np.interp(GRID, x_norm, y)


def safe_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return np.nan, np.nan
    pearson = float(np.corrcoef(a, b)[0, 1])
    ranks_a = pd.Series(a).rank().to_numpy()
    ranks_b = pd.Series(b).rank().to_numpy()
    spearman = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
    return pearson, spearman


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trials: dict[str, pd.DataFrame] = {}
    profiles: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | str | int]] = []

    for trial_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("TRIAL_")):
        path = trial_dir / "grasp_data.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["acceleration_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
        trials[trial_dir.name] = df
        frame = df["frame"].to_numpy(float)
        profiles[trial_dir.name] = {
            variable: resample(frame, df[variable].to_numpy(float))
            for variable in VARIABLES
        }

    medians = {
        variable: np.median([profiles[name][variable] for name in profiles], axis=0)
        for variable in VARIABLES
    }

    for name, df in trials.items():
        p = profiles[name]
        width = df["finger_width"].to_numpy(float)
        area = df["object_area"].to_numpy(float)
        speed = df["velocity"].to_numpy(float)
        width_max_index = int(np.argmax(p["finger_width"]))
        speed_peak_index = int(np.argmax(p["velocity"]))
        area_max_at_width = float(p["object_area"][width_max_index])
        final_area = float(np.median(p["object_area"][-5:]))
        area_diff = np.diff(area)
        negative_area = area_diff[area_diff < 0]
        shape_rmse = {
            f"{variable}_shape_rmse": float(np.sqrt(np.mean((p[variable] - medians[variable]) ** 2)))
            for variable in VARIABLES
        }
        shape_relative = {
            f"{variable}_relative_shape_rmse": shape_rmse[f"{variable}_shape_rmse"]
            / max(float(np.percentile(medians[variable], 95) - np.percentile(medians[variable], 5)), 1e-8)
            for variable in VARIABLES
        }
        row: dict[str, float | str | int] = {
            "trial": name,
            "n_frames": len(df),
            "start_width_median": float(np.median(width[:5])),
            "max_width": float(np.max(width)),
            "final_width_median": float(np.median(width[-5:])),
            "start_area_median": float(np.median(area[:5])),
            "final_area_median": float(np.median(area[-5:])),
            "max_width_progress_pct": width_max_index,
            "peak_speed_progress_pct": speed_peak_index,
            "max_width_minus_peak_speed_pct": width_max_index - speed_peak_index,
            "area_at_max_width_over_final": area_max_at_width / final_area if final_area else np.nan,
            "area_negative_step_fraction": float(np.mean(area_diff < 0)),
            "area_largest_negative_step": float(np.min(area_diff)) if len(area_diff) else 0.0,
            "area_largest_negative_step_pct": float(np.min(negative_area) / np.median(area) * 100) if len(negative_area) else 0.0,
            **shape_rmse,
            **shape_relative,
        }
        rows.append(row)

    metrics = pd.DataFrame(rows)
    relative_columns = [f"{variable}_relative_shape_rmse" for variable in VARIABLES]
    # Composite deviation: equal weight for each signal's normalized curve shape.
    metrics["composite_shape_deviation"] = metrics[relative_columns].mean(axis=1)
    metrics = metrics.sort_values("composite_shape_deviation", ascending=False)
    metrics.to_csv(OUT / "per_trial_stability.csv", index=False)

    normalized_rows = []
    for name in profiles:
        for index, progress in enumerate(GRID):
            row = {"trial": name, "progress_pct": progress * 100}
            row.update({variable: profiles[name][variable][index] for variable in VARIABLES})
            normalized_rows.append(row)
    pd.DataFrame(normalized_rows).to_csv(OUT / "normalized_profiles.csv", index=False)

    # Overlay plot with the three most deviant trials highlighted.
    outlier_names = set(metrics.head(3)["trial"])
    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True)
    labels = {
        "object_area": "Object area (px²)",
        "finger_width": "Finger width (px)",
        "velocity": "3D speed (m/s)",
        "acceleration_mag": "Acceleration magnitude",
    }
    for axis, variable in zip(axes, VARIABLES):
        for name in profiles:
            color = "tab:red" if name in outlier_names else "0.55"
            alpha = 0.95 if name in outlier_names else 0.45
            width = 1.8 if name in outlier_names else 1.0
            axis.plot(GRID * 100, profiles[name][variable], color=color, alpha=alpha, linewidth=width)
        axis.plot(GRID * 100, medians[variable], color="black", linewidth=2.5, label="Median profile")
        axis.set_ylabel(labels[variable])
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Normalized movement progress (%)")
    fig.suptitle("D5 trial stability: red = three largest composite deviations")
    fig.tight_layout()
    fig.savefig(OUT / "normalized_overlays.png", dpi=160)
    plt.close(fig)

    # Deviation ranking plot.
    ranked = metrics.sort_values("composite_shape_deviation", ascending=True)
    fig, axis = plt.subplots(figsize=(12, 7))
    colors = ["tab:red" if name in outlier_names else "0.55" for name in ranked["trial"]]
    axis.barh(ranked["trial"], ranked["composite_shape_deviation"], color=colors)
    axis.set_xlabel("Mean relative curve-shape deviation")
    axis.set_title("D5 trial similarity ranking")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "stability_ranking.png", dpi=160)
    plt.close(fig)

    # Correlations against the median profiles, for a compact stability summary.
    correlation_rows = []
    for name in profiles:
        row = {"trial": name}
        for variable in VARIABLES:
            pearson, spearman = safe_corr(profiles[name][variable], medians[variable])
            row[f"{variable}_pearson_to_median"] = pearson
            row[f"{variable}_spearman_to_median"] = spearman
        correlation_rows.append(row)
    pd.DataFrame(correlation_rows).to_csv(OUT / "correlation_to_median.csv", index=False)

    lines = [
        f"D5 stability analysis: {len(trials)} trials",
        "",
        "Three largest composite shape deviations:",
    ]
    for _, row in metrics.head(3).iterrows():
        lines.append(f"- {row['trial']}: {row['composite_shape_deviation']:.4f}")
    lines += ["", "Three smallest composite shape deviations:"]
    for _, row in metrics.tail(3).sort_values("composite_shape_deviation").iterrows():
        lines.append(f"- {row['trial']}: {row['composite_shape_deviation']:.4f}")
    lines += [
        "",
        "Aggregate median ± SD:",
    ]
    for column in [
        "start_width_median", "max_width", "final_width_median", "start_area_median",
        "final_area_median", "max_width_progress_pct", "peak_speed_progress_pct",
        "max_width_minus_peak_speed_pct", "area_at_max_width_over_final",
        "area_negative_step_fraction", "composite_shape_deviation",
    ]:
        lines.append(f"- {column}: {metrics[column].median():.4f} ± {metrics[column].std():.4f}")
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Output:", OUT)
    print(metrics[["trial", "composite_shape_deviation", "max_width_progress_pct", "peak_speed_progress_pct", "area_at_max_width_over_final", "area_negative_step_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
