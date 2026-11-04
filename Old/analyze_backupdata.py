"""Descriptive analysis of Data/BackupData/*/grasp_data.csv.

Run with: py -3 analyze_backupdata.py
All generated files go to analysis/backupdata; source CSVs are read only.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Data" / "BackupData"
OUTPUT = ROOT / "analysis" / "backupdata"
FEATURES = ("object_area", "finger_width", "mag", "velocity")
SMOOTH_WINDOW = 5


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def spike_mask(values: pd.Series) -> np.ndarray:
    """Flag isolated reversals large relative to local residuals and typical steps."""
    x = values.to_numpy(dtype=float)
    found = np.zeros(len(x), dtype=bool)
    if len(x) < 5:
        return found
    local = values.rolling(5, center=True, min_periods=3).median().to_numpy()
    residual = np.abs(x - local)
    finite = residual[np.isfinite(residual)]
    if len(finite) == 0:
        return found
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    steps = np.abs(np.diff(x))
    typical_step = np.nanmedian(steps)
    threshold = max(6 * 1.4826 * mad, 3 * typical_step, 1e-12)
    for i in range(1, len(x) - 1):
        before, here, after = x[i - 1 : i + 2]
        if not np.all(np.isfinite((before, here, after))):
            continue
        reversal = (here - before) * (after - here) < 0
        if reversal and residual[i] > threshold and min(abs(here - before), abs(here - after)) > threshold:
            found[i] = True
    return found


def fmt(value: object, digits: int = 3) -> str:
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def main() -> None:
    paths = sorted(SOURCE.glob("TRIAL_*/grasp_data.csv"))
    if len(paths) != 7:
        raise ValueError(f"Expected 7 grasp_data.csv files; found {len(paths)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    trials = {}
    flags = []
    for path in paths:
        trial = path.parent.name
        df = pd.read_csv(path)
        required = {"frame", "t_camera", *FEATURES}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns in {path}: {required - set(df.columns)}")
        for column in required:
            df[column] = numeric(df[column])
        if df.empty or df[["frame", "t_camera"]].isna().any().any():
            raise ValueError(f"Missing frame/time values in {path}")
        if not df.frame.is_monotonic_increasing or not df.t_camera.is_monotonic_increasing:
            raise ValueError(f"Frame/time is not monotonic in {path}")
        if df.frame.duplicated().any() or df.t_camera.duplicated().any():
            raise ValueError(f"Duplicate frame/time in {path}")
        df["elapsed_s"] = df.t_camera - df.t_camera.iloc[0]
        trials[trial] = df
        n = len(df)
        k = max(1, int(np.ceil(n * 0.10)))
        row = {
            "trial": trial,
            "frames": n,
            "first_frame": int(df.frame.iloc[0]),
            "last_frame": int(df.frame.iloc[-1]),
            "duration_s": float(df.elapsed_s.iloc[-1]),
            "time_step_median_s": float(df.t_camera.diff().median()),
            "frame_gap_count": int((df.frame.diff().dropna() != 1).sum()),
            "window_frames": k,
        }
        for feature in FEATURES:
            x = df[feature]
            valid = x.dropna()
            stats = ("min", "max", "mean", "median", "std")
            for stat in stats:
                row[f"{feature}_{stat}"] = float(getattr(valid, stat)()) if len(valid) else np.nan
            row[f"{feature}_first10_mean"] = float(x.iloc[:k].mean())
            row[f"{feature}_last10_mean"] = float(x.iloc[-k:].mean())
            missing = int(x.isna().sum())
            zeros = int((x == 0).sum())
            row[f"{feature}_missing_count"] = missing
            row[f"{feature}_zero_count"] = zeros
            row[f"{feature}_missing_ratio"] = missing / n
            row[f"{feature}_zero_ratio"] = zeros / n
            row[f"{feature}_missing_or_zero_ratio"] = (missing + zeros) / n
            spikes = spike_mask(x)
            row[f"{feature}_spike_count"] = int(spikes.sum())
            for i in np.flatnonzero(spikes):
                flags.append({"trial": trial, "feature": feature, "frame": int(df.frame.iloc[i]),
                              "elapsed_s": float(df.elapsed_s.iloc[i]), "value": float(x.iloc[i])})
            if feature in ("finger_width", "mag", "velocity"):
                target = x.abs() if feature == "velocity" else x
                idx = target.idxmax() if target.notna().any() else None
                name = "abs_velocity" if feature == "velocity" else feature
                row[f"max_{name}_frame"] = int(df.frame.iloc[idx]) if idx is not None else np.nan
                row[f"max_{name}_elapsed_s"] = float(df.elapsed_s.iloc[idx]) if idx is not None else np.nan
                row[f"max_{name}_value"] = float(x.iloc[idx]) if idx is not None else np.nan
        row["velocity_abs_mean"] = float(df.velocity.abs().mean())
        row["overall_missing_or_zero_ratio"] = sum(
            row[f"{f}_missing_count"] + row[f"{f}_zero_count"] for f in FEATURES
        ) / (n * len(FEATURES))
        row["overall_spike_count"] = sum(row[f"{f}_spike_count"] for f in FEATURES)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "summary.csv", index=False, float_format="%.9g")
    pd.DataFrame(flags, columns=["trial", "feature", "frame", "elapsed_s", "value"]).to_csv(
        OUTPUT / "spike_flags.csv", index=False, float_format="%.9g"
    )

    metrics = [f"{f}_mean" for f in FEATURES] + ["velocity_abs_mean"]
    pairwise = []
    for a, b in combinations(rows, 2):
        record = {"trial_a": a["trial"], "trial_b": b["trial"]}
        for metric in metrics:
            record[f"{metric}_difference_b_minus_a"] = b[metric] - a[metric]
            record[f"{metric}_absolute_difference"] = abs(b[metric] - a[metric])
        pairwise.append(record)
    pd.DataFrame(pairwise).to_csv(OUTPUT / "pairwise_differences.csv", index=False, float_format="%.9g")

    variability = []
    outliers = []
    for metric in metrics:
        vals = summary[metric].to_numpy(dtype=float)
        mean = np.mean(vals)
        cv = np.std(vals, ddof=1) / abs(mean) if mean != 0 else np.nan
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        variability.append({"metric": metric, "across_trial_mean": mean,
                            "across_trial_sample_std": np.std(vals, ddof=1), "cv": cv,
                            "median": med, "mad": mad})
        if mad > 0:
            z = 0.6745 * (vals - med) / mad
            for i in np.flatnonzero(abs(z) > 3.5):
                outliers.append({"trial": summary.trial.iloc[i], "metric": metric,
                                 "value": vals[i], "modified_z": z[i]})
    pd.DataFrame(variability).to_csv(OUTPUT / "between_trial_variability.csv", index=False,
                                     float_format="%.9g")
    pd.DataFrame(outliers, columns=["trial", "metric", "value", "modified_z"]).to_csv(
        OUTPUT / "possible_outliers.csv", index=False, float_format="%.9g"
    )

    palette = plt.get_cmap("tab10")
    for feature in FEATURES:
        fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
        for i, (trial, df) in enumerate(trials.items()):
            color = palette(i)
            smooth = df[feature].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
            ax.plot(df.elapsed_s, df[feature], color=color, alpha=0.25, linewidth=1)
            ax.plot(df.elapsed_s, smooth, color=color, linewidth=1.8, label=trial)
        ax.set(xlabel="Elapsed time (s)", ylabel=feature,
               title=f"{feature}: raw (faint) and 5-frame moving mean (solid)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(OUTPUT / f"{feature}_comparison.png", dpi=170)
        plt.close(fig)

    headings = ["Trial", "Frames", "Duration (s)", "Area mean", "Width mean", "Mag mean",
                "Velocity mean", "Missing/zero", "Spikes"]
    lines = ["| " + " | ".join(headings) + " |", "|" + "|".join(["---"] * len(headings)) + "|"]
    for r in rows:
        cells = [r["trial"], str(r["frames"]), fmt(r["duration_s"]), fmt(r["object_area_mean"]),
                 fmt(r["finger_width_mean"]), fmt(r["mag_mean"]), fmt(r["velocity_mean"], 4),
                 fmt(100 * r["overall_missing_or_zero_ratio"], 2) + "%", str(r["overall_spike_count"])]
        lines.append("| " + " | ".join(cells) + " |")
    (OUTPUT / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = [
        "# Reach-to-grasp trial comparison", "",
        "Seven `grasp_data.csv` files were analyzed. Source files were read only. All values use the CSV columns as recorded; no trial was removed or adjusted.", "",
        "## Summary", "", *lines, "",
        "## Methods", "",
        "- Frames are CSV row counts; duration is last minus first `t_camera` in seconds. Event times are seconds since the first row. If a maximum ties, its first occurrence is reported. Feature units are unspecified in the source CSV and are left unchanged.",
        "- Each first/last 10% mean uses ceil(0.10 × row count) rows. Standard deviations within trials and CV across trials use sample standard deviation (ddof=1).",
        "- Missing means blank, nonnumeric, or nonfinite. Zero means an exact numeric zero. The overall missing/zero ratio divides their combined count across the four features by four times the row count. Zeros are retained in all statistics.",
        "- Spike flags are isolated one-frame reversals: the point differs from a centered 5-frame median, and both adjacent steps exceed max(6 × 1.4826 × MAD of absolute local residuals, 3 × median absolute adjacent step). Endpoints are not flagged. These are descriptive flags, not confirmed errors.",
        "- Smoothed curves use a centered 5-frame moving mean (`min_periods=1`) for display only. Statistics use raw values. Curves use elapsed seconds; no time normalization or alignment was imposed.",
        "- Pairwise differences are trial B minus trial A in every feature mean. Possible outliers use modified z = 0.6745 × (value − across-trial median) / MAD, with |z| > 3.5. When MAD is zero, no flag is computed for that metric. Seven trials give limited evidence for outliers.",
        "- CV is across trial means. Mean absolute velocity is also reported because signed velocity can cancel, making its CV hard to interpret.", "",
        "## Observations", "",
        f"- Trial lengths range from {int(summary.frames.min())} to {int(summary.frames.max())} frames; durations range from {summary.duration_s.min():.3f} to {summary.duration_s.max():.3f} s.",
    ]
    for feature in FEATURES:
        metric = f"{feature}_mean"
        low = summary.loc[summary[metric].idxmin()]
        high = summary.loc[summary[metric].idxmax()]
        cv = next(x["cv"] for x in variability if x["metric"] == metric)
        report.append(f"- {feature} trial means range from {low[metric]:.4g} ({low.trial}) to {high[metric]:.4g} ({high.trial}); across-trial CV = {cv:.3f}.")
        pair_metric = f"{metric}_absolute_difference"
        closest = min(pairwise, key=lambda x: x[pair_metric])
        farthest = max(pairwise, key=lambda x: x[pair_metric])
        report.append(
            f"  - Closest pair of trial means: {closest['trial_a']} and {closest['trial_b']} "
            f"(absolute difference {closest[pair_metric]:.4g}); largest pairwise difference: "
            f"{farthest['trial_a']} and {farthest['trial_b']} ({farthest[pair_metric]:.4g})."
        )
        change = summary[f"{feature}_last10_mean"] - summary[f"{feature}_first10_mean"]
        report.append(
            f"  - Last 10% mean exceeds first 10% mean in {int((change > 0).sum())} trials, "
            f"is lower in {int((change < 0).sum())}, and is equal in {int((change == 0).sum())}."
        )
    report.append(
        f"- Mean absolute velocity varies across trials with CV = "
        f"{next(x['cv'] for x in variability if x['metric'] == 'velocity_abs_mean'):.3f}. "
        "The signed velocity CV is sensitive to cancellation around zero."
    )
    report += [
        f"- The four feature columns contain {sum(r[f'{f}_missing_count'] for r in rows for f in FEATURES)} missing values and {sum(r[f'{f}_zero_count'] for r in rows for f in FEATURES)} exact zeros across {sum(r['frames'] for r in rows) * 4} cells. The spike rule flags {len(flags)} points in total.",
        "- Frame gaps: " + (", ".join(f"{r['trial']} ({r['frame_gap_count']})" for r in rows if r["frame_gap_count"]) or "none") + ".",
    ]
    if outliers:
        report.append("- Possible outlier flags: " + "; ".join(
            f"{x['trial']} {x['metric']} (modified z={x['modified_z']:.2f})" for x in outliers
        ) + ". No rows or trials were excluded.")
    else:
        report.append("- No trial mean exceeded the modified-z threshold on the tested metrics. No rows or trials were excluded.")
    report += ["", "## Files", "",
               "`summary.csv` contains all per-trial metrics and event locations. `pairwise_differences.csv` contains all 21 trial pairs. `between_trial_variability.csv` contains the CVs. `spike_flags.csv` and `possible_outliers.csv` contain inspection flags. Four comparison PNGs show raw and smoothed curves."]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote analysis for {len(rows)} trials to {OUTPUT}")


if __name__ == "__main__":
    main()
