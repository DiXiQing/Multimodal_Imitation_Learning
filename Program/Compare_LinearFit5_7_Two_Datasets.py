"""Compare causal five/seven-frame velocity using matched 70/30 trial splits."""
import argparse
import importlib.util
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("1101_D", "1115_Song")


def load_training_module():
    path = Path(__file__).with_name("Train_Song_Joint_Three_Models_LinearFit7_70_30.py")
    spec = importlib.util.spec_from_file_location("velocity_comparison_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed = random.SystemRandom().randrange(2**31)
    output = PROJECT / "Data/Result" / ("Compare_LinearFit5_7_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    combined = []
    for dataset in DATASETS:
        root = PROJECT / "Data" / dataset
        folders = {}
        rng = random.Random(seed)
        for size in (2, 3, 4):
            home = root / (f"{size*10}mm" if dataset == "1101_D" else str(size*10))
            trials = sorted(p for p in home.iterdir() if p.is_dir() and len(p.name) == 15
                            and p.name.replace("_", "").isdigit() and (p / "camera.csv").is_file())
            if len(trials) < 2:
                raise ValueError(f"Insufficient trials in {home}")
            rng.shuffle(trials)
            cutoff = round(len(trials) * .7)
            folders[size] = (trials[:cutoff], trials[cutoff:])
        supports = None
        for velocity_window in (5, 7):
            module = load_training_module()
            module.SEED = seed
            module.ROOT = root
            module.VELOCITY_METHOD = f"LinearFit{velocity_window}"
            # Both variants use the same seventh-frame targets and LSTM prefix setup.
            # Only the slope estimator window changes, isolating the velocity feature.
            original_estimator = module.estimate_velocity
            def estimate(distance, times, runs):
                previous = module.VELOCITY_WINDOW
                module.VELOCITY_WINDOW = velocity_window
                try:
                    result = original_estimator(distance, times, runs)
                    for run in runs:
                        result[run[:6]] = np.nan
                    return result
                finally:
                    module.VELOCITY_WINDOW = previous
            module.estimate_velocity = estimate
            train, test = [], []
            for size, (training, testing) in folders.items():
                train.extend(dict(module.load_trial(p), size_cm=size) for p in training)
                test.extend(dict(module.load_trial(p), size_cm=size) for p in testing)
            current = [(g["name"], g["common"].tolist()) for g in train + test]
            if supports is None:
                supports = current
            assert current == supports
            module.configure_velocity(train, test, args)
            args.lstm_train, static = module.extend_lstm_training(train, args.velocity_scaling)
            print(f"{dataset} | LinearFit{velocity_window} | {len(train)} train / {len(test)} test", flush=True)
            module.check_data(test)
            if args.check_data:
                continue
            dest = output / dataset / f"LinearFit{velocity_window}"
            dest.mkdir(parents=True)
            module.write_json(dest / "split_manifest.json", dict(
                source=str(root), seed=seed, velocity_method=module.VELOCITY_METHOD,
                velocity_window_frames=velocity_window, target_start_frame=7,
                evaluation="Matched trial split and valid target frames; trial-level 70/30",
                epochs=args.epochs,
                Train=[dict(trial=g["name"], size_cm=g["size_cm"], csv_sha256=g["csv_sha256"]) for g in train],
                Test=[dict(trial=g["name"], size_cm=g["size_cm"], csv_sha256=g["csv_sha256"]) for g in test]))
            module.write_csv(dest / "lstm_static_prefixes.csv", static)
            module.write_json(dest / "velocity_scaling.json", args.velocity_scaling)
            metrics = module.run_holdout(train, test, args, dest)
            summary = module.summarize(metrics)
            module.write_csv(dest / "test_mean_sd.csv", summary)
            module.plot_summary(dest, summary)
            module.plot_training(dest)
            module.plot_trials(dest)
            combined.extend(dict(dataset=dataset, velocity_window=velocity_window, **row) for row in summary)
            module.write_csv(output / "comparison_mean_sd.csv", combined)
    if args.check_data:
        print("PASS: both datasets, both estimators, matched splits and target frames.", flush=True)
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    labels = ("Function", "MLP 5f", "LSTM Full")
    for row, dataset in enumerate(DATASETS):
        for col, (metric, title) in enumerate((("mae_mm", "MAE (mm)"), ("rmse_mm", "RMSE (mm)"), ("r2", "R²"))):
            ax = axes[row, col]
            for window, offset, color in ((5, -.18, "#3285bd"), (7, .18, "#ef9b28")):
                entries = [next(v for v in combined if v["dataset"] == dataset
                                and v["velocity_window"] == window and v["size_cm"] == "all"
                                and v["model"] == label and v["metric"] == metric) for label in labels]
                ax.bar(np.arange(3)+offset, [v["mean"] for v in entries], width=.35,
                       yerr=[v["sd"] for v in entries], capsize=3, color=color, label=f"Linear Fit {window}")
            ax.set_xticks(range(3), labels)
            ax.set_title(f"{dataset} | {title}")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
