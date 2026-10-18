from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, default=ROOT/'Data/Exprement Data/Khairy/3/Result/temporalLoss_20261016_140009_768410')
    parser.add_argument('--comparison', choices=['loss', 'architecture'], default='loss')
    args = parser.parse_args()
    data = pd.read_csv(args.result_dir/'test_predictions_metrics.csv')
    objectives = ['MLP', 'CNN'] if args.comparison == 'architecture' else ['MSE', 'MSE_delta']
    labels = ['MLP', 'CNN'] if args.comparison == 'architecture' else ['Before: MSE', 'After: MSE + temporal loss']
    stem = 'mlp_cnn_overall_comparison' if args.comparison == 'architecture' else 'temporal_loss_overall_comparison'
    feature = 'Area + signed forward acceleration'
    trial_sets = [set(data.loc[data.model == f'{objective} | {feature}', 'trial']) for objective in objectives]
    if not trial_sets[0] or trial_sets[0] != trial_sets[1]:
        raise ValueError('Comparison requires matching, nonempty test trial sets')
    metrics = [('rmse_px', 'RMSE'), ('mae_px', 'MAE')]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = .32
    summary = []
    for j, (metric, label) in enumerate(metrics):
        groups = [data.loc[data.model == f'{objective} | {feature}', metric] for objective in objectives]
        means = np.array([g.mean() for g in groups])
        stds = np.array([g.std(ddof=1) for g in groups])
        bars = ax.bar(x+(j-.5)*width, means, width, yerr=stds, capsize=5, color=['#3976ad', '#ee9238'][j], label=label, zorder=3)
        for bar, value, std, group, objective in zip(bars, means, stds, groups, objectives):
            ax.annotate(f'{value:.2f}', (bar.get_x()+bar.get_width()/2, value), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, zorder=5)
            summary.append(dict(objective=objective, inputs=feature, metric=metric, mean=value, std=std, trials=len(group)))
    ax.set_xticks(x, labels)
    ax.set_ylabel('Prediction error (px)')
    ax.legend()
    ax.grid(axis='y', alpha=.2, zorder=0)
    ax.margins(y=.2)
    fig.tight_layout()
    destination = args.result_dir/f'{stem}.png'
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    pd.DataFrame(summary).to_csv(args.result_dir/f'{stem}.csv', index=False)
    print(destination)

if __name__ == '__main__':
    main()
