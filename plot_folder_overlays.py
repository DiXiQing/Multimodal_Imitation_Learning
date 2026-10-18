from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\Exprement Data\Khairy\selected_15_3cm')
TEST_PREDICTIONS_DIR = None
INPUT_CSV = 'grasp_data_smoothed.csv'
EXCLUDED_FOLDERS = {'backup', '备用', '待检查', 'filter_comparison', 'overlay_plots'}
SIGNALS = [('finger_width', 'Finger width (px)'),
           ('object_area', 'Object area (px²)'),
           ('forward_acceleration', 'Signed forward acceleration (m/s²)')]


def trial_key(name):
    return re.sub(r'(Size_\d+)p(\d+cm_)', r'\1.\2', name).casefold()


def read_trials(base, filename, recursive, include_backup, excluded_trials=()):
    candidates = base.rglob(filename) if recursive else base.glob(f'TRIAL_*/{filename}')
    trials, report = [], []
    for path in sorted(candidates):
        if not path.parent.name.startswith('TRIAL_'):
            continue
        if not include_backup and any(p.casefold() in EXCLUDED_FOLDERS for p in path.relative_to(base).parts[:-1]):
            continue
        if trial_key(path.parent.name) in excluded_trials:
            print(f'[Exclude test] {path.parent.name}')
            report.append(dict(trial=path.parent.name, status='excluded_test', reason='Found in test prediction images'))
            continue
        try:
            d = pd.read_csv(path)
            if not {'forward_acceleration', 'forward_raw'}.intersection(d.columns):
                raise ValueError('未记录有符号加速度：需要 forward_acceleration 或 forward_raw 列')
            acceleration_column = 'forward_acceleration' if 'forward_acceleration' in d.columns else 'forward_raw'
            required = ['frame', 'finger_width', 'object_area', acceleration_column]
            missing = sorted(set(required) - set(d.columns))
            if missing:
                raise ValueError(f'missing columns: {missing}')
            d[required] = d[required].apply(pd.to_numeric, errors='coerce')
            if len(d) < 2 or not np.isfinite(d.frame).all() or not (np.diff(d.frame) > 0).all():
                raise ValueError('requires at least two strictly increasing, finite frame numbers')
            if not np.isfinite(d[required]).all().all():
                raise ValueError('missing/invalid signal values; skipped without interpolation')
            if (d.finger_width <= 0).any() or (d.object_area <= 0).any():
                raise ValueError('FW/OA contains zero or negative detections')
            d['forward_acceleration'] = d[acceleration_column]
            x = 100 * (d.frame.to_numpy() - d.frame.iloc[0]) / (d.frame.iloc[-1] - d.frame.iloc[0])
            match = re.search(r'Size_([0-9]+(?:[.p][0-9]+)?)cm_', path.parent.name)
            size = match.group(1).replace('p', '.') if match else 'unknown'
            label = path.parent.name.split('cm_', 1)[-1]
            trials.append(dict(path=path, data=d, x=x, size=size, label=label))
            report.append(dict(trial=path.parent.name, size_cm=size, rows=len(d), status='included', reason='', csv_path=str(path)))
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            print(f'[Skip] {path.parent.name}: {exc}')
            report.append(dict(trial=path.parent.name, status='skipped', reason=str(exc), csv_path=str(path)))
    return trials, report


def mean_curve(trials):
    columns = ['progress_percent'] + [col for col, _ in SIGNALS]
    if not trials:
        return pd.DataFrame(columns=columns)
    progress = np.linspace(0, 100, 101)
    result = {'progress_percent': progress}
    for col, _ in SIGNALS:
        result[col] = np.mean([np.interp(progress, t['x'], t['data'][col]) for t in trials], axis=0)
    return pd.DataFrame(result)


def plot_group(trials, output, title, dpi):
    output.mkdir(parents=True, exist_ok=True)
    average = mean_curve(trials)
    average.to_csv(output/'mean_curve.csv', index=False, encoding='utf-8-sig')
    colors = plt.get_cmap('turbo')(np.linspace(.05, .95, len(trials)))
    def lines(ax, col, ylabel):
        for t, color in zip(trials, colors):
            ax.plot(t['x'], t['data'][col], color=color, lw=1.2)
        ax.set(ylabel=ylabel, xlim=(0, 100))
        if col == 'forward_acceleration':
            ax.axhline(0, color='black', lw=.8, alpha=.4)
        ax.grid(alpha=.2)
    fig, axs = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    for ax, (col, ylabel) in zip(axs, SIGNALS):
        lines(ax, col, ylabel)
    axs[-1].set_xlabel('Recorded trial progress (%)')
    fig.tight_layout()
    fig.savefig(output/'overlay.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--test-predictions-dir', type=Path, default=TEST_PREDICTIONS_DIR,
                        help='Optional prediction-image directory: exclude matching test trials')
    parser.add_argument('--csv-name', default=INPUT_CSV, choices=['grasp_data_smoothed.csv', 'grasp_data.csv'])
    parser.add_argument('--direct-only', action='store_true', help='Read only immediate TRIAL_* folders')
    parser.add_argument('--include-backup', action='store_true', help='Include Backup, 备用 and 待检查 folders')
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--static', action='store_true', help='Export PNG without opening the classification window')
    args = parser.parse_args()
    base = args.data_root.resolve()
    if not base.is_dir():
        parser.error(f'Data directory not found: {base}')
    output = args.output_dir or base
    excluded_trials = set()
    if args.test_predictions_dir is not None:
        if not args.test_predictions_dir.is_dir():
            parser.error(f'Test predictions directory not found: {args.test_predictions_dir}')
        excluded_trials = {trial_key(p.stem) for p in args.test_predictions_dir.iterdir()
                           if p.is_file() and p.name.startswith('TRIAL_')
                           and p.suffix.casefold() in {'.png', '.jpg', '.jpeg', '.webp'}}
        if not excluded_trials:
            parser.error('No TRIAL_* prediction images found in the specified test directory')
    if not args.static:
        from overlay_selector import open_selector
        try:
            open_selector(base, args.csv_name, excluded_trials, output, args.dpi)
        except ValueError as exc:
            parser.error(str(exc))
        return
    trials, report = read_trials(base, args.csv_name, not args.direct_only, args.include_backup, excluded_trials)
    if not trials:
        parser.error(f'没有可用的有符号加速度实验，请选择包含方向数据的新录制目录：{base}')
    output.mkdir(parents=True, exist_ok=True)
    kind = 'Smooth' if args.csv_name == INPUT_CSV else 'Raw processed'
    if args.test_predictions_dir is not None:
        kind += ' / training trials (test images excluded)'
    plot_group(trials, output, kind, args.dpi)
    print(f'Included: {len(trials)} trials; skipped: {sum(r["status"] == "skipped" for r in report)}')
    print(f'Excluded test trials: {sum(r["status"] == "excluded_test" for r in report)}')
    print(f'Source: {args.csv_name}; no additional smoothing')
    print(f'Output: {(output / "overlay.png").resolve()}')


if __name__ == '__main__':
    main()
