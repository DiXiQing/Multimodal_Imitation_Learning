from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\临时数据\D11\2')
INPUT_CSV = 'grasp_data.csv'
OUTPUT_CSV = 'grasp_data_smoothed.csv'
PREVIEW_NAME = 'smoothing_preview.png'
MEDIAN_WINDOW = 3
EMA_SPAN = 5
SMOOTH_FINGER_WIDTH = True
IMU_RATE_HZ = 50.0
CUTOFF_HZ = 3.0
FILTER_ORDER = 2
MAX_IMU_GAP_SECONDS = 0.25
EXCLUDED_FOLDERS = {'backup', '备用', '待检查', 'filter_comparison'}


def median_then_ema(series):
    return series.rolling(MEDIAN_WINDOW, min_periods=1).median().ewm(span=EMA_SPAN, adjust=False).mean()


def ema(series):
    return series.ewm(span=EMA_SPAN, adjust=False).mean()


def smooth_trial(df, filtered_axes=None):
    smooth = df.copy()
    smooth['object_area'] = median_then_ema(df['object_area'])
    if SMOOTH_FINGER_WIDTH:
        smooth['finger_width'] = median_then_ema(df['finger_width'])
    if 'velocity' in df:
        smooth['velocity'] = ema(df['velocity'])
    if filtered_axes is None:
        for col in ['ax', 'ay', 'az']:
            smooth[col] = ema(df[col])
    else:
        smooth[['ax', 'ay', 'az']] = filtered_axes
    smooth['mag'] = np.linalg.norm(smooth[['ax', 'ay', 'az']], axis=1)
    return smooth


def filter_original_imu(trial, raw, cutoff, rate):
    imu = pd.read_csv(trial / 'imu.csv')
    columns = ['t_imu', 'ax', 'ay', 'az']
    imu[columns] = imu[columns].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(imu[columns]).all().all():
        raise ValueError('IMU contains missing or invalid values')
    imu = imu.sort_values('t_imu').drop_duplicates('t_imu', keep='last')
    if 't_camera' in raw:
        tc = pd.to_numeric(raw.t_camera, errors='coerce').to_numpy()
    else:
        camera = pd.read_csv(trial / 'camera.csv').set_index('frame')
        tc = camera.loc[raw.frame, 't_camera'].to_numpy(dtype=float)
    if not np.isfinite(tc).all() or not (np.diff(tc) > 0).all():
        raise ValueError('Camera timestamps must be finite and increasing')
    before = imu.loc[imu.t_imu <= tc[0]].tail(1)
    after = imu.loc[imu.t_imu >= tc[-1]].head(1)
    if before.empty or after.empty:
        raise ValueError('IMU does not cover all camera timestamps')
    segment = imu.loc[imu.t_imu.between(before.t_imu.iloc[0], after.t_imu.iloc[0])]
    ti = segment.t_imu.to_numpy()
    if len(ti) < 3:
        raise ValueError('Too few IMU samples')
    gap = float(np.diff(ti).max())
    if gap > MAX_IMU_GAP_SECONDS:
        raise ValueError(f'IMU gap {gap:.3f}s exceeds {MAX_IMU_GAP_SECONDS}s')
    axes = segment[['ax', 'ay', 'az']].to_numpy()
    uniform = ti[0] + np.arange(int(np.floor((ti[-1] - ti[0]) * rate)) + 1) / rate
    u = np.column_stack([np.interp(uniform, ti, axes[:, j]) for j in range(3)])
    sos = butter(FILTER_ORDER, cutoff, fs=rate, output='sos')
    filtered = np.column_stack([sosfilt(sos, u[:, j], zi=sosfilt_zi(sos) * u[0, j])[0] for j in range(3)])
    return np.column_stack([np.interp(tc, uniform, filtered[:, j]) for j in range(3)])


def save_preview(raw, smooth, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, col, label in zip(axes[:2], ['finger_width', 'object_area'], ['FW (px)', 'OA (px²)']):
        ax.plot(raw.frame, raw[col], alpha=.45, label='Raw')
        ax.plot(smooth.frame, smooth[col], lw=2, label='Smoothed')
        ax.set_ylabel(label)
    axes[2].plot(raw.frame, np.linalg.norm(raw[['ax', 'ay', 'az']], axis=1), alpha=.45, label='Raw')
    axes[2].plot(smooth.frame, smooth.mag, lw=2, label='Butterworth low-pass')
    axes[2].set_ylabel('Acceleration magnitude (m/s²)')
    axes[2].set_xlabel('Frame')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Batch smooth grasp trials')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--cutoff-hz', type=float, default=CUTOFF_HZ)
    parser.add_argument('--imu-rate', type=float, default=IMU_RATE_HZ)
    parser.add_argument('--include-backup', action='store_true')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 0 < args.cutoff_hz < args.imu_rate / 2:
        parser.error('Cutoff must be positive and below half the IMU sample rate')
    base = args.data_root.resolve()
    if not base.is_dir():
        parser.error(f'Directory not found: {base}')
    paths = sorted(p for p in base.rglob(INPUT_CSV) if p.parent.name.startswith('TRIAL_')
                   and (args.include_backup or not any(x.casefold() in EXCLUDED_FOLDERS for x in p.relative_to(base).parts[:-1])))
    if not paths:
        parser.error(f'No {INPUT_CSV} found under {base}')
    ok = skipped = 0
    for path in paths:
        try:
            raw = pd.read_csv(path)
            required = ['frame', 'finger_width', 'object_area', 'ax', 'ay', 'az']
            raw[required] = raw[required].apply(pd.to_numeric, errors='coerce')
            if len(raw) < 2 or not np.isfinite(raw[required]).all().all() or not (np.diff(raw.frame) > 0).all():
                raise ValueError('Invalid input values or frame order')
            filtered = filter_original_imu(path.parent, raw, args.cutoff_hz, args.imu_rate)
            smooth = smooth_trial(raw, filtered)
            if not args.dry_run:
                temporary = path.parent / 'grasp_data_smoothed.tmp.csv'
                smooth.to_csv(temporary, index=False, encoding='utf-8-sig', float_format='%.9g')
                temporary.replace(path.parent / OUTPUT_CSV)
                if not args.no_preview:
                    save_preview(raw, smooth, path.parent / PREVIEW_NAME)
            ok += 1
            print(f'[OK] {path.parent.name}: {len(raw)} frames')
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
            skipped += 1
            print(f'[Skip] {path.parent.name}: {exc}; existing output retained')
    print(f'Complete: {ok}; skipped: {skipped}; cutoff: {args.cutoff_hz:g} Hz')
    if args.dry_run:
        print('Dry run: no files changed')


if __name__ == '__main__':
    main()
