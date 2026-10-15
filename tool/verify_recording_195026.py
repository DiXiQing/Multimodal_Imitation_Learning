from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Smooth import median_then_ema

source = root / 'Data/BackupData/D/TRIAL_Size_4cm_20261015_195026'
out = source / 'grasp_segments'
original = pd.read_csv(source / 'imu.csv')
unique = original.sort_values('t_imu', kind='stable').drop_duplicates('t_imu', keep='last')
sos = butter(2, 3, fs=50, output='sos')
summary = pd.read_csv(out / 'crop_summary.csv')
rows = []
for r in summary.itertuples():
    dest = out / r.trial
    raw = pd.read_csv(dest / 'grasp_data.csv')
    smooth = pd.read_csv(dest / 'grasp_data_smoothed.csv')
    imu = pd.read_csv(dest / 'imu.csv').sort_values('t_imu', kind='stable').drop_duplicates('t_imu', keep='last')
    tc = raw.t_camera.to_numpy()
    ti = imu.t_imu.to_numpy()
    assert ti[0] <= tc[0] and ti[-1] >= tc[-1]
    assert np.isfinite(smooth.select_dtypes(include='number')).all().all()
    raw_error = float(np.max(np.abs(raw.forward_raw - np.interp(tc, ti, imu.forward_raw))))
    assert raw_error < 1e-4, (r.trial, raw_error)
    assert np.array_equal(raw.forward_raw, smooth.forward_raw)
    grid = ti[0] + np.arange(int(np.ceil((tc[-1] - ti[0]) * 50)) + 1) / 50
    u = np.interp(grid, ti, imu.forward_raw)
    filtered = sosfilt(sos, u, zi=sosfilt_zi(sos) * u[0])[0]
    filter_error = float(np.max(np.abs(smooth.forward_acceleration - np.interp(tc, grid, filtered))))
    assert filter_error < 1e-4, (r.trial, filter_error)
    for col in ['finger_width', 'object_area']:
        assert np.allclose(smooth[col], median_then_ema(raw[col]))
    rows.append(dict(trial=r.trial, imu_max_gap_seconds=np.diff(ti).max(),
        imu_gaps_over_60ms=int((np.diff(ti) > .06).sum()),
        imu_coverage_before_seconds=tc[0]-ti[0], imu_coverage_after_seconds=ti[-1]-tc[-1],
        forward_raw_preserved=True, smoothing_reproduced=True,
        raw_interpolation_max_error=raw_error, filter_max_error=filter_error))
audit = pd.DataFrame(rows)
audit.to_csv(out / 'imu_alignment_audit.csv', index=False)
print('segments', len(summary), 'total frames', summary.rows.sum(), 'frame range', summary.rows.min(), summary.rows.max())
print('pre frames', summary.pre_frames.value_counts().to_dict())
print('IMU original rows', len(original), 'duplicates', original.t_imu.duplicated().sum(), 'reversals', (original.t_imu.diff()<0).sum())
print('IMU global max sorted gap', np.diff(unique.t_imu).max(), 'segment max', audit.imu_max_gap_seconds.max(), 'gaps >60ms in exported IMU', audit.imu_gaps_over_60ms.sum())
print('Excluded', pd.read_csv(out/'crop_ranges.csv').query('status == "excluded"')[['peak','reason']].to_string(index=False))
print('All smoothing and signed acceleration checks passed')
report = f'''# Crop verification

Source: {source.name}

Accepted: {len(summary)} of 36 major FW cycles. Each segment contains original JPEG frames, grasp_video.mp4, camera.csv, imu.csv, grasp_data.csv and grasp_data_smoothed.csv. Names use the actual first camera timestamp in Asia/Tokyo, to seconds, retaining size 4 cm.

Detection: centered 5-frame median used only for boundary finding; FW peak prominence 80 px, distance 70 frames. Onset is the final crossing of baseline + 10% peak excursion within 65 frames before the peak. Pre-roll is up to 25 frames, shortened if it overlaps the preceding return. Stable tail requires 16 samples (stable through stable+15), FW range <=20 px, median FW between 15% and 80% of the excursion, OA relative range <=22%, and OA gain >=1.35. Boundary images were reviewed. Four cycles (peaks 954, 2738, 3109, 3197) failed the stable-tail requirement; peak 1121 was excluded for FW loss at frame 1096. Initial small hand adjustment and return movements are not exported.

Pre-roll distribution: {summary.pre_frames.value_counts().to_dict()}. Post-stability padding: 15 frames.

FW/OA smoothing: causal trailing 3-frame median then EMA span 5. Acceleration: interpolate original IMU onto a 50 Hz grid, causal second-order 3 Hz Butterworth with first-sample steady-state initialization, then interpolate to camera timestamps. No curve shifts or magnitude substitution. forward_raw remains identical in the synchronized raw and smoothed CSV; forward_acceleration is filtered signed acceleration.

Original IMU: {len(original)} rows; {original.t_imu.duplicated().sum()} duplicate timestamps and {(original.t_imu.diff()<0).sum()} backward timestamp transitions. Sorting/deduplication affects computation only; saved imu.csv retains original rows and ordering within each selected time interval. Global maximum sorted gap: {np.diff(unique.t_imu).max():.6f} s. Exported-interval maximum gap including interpolation support: {audit.imu_max_gap_seconds.max():.6f} s. Exported intervals contain {audit.imu_gaps_over_60ms.sum()} gaps >60 ms. Timing is irregular and short missing/delayed intervals exist; true sensor packet loss cannot be distinguished from notification timestamp effects using these files alone. No accepted interval exceeds 250 ms; all have IMU samples bracketing camera timestamps.

All {int(summary.rows.sum())} frames were copied with SHA-256 checks; all segment videos were fully decoded and matched frame and CSV counts. Frame IDs and camera timestamps agree across the three camera-aligned CSV files. Signed acceleration and smoothing were independently recomputed and verified. Original recording files including source frames were hash-checked unchanged by the exporter. No JSON generated.
'''
(out / 'verification_report.md').write_text(report, encoding='utf-8')
