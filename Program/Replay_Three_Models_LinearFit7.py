"""Live comparison of the three joint models with the causal seven-frame linear-fit velocity."""
import argparse

import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'New/Result/RealTime_Joint_Three_Models_LinearFit7'
DEFAULT_MODELS = ROOT / 'Data/1101_D/Result/Joint_Three_Models_LinearFit7_Spline1e-2_Static_300_70_30_20261114_132009'
KEYS = ['predicted_function_mm', 'predicted_mlp_mm', 'predicted_lstm_mm']
LABELS = ['Function', 'MLP 5f (MC median)', 'LSTM Full']
COLORS = ['#3485b8', '#e59a32', '#519e58']


def crop_camera_frame(frame, margins):
    """Crop without rescaling; all following coordinates use this cropped origin."""
    left, top, right, bottom = margins
    h, w = frame.shape[:2]
    if any(not isinstance(v, int) or v < 0 for v in margins):
        raise ValueError('裁剪值必须为非负整数')
    cw, ch = w-left-right, h-top-bottom
    if cw < 64 or ch < 64:
        raise ValueError('裁剪后宽、高均需至少 64 px')
    if cw % 2 or ch % 2:
        raise ValueError('为保证视频尺寸一致，裁剪后宽、高需为偶数；请将对应裁剪值调整 1 px')
    return frame[top:h-bottom, left:w-right].copy()

def color_ranges(color, black_max=60, saturation_max=120, color_s_min=140, color_v_min=50):
    if color == 'black':
        return [((0, 0, 0), (179, saturation_max, black_max))]
    if color == 'blue':
        return [((90, color_s_min, color_v_min), (135, 255, 255))]
    if color == 'red':
        return [((0, color_s_min, color_v_min), (8, 255, 255)),
                ((172, color_s_min, color_v_min), (179, 255, 255))]
    raise ValueError(f'Unsupported object color: {color}')

def detect_cube(frame, black_max=60, saturation_max=120, min_area=500,
                color='black', color_s_min=140, color_v_min=50, max_area=None):
    # Same black HSV range, 5x5 opening/closing, fill filter and largest-box
    # selection as the config.detect_cube used by Record_mediapipe_continuous.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2],np.uint8)
    for lower,upper in color_ranges(color,black_max,saturation_max,color_s_min,color_v_min):
        mask = cv2.bitwise_or(mask,cv2.inRange(hsv,lower,upper))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # Match the displayed/saved object_area (bounding-box pixels), not contour area.
        if max_area is not None and w*h > max_area:
            continue
        if cv2.contourArea(contour) / max(w*h, 1) < .30:
            continue
        candidates.append((float(w*h), (x, y, w, h)))
    return (max(candidates, key=lambda item: item[0]) if candidates else (0., None)), mask

def detect_fingers(frame, hands):
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None, None, None
    landmarks = result.multi_hand_landmarks[0]
    h, w = frame.shape[:2]
    points = [np.array([landmarks.landmark[i].x*w, landmarks.landmark[i].y*h]) for i in (4, 8)]
    if any(not (0 <= p[0] < w and 0 <= p[1] < h) for p in points):
        return None, None, landmarks
    return *points, landmarks

class JointMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7,32),nn.ReLU(),nn.Dropout(.1),
                                 nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self, x):
        return self.net(x).reshape(-1)

class JointLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, features):
        hidden, _ = self.lstm(features)
        return self.head(hidden).squeeze(-1)


class ComparisonEngine:
    gap = .10

    def __init__(self, args):
        self.size = args.size_cm
        self.mlp_samples = args.mlp_samples
        self.lstm_samples = args.lstm_samples
        fn = json.loads((args.models/'function.json').read_text(encoding='utf-8-sig'))
        self.function_spec = fn
        if fn['velocity_method'] != 'LinearFit7':
            raise ValueError('Expected the LinearFit7 model bundle.')
        self.velocity_scaling = fn['velocity_scaling']
        self.set_size(args.size_cm)
        self.low, self.span = fn['train_x_min'], fn['train_x_range']
        self.mlp, self.lstm = JointMLP(), JointLSTM()
        self.scales = []
        for model, name in [(self.mlp,'mlp5.pth'),(self.lstm,'lstm_full.pth')]:
            path = args.lstm_checkpoint if model is self.lstm and args.lstm_checkpoint else args.models/name
            ck = torch.load(path,map_location='cpu',weights_only=True)
            if (ck['velocity_method'] != fn['velocity_method']
                    or (path == args.models/name and ck['velocity_scaling'] != self.velocity_scaling)):
                raise ValueError(f'Velocity settings differ in {name}')
            if model is self.lstm:
                self.lstm_velocity_scaling = ck['velocity_scaling']
            model.load_state_dict(ck['state_dict'])
            model.eval()
            self.scales.append((float(ck['mean_mm']),float(ck['std_mm'])))
        if self.mlp_samples > 1:
            self.mlp.net[2].train()
        if self.lstm_samples > 1:
            self.lstm.head[0].train()
        self.reset()

    def set_size(self, size):
        if size not in (2, 3, 4):
            raise ValueError('Expected cube size 2, 3 or 4 cm')
        self.size = size
        spec = self.function_spec
        base, slope, velocity = np.asarray(spec['coefficients']).reshape(3, -1)
        self.function = BSpline(spec['knots'], base + (size - 3) * slope, spec['degree'])
        self.velocity_function = BSpline(spec['knots'], velocity, spec['degree'])
        self.reset()

    def reset(self,clear_initial=True):
        self.history = deque(maxlen=7)
        self.velocity_mm_s = np.nan
        self.state = self.previous = None
        if clear_initial:
            self.initial_distance = None
        self.normalized_distance = np.nan
        self.last_timing = dict.fromkeys(('function_ms','mlp_5f_ms','lstm_full_ms'), np.nan)

    @torch.inference_mode()
    def step(self,distance,timestamp,frame):
        if distance is None or not np.isfinite(distance) or distance <= 0 or not np.isfinite(timestamp):
            self.reset(False)
            return dict.fromkeys(KEYS,np.nan),False,True
        gap = self.previous is not None and (frame-self.previous[1]!=1 or
                                             not 0 < timestamp-self.previous[0] <= self.gap)
        if gap:
            self.reset(False)
        if self.initial_distance is None:
            self.initial_distance = distance
        self.previous = timestamp,frame
        self.history.append((distance,timestamp))
        self.velocity_mm_s = np.nan
        if len(self.history) < 7:
            return dict.fromkeys(KEYS, np.nan), False, gap

        distances, times = np.asarray(self.history).T
        self.velocity_mm_s = float(np.polyfit(times - times[-1], distances - distances[-1], 1)[0])
        scale = self.velocity_scaling
        velocity = (self.velocity_mm_s-scale['mean_mm_s'])/scale['std_mm_s']
        history = distances[-5:]/self.initial_distance
        x = self.normalized_distance = history[-1]

        start = time.perf_counter()
        spline_x = (x-self.low)/self.span
        function = float(self.function(spline_x) + velocity*self.velocity_function(spline_x))
        self.last_timing['function_ms'] = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        features = torch.tensor([*history,self.size,velocity],dtype=torch.float32)
        result = self.mlp(features.expand(self.mlp_samples,-1))
        mean,std = self.scales[0]
        mlp = torch.quantile(result,.5).item()*std+mean
        self.last_timing['mlp_5f_ms'] = (time.perf_counter()-start)*1000

        start = time.perf_counter()
        scale = self.lstm_velocity_scaling
        lstm_velocity = (self.velocity_mm_s-scale['mean_mm_s'])/scale['std_mm_s']
        features = torch.tensor([[[x,self.size,lstm_velocity]]],dtype=torch.float32)
        hidden,self.state = self.lstm.lstm(features,self.state)
        result = self.lstm.head(hidden[0].expand(self.lstm_samples,-1)).flatten()
        mean,std = self.scales[1]
        lstm = torch.quantile(result,.5).item()*std+mean
        self.last_timing['lstm_full_ms'] = (time.perf_counter()-start)*1000
        return dict(zip(KEYS,(function,mlp,lstm))),True,gap




def export_plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    labels = ['Function', 'MLP 5f', 'LSTM Full']
    colors = ['#3182ad', '#e69b32', '#4d9850']
    old = ['predicted_function_mm', 'predicted_mlp_mm', 'predicted_lstm_mm']
    new = ['replay_function_mm', 'replay_mlp_mm', 'replay_lstm_mm']
    target = 'finger_width_mm'
    folder = output / 'prediction_curve_comparisons'
    folder.mkdir(exist_ok=True)
    rows = []
    for path in sorted(output.glob('trial_*/predictions.csv')):
        data = pd.read_csv(path)
        valid = np.isfinite(data[[target, *new]]).all(axis=1)
        shared = valid & np.isfinite(data[old]).all(axis=1)
        y = data.loc[valid, target].to_numpy()
        denominator = np.sum((y - y.mean()) ** 2) if len(y) else 0
        for label, column in zip(labels, new):
            if len(y):
                error = data.loc[valid, column].to_numpy() - y
                rows.append(dict(trial=path.parent.name, model=label, frames=len(y),
                                 mae_mm=np.mean(abs(error)), rmse_mm=np.sqrt(np.mean(error**2)),
                                 r2=1-np.sum(error**2)/denominator if denominator > 0 else np.nan))
        t = data.t_camera - data.t_camera.iloc[0]
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 5), layout='constrained')
        fig.suptitle(path.parent.name)
        for ax, columns, title in zip(axes, [old, new], ['Mean4 | Recorded live', 'LinearFit7 | Offline replay']):
            ax.plot(t, data[target], color='black', label='Measured')
            for label, color, column in zip(labels, colors, columns):
                ax.plot(t, data[column].where(shared), color=color, label=label)
            ax.set(title=title, ylabel='d (mm)')
            ax.grid(alpha=.2)
        axes[0].legend(ncol=4, fontsize=8)
        axes[-1].set_xlabel('Time (s)')
        fig.savefig(folder / f'{path.parent.name}.png', dpi=160)
        plt.close(fig)

    if not rows:
        raise ValueError('No jointly valid replay predictions for the metrics chart')
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / 'test_per_trial_metrics.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for ax, key, title in zip(axes, ['mae_mm', 'rmse_mm', 'r2'], ['MAE (mm)', 'RMSE (mm)', 'R²']):
        group = metrics.groupby('model')[key].agg(['mean', 'std']).reindex(labels)
        means, sd = group['mean'], group['std'].fillna(0)
        ax.bar(labels, means, yerr=sd, capsize=4, color=colors)
        digits = 3 if key == 'r2' else 2
        for i, (mean, spread) in enumerate(zip(means, sd)):
            if np.isfinite(mean):
                ax.annotate(f'{mean:.{digits}f} ± {spread:.{digits}f}', (i, mean + spread),
                            xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
        ax.set_title(title)
        ax.margins(y=.19)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
    fig.savefig(output / 'test_model_comparison.png', dpi=220)
    plt.close(fig)


def main():
    from types import SimpleNamespace
    import pandas as pd

    source = ROOT / 'Data/1101_D/Realtime/LSTM_Best_3/4'
    files = sorted(source.glob('trial_*/predictions.csv'))
    if not files:
        raise FileNotFoundError(source)
    torch.set_num_threads(1)
    torch.manual_seed(20261029)
    engine = ComparisonEngine(SimpleNamespace(size_cm=4, mlp_samples=128,
        lstm_samples=1, models=DEFAULT_MODELS, lstm_checkpoint=None))
    output = OUTPUT / f'Timed_Replay_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True)
    timing_columns = ['replay_inference_ms', 'function_ms', 'mlp_5f_ms', 'lstm_full_ms']
    all_timings = []
    summary = []
    original_settings = {
        path.parent.name: path
        for path in (ROOT / 'New/Result/RealTime_Joint_Three_Models_BackwardDiff_Mean4')
        .glob('three_model_live_*/trial_*/settings.json')
    }
    for path in files:
        settings_path = path.with_name('settings.json')
        if not settings_path.is_file():
            settings_path = original_settings.get(path.parent.name)
        if settings_path is None:
            raise FileNotFoundError(f'Missing original cube size settings for {path.parent.name}')
        settings = json.loads(settings_path.read_text(encoding='utf-8-sig'))
        engine.set_size(settings['size_cm'])
        data = pd.read_csv(path)
        records = []
        for row in data.itertuples():
            distance = row.index_mcp_object_distance_mm if row.valid else np.nan
            start = time.perf_counter_ns()
            values, valid, _ = engine.step(distance, row.t_camera, row.source_frame)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            records.append(dict(
                zip(('replay_function_mm', 'replay_mlp_mm', 'replay_lstm_mm'), values.values()),
                replay_valid=valid, replay_velocity_mm_s=engine.velocity_mm_s,
                replay_inference_ms=elapsed if valid else np.nan,
                **(engine.last_timing if valid else dict.fromkeys(timing_columns[1:], np.nan))))
        replay = pd.DataFrame(records)
        timings = replay.loc[replay.replay_valid, timing_columns]
        all_timings.append(timings)
        replay = replay.rename(columns={key: 'replay_' + key for key in timing_columns[1:]})
        folder = output / path.parent.name
        folder.mkdir()
        pd.concat([data, replay], axis=1).to_csv(folder/'predictions.csv', index=False)
        print(f'{path.parent.name}: {len(timings)} prediction frames', flush=True)

    groups = [(path.parent.name, timing) for path, timing in zip(files, all_timings)]
    groups.append(('all', pd.concat(all_timings)))
    for trial, timing in groups:
        for key in timing_columns:
            values = timing[key].dropna()
            summary.append(dict(trial=trial, stage=key, frames=len(values),
                mean_ms=values.mean(), median_ms=values.median(),
                p95_ms=values.quantile(.95), max_ms=values.max()))
    pd.DataFrame(summary).to_csv(output/'timing_summary.csv', index=False)
    (output/'sources.json').write_text(json.dumps(dict(source=str(source),
        models=str(DEFAULT_MODELS), mlp_samples=128, lstm_samples=1,
        timing='CSV input ready to all three predictions complete'), indent=2), encoding='utf-8')
    export_plots(output)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
