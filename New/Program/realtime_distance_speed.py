"""Measure signed image-distance velocity using past frames only. Never trains.

N starts a recording and fixes the approach axis/object centre at the current
initial pose. S saves CSV and plots. Q quits. --self-test never opens a camera.
"""
import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAMERA_INDEX = 1
SETTINGS = ROOT / 'Data/TopView/TRIAL_Color_red_Size_2cm_20261028_140605_784583/settings.json'
OUTPUT = ROOT / 'New/Result/RealTime'
WINDOW_SIZE = 5


class SpeedEstimator:
    def __init__(self, window=5, max_gap=.10):
        self.window, self.max_gap = window, max_gap
        self.samples = deque(maxlen=window)

    def reset(self):
        self.samples.clear()

    def step(self, distance, timestamp, frame):
        raw = speed = span = np.nan
        reset = False
        if not np.isfinite(distance) or not np.isfinite(timestamp):
            self.reset()
            return raw, speed, span, reset
        if self.samples:
            previous = self.samples[-1]
            dt = timestamp - previous[0]
            if frame - previous[2] != 1 or not 0 < dt <= self.max_gap:
                self.reset()
                reset = True
            else:
                raw = (distance - previous[1]) / dt
        self.samples.append((timestamp, distance, frame))
        if len(self.samples) == self.window:
            points = np.array(self.samples)
            t = points[:, 0] - points[0, 0]
            t -= t.mean()
            speed = float(t @ (points[:, 1] - points[:, 1].mean()) / (t @ t))
            span = float(points[-1, 0] - points[0, 0])
        return raw, speed, span, reset


def save_trial(session, rows, metadata):
    if not rows:
        return
    session.mkdir(parents=True, exist_ok=False)
    with (session / 'measurements.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (session / 'settings.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = [r['elapsed_s'] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, [r['signed_distance_px'] for r in rows])
    axes[0].set_ylabel('Signed distance (px)')
    axes[1].plot(t, [r['raw_velocity_px_s'] for r in rows], alpha=.35, lw=.8, label='Two-frame difference')
    axes[1].plot(t, [r['velocity_px_s'] for r in rows], lw=1.3, label=f"Past {metadata['window']} frames: linear fit")
    axes[1].set_ylabel('Velocity (px/s)')
    axes[1].axhline(0, color='gray', lw=.7)
    axes[1].legend()
    axes[2].plot(t, [r['finger_width_px'] for r in rows], color='black')
    axes[2].set_ylabel('Aperture (px)')
    axes[2].set_xlabel('Time from trial start (s)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(session / f'distance_speed.{ext}', dpi=180)
    plt.close(fig)
    print('Saved:', session, flush=True)


def live_chart(rows):
    image = np.full((420, 800, 3), 255, np.uint8)
    if len(rows) < 2:
        return image
    part = rows[-300:]  # Display only; CSV retains every recorded frame.
    t = np.array([r['elapsed_s'] for r in part])
    x = 65 + (t - t[0]) / max(t[-1] - t[0], .001) * 710
    for panel, (key, label, color) in enumerate([
        ('signed_distance_px', 'Signed distance (px)', (184, 133, 52)),
        ('velocity_px_s', 'Velocity (px/s): negative = approaching', (50, 154, 229)),
    ]):
        values = np.array([r[key] for r in part])
        finite = values[np.isfinite(values)]
        top = panel * 205 + 30
        cv2.putText(image, label, (12, top - 10), 0, .5, (30, 30, 30), 1)
        if not len(finite):
            continue
        lo, hi = min(float(finite.min()), 0.), max(float(finite.max()), 1.)
        y = top + 150 - (values - lo) / max(hi - lo, 1.) * 145
        for i in range(1, len(part)):
            if np.isfinite(values[i-1:i+1]).all():
                cv2.line(image, (int(x[i-1]), int(y[i-1])), (int(x[i]), int(y[i])), color, 1)
        cv2.putText(image, f'{lo:.0f} .. {hi:.0f}', (10, top + 173), 0, .45, (50, 50, 50), 1)
    return image


def self_test():
    estimator = SpeedEstimator()
    times = np.cumsum([.03, .04, .035, .045, .032, .038, .04])
    for i, t in enumerate(times):
        _, speed, _, _ = estimator.step(250 - 120*t, t, i)
        if i >= 4:
            assert abs(speed + 120) < 1e-8
        else:
            assert np.isnan(speed)
    estimator.reset()
    for i in range(8):
        _, speed, _, _ = estimator.step(200., i/30, i)
    assert speed == 0
    _, speed, _, gap = estimator.step(100., 2., 8)
    assert gap and np.isnan(speed)
    estimator.step(np.nan, 2.03, 9)
    assert not estimator.samples
    print('PASS: irregular timestamps, known velocity, stationary input, warmup and gap/invalid resets. No camera.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX)
    parser.add_argument('--settings', type=Path, default=SETTINGS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--window', type=int, default=WINDOW_SIZE)
    parser.add_argument('--max-gap', type=float, default=.10)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.window < 2 or not np.isfinite(args.max_gap) or args.max_gap <= 0:
        parser.error('Window must be >=2 and max-gap must be finite and positive.')
    if args.self_test:
        self_test()
        return
    from Record import detect_cube, detect_fingers, crop_camera_frame, approach_axis, signed_hand_distance
    import mediapipe as mp
    settings = json.loads(args.settings.read_text(encoding='utf-8-sig'))
    margins = tuple(int(settings.get('crop_margins_px', {}).get(k, 0)) for k in ('left', 'top', 'right', 'bottom'))
    expected = (int(settings['width']), int(settings['height']))
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Cannot open camera; change CAMERA_INDEX or --camera.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.get('camera_source_width', 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.get('camera_source_height', 480))
    cap.set(cv2.CAP_PROP_FPS, 30)
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=.5, min_tracking_confidence=.5)
    estimator = SpeedEstimator(args.window, args.max_gap)
    rows, axis, centre, session, metadata = [], None, None, None, None
    frame = 0
    previous_time = None
    print('N: start/reset axis and lock object centre; S: save; Q: quit. Keep the object stationary.')
    try:
        while True:
            ok, raw = cap.read()
            timestamp = time.perf_counter()  # Monotonic actual capture-completion time.
            if not ok:
                raise RuntimeError('Camera read failed.')
            image = crop_camera_frame(raw, margins)
            if (image.shape[1], image.shape[0]) != expected:
                raise RuntimeError(f'Unexpected cropped size: {image.shape[1]}x{image.shape[0]}; expected {expected}.')
            thumb, index, landmarks = detect_fingers(image, hands)
            (_, box), _ = detect_cube(image, settings.get('v', 60), settings.get('s', 120), settings.get('area', 52),
                color=settings.get('object_color', 'red'), color_s_min=settings.get('color_s_min', 140),
                color_v_min=settings.get('color_v_min', 50), max_area=settings.get('max_area', 640000))
            detected = np.array([box[0]+box[2]/2, box[1]+box[3]/2]) if box is not None else None
            midpoint = (thumb+index)/2 if thumb is not None and index is not None else None
            aperture = float(np.linalg.norm(thumb-index)) if midpoint is not None else np.nan
            distance = speed = raw_speed = span = np.nan
            reset = False
            if session is not None:
                distance = signed_hand_distance(centre, midpoint, axis) if midpoint is not None else np.nan
                raw_speed, speed, span, reset = estimator.step(distance, timestamp, frame)
                rows.append(dict(source_frame=frame, elapsed_s=timestamp-metadata['start_monotonic_s'],
                    capture_monotonic_s=timestamp, valid=bool(np.isfinite(distance)), signed_distance_px=distance,
                    finger_width_px=aperture, raw_velocity_px_s=raw_speed, velocity_px_s=speed,
                    approach_speed_px_s=-speed, window_span_s=span, velocity_valid=bool(np.isfinite(speed)), reset_on_gap=reset))
            display = image.copy()
            if landmarks is not None:
                mp.solutions.drawing_utils.draw_landmarks(display, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            a = centre if session is not None else detected
            if a is not None:
                cv2.circle(display, tuple(a.astype(int)), 4, (0, 0, 255), -1)
                if midpoint is not None:
                    cv2.line(display, tuple(a.astype(int)), tuple(midpoint.astype(int)), (0, 255, 255), 1)
            fps = 1/(timestamp-previous_time) if previous_time is not None else np.nan
            previous_time = timestamp
            labels = ['REC' if session is not None else 'N start | S save | Q quit',
                      f'Distance: {distance:+.1f} px', f'Velocity: {speed:+.1f} px/s',
                      f'Aperture: {aperture:.1f} px | FPS: {fps:.1f}']
            for j, label in enumerate(labels):
                cv2.putText(display, label, (10, 25+j*24), 0, .52, (0, 0, 0), 3)
                cv2.putText(display, label, (10, 25+j*24), 0, .52, (255, 255, 255), 1)
            cv2.imshow('Distance and speed | N start S save Q quit', display)
            if rows and frame % 3 == 0:
                cv2.imshow('Distance and speed curves', live_chart(rows))
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('s'), ord('S')) and session is not None:
                save_trial(session, rows, metadata)
                rows, session, axis, centre = [], None, None, None
                estimator.reset()
            if key in (ord('n'), ord('N')) and session is None:
                if detected is None or midpoint is None:
                    print('Hand and object must both be detected before N.')
                else:
                    centre = detected.copy()
                    axis = approach_axis(centre, midpoint)
                    estimator.reset()
                    session = args.output / f'distance_speed_{datetime.now():%Y%m%d_%H%M%S_%f}'
                    metadata = dict(recording_settings=settings, camera=args.camera, window=args.window,
                        max_gap_s=args.max_gap, start_monotonic_s=timestamp, start_unix_s=time.time(),
                        signed_distance_axis_xy=axis.tolist(), locked_object_center_xy=centre.tolist(),
                        method='Ordinary least-squares distance/time slope over trailing valid consecutive frames; no future input.',
                        units='px and px/s; projection velocity, not calibrated physical hand speed',
                        sign='velocity negative approaching, positive retreating; approach_speed is its negative',
                        coordinate_note='Fingertip midpoint motion includes changes caused by finger opening/closing.')
                    print('Recording:', session.name)
            frame += 1
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        if session is not None:
            save_trial(session, rows, metadata)


if __name__ == '__main__':
    main()
