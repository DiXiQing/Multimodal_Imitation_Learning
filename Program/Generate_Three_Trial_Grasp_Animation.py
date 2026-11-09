"""Render a straight-on grasp schematic and smoothed distance–aperture curves."""
import argparse
import csv
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Data/1101_D'
OUTPUT = ROOT / 'Report/Animations/Three_Trial_Schematic'
DEFAULT_TRIALS = ('20261101_195719', '20261103_143927', '20261103_114552')
INK = (48, 40, 32)
TEAL = (139, 130, 23)
ORANGE = (46, 124, 215)
PINK = (144, 73, 205)
FONT = cv.FONT_HERSHEY_SIMPLEX
FPS = 60
VIDEO_WIDTH = 1280
MOTION_TOP = 16
MOTION_HEIGHT = 132
MOTION_CHART_GAP = 62
CHART_HEIGHT = 224
CHART_TOP = MOTION_TOP + MOTION_HEIGHT + MOTION_CHART_GAP
CHART_BOTTOM = CHART_TOP + CHART_HEIGHT
VIDEO_HEIGHT = CHART_BOTTOM + 76


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def load_trial(size, name, data):
    folder = data / f'{size}mm' / name
    records = read_csv(folder / 'camera.csv')
    if len(records) < 2:
        raise ValueError(f'Trial is too short: {folder}')
    t = np.array([float(r['t_camera']) for r in records])
    t -= t[0]
    distance = np.array([float(r['index_mcp_object_distance_mm']) for r in records])
    measured = np.array([float(r['finger_width_mm']) for r in records])
    if (not all(np.isfinite(values).all() for values in (t, distance, measured))
            or not np.all(np.diff(t) > 0)):
        raise ValueError(f'Invalid measurements or timestamps: {folder}')
    display_times = np.r_[np.arange(0, t[-1], 1/FPS), t[-1]]
    window = min(11, len(t) if len(t) % 2 else len(t)-1)

    def smooth(values):
        if window >= 5:
            filtered = savgol_filter(values, window, 3)
            # Keep the measured starting and ending values.
            filtered += np.interp(t, t[[0, -1]], (values-filtered)[[0, -1]])
        else:
            filtered = values
        return np.maximum(0, PchipInterpolator(t, filtered)(display_times))

    return size, display_times, smooth(distance), smooth(measured)


def text(image, label, x, y, scale=.56, color=INK, weight=1):
    cv.putText(image, label, (x, y), FONT, scale, color, weight, cv.LINE_AA)


def render(trials, timeline, canvas, limits):
    image = canvas.copy()
    maximum_l, maximum_d = limits
    pose_scale = 310 / maximum_l
    for column, (size, t, distance, measured) in enumerate(trials):
        i = int(np.searchsorted(t, timeline, side='right') - 1)
        x = 22 + 420 * column
        cx, cy = x+48, MOTION_TOP + MOTION_HEIGHT//2
        center = cx, cy
        base = cx+round(distance[i]*pose_scale), cy
        tip_x = cx+round((distance[i]-distance[-1])*pose_scale)
        opening = round(measured[i]*pose_scale/2)
        thumb, index = (tip_x, cy-opening), (tip_x, cy+opening)
        half = round(size * pose_scale / 2)
        cv.rectangle(image, (cx-half, cy-half), (cx+half, cy+half), INK, -1)
        cv.line(image, center, base, ORANGE, 2, cv.LINE_AA)
        for tip in (thumb, index):
            cv.line(image, base, tip, TEAL, 5, cv.LINE_AA)
        cv.line(image, thumb, index, PINK, 2, cv.LINE_AA)
        for point, color, radius in [(center, (250, 250, 250), 3), (base, ORANGE, 7),
                                     (thumb, PINK, 4), (index, PINK, 4)]:
            cv.circle(image, point, radius, color, -1, cv.LINE_AA)

        text(image, f'l = {distance[i]:.1f} mm   d = {measured[i]:.1f} mm   |   {t[i]:.2f} s',
             x+35, MOTION_TOP + MOTION_HEIGHT + 20)
        xs = x + 48 + 336 * (1 - distance[:i+1] / maximum_l)
        ys = CHART_BOTTOM - CHART_HEIGHT * measured[:i+1] / maximum_d
        trajectory = np.rint(np.column_stack((xs, ys))).astype(np.int32)
        if len(trajectory) > 1:
            cv.polylines(image, [trajectory], False, TEAL, 2, cv.LINE_AA)
        cv.circle(image, tuple(trajectory[-1]), 4, TEAL, -1, cv.LINE_AA)
    return image



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--trials', nargs=3, metavar=('20MM', '30MM', '40MM'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    trials = [load_trial(size, name, args.data)
              for size, name in zip((20, 30, 40), args.trials or DEFAULT_TRIALS)]
    max_l = int(np.ceil(max(np.max(t[2]) for t in trials) / 50) * 50)
    max_d = int(np.ceil(max(np.max(t[3]) for t in trials) / 20) * 20)
    canvas = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), 255, np.uint8)
    for i in range(len(trials)):
        x = 22+420*i
        for d in range(0,max_d+1,20):
            y = round(CHART_BOTTOM-CHART_HEIGHT*d/max_d)
            cv.line(canvas,(x+48,y),(x+384,y),(224,224,224),1)
            text(canvas,str(d),x+10,y+4,.43)
        for l in range(0,max_l+1,50):
            px = round(x+48+336*(1-l/max_l))
            cv.line(canvas,(px,CHART_TOP),(px,CHART_BOTTOM),(224,224,224),1)
            text(canvas,str(l),px-11,CHART_BOTTOM+18,.42)
        cv.rectangle(canvas,(x+48,CHART_TOP),(x+384,CHART_BOTTOM),(124,124,124),1)
        text(canvas,'Aperture d (mm)',x+48,CHART_TOP-9,.49)
        text(canvas,'Distance l (mm)',x+155,CHART_BOTTOM+40,.49)
    args.output.mkdir(parents=True,exist_ok=True)
    # text(canvas, 'Schematic motion and smoothed curves for illustration', 25, CHART_BOTTOM+67, .48)
    destination = args.output / 'three_trial_schematic.mp4'
    fps = FPS
    dimensions = (VIDEO_WIDTH, VIDEO_HEIGHT)
    writer = cv.VideoWriter(str(destination), cv.VideoWriter_fourcc(*'avc1'),fps,dimensions)
    if not writer.isOpened():
        writer = cv.VideoWriter(str(destination), cv.VideoWriter_fourcc(*'mp4v'),fps,dimensions)
    if not writer.isOpened():
        raise RuntimeError('Cannot open MP4 writer')
    duration = max(t[1][-1] for t in trials)
    count = int(np.ceil((duration+.5)*fps))+1
    try:
        for i in range(count):
            writer.write(render(trials, i/fps, canvas, (max_l, max_d)))
    finally:
        writer.release()
    check = cv.VideoCapture(str(destination))
    assert sum(iter(lambda: check.read()[0], False)) == count
    check.release()
    assert cv.imwrite(str(args.output/'three_trial_schematic.png'),
                      render(trials, duration, canvas, (max_l, max_d)))
    print(destination)


if __name__ == '__main__':
    main()

