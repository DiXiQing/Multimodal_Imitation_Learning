"""Draw the grasp animation from mathematical parameters, then play the MP4.

Run with Python. Dependencies: numpy, Pillow and opencv-python.
No input video or experiment data is required. This is a schematic simulation.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1280, 640
SCALE = 2
NAVY = '#263E5D'
TEAL = '#208B91'
TRACE = '#B9D6D8'
ORANGE = '#DC6D2B'
RED = '#B64059'
GRID = '#DDDFDF'
JOINT_X = 293.0
JOINT_START_Y = 100.0
JOINT_TRAVEL = 290.0
FINGER_LENGTH_Y = 77.0
OBJECT_Y = 467.0
INITIAL_DISTANCE = OBJECT_Y - JOINT_START_Y
PEAK_PROGRESS = 0.5 * INITIAL_DISTANCE / JOINT_TRAVEL


def ease(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def motion(progress):
    """Return normalized root distance, aperture and fingertip coordinates."""
    progress = np.asarray(progress)
    closing_phase = np.clip((progress - PEAK_PROGRESS) / (1.0 - PEAK_PROGRESS), 0.0, 1.0)
    closing = ease(closing_phase)
    # Join the final part continuously, with aperture and translation both
    # still changing until the shared motion clock reaches the endpoint.
    join = 0.85
    tail = np.clip((closing_phase - join) / (1.0 - join), 0.0, 1.0)
    start_value = float(ease(join))
    start_slope = 6.0 * join * (1.0 - join)
    end_slope = 0.3
    tail_value = ((2*tail**3 - 3*tail**2 + 1) * start_value
                  + (tail**3 - 2*tail**2 + tail) * (1-join) * start_slope
                  + (-2*tail**3 + 3*tail**2)
                  + (tail**3 - tail**2) * (1-join) * end_slope)
    closing = np.where(closing_phase > join, tail_value, closing)
    distance = 1.0 - JOINT_TRAVEL * progress / INITIAL_DISTANCE
    opening = np.where(
        progress <= PEAK_PROGRESS,
        5.0 + 67.0 * ease(progress / PEAK_PROGRESS),
        72.0 - 42.0 * closing,
    )
    joint_y = JOINT_START_Y + JOINT_TRAVEL * progress
    tip_y = joint_y + FINGER_LENGTH_Y
    return distance, opening, joint_y, JOINT_X - opening / 2.0, JOINT_X + opening / 2.0, tip_y


def chart_point(distance, aperture):
    return 606.0 + (1.0 - distance) * 740.0, 524.0 - aperture * 5.44


class Canvas:
    def __init__(self, image, font_path):
        self.image = image
        self.draw = ImageDraw.Draw(image)
        self.font_path = font_path
        self.fonts = {}

    def text(self, position, value, size=22, color=NAVY, anchor=None):
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(str(self.font_path), size * SCALE)
        self.draw.text(tuple(v * SCALE for v in position), value,
                       font=self.fonts[size], fill=color, anchor=anchor)

    def line(self, points, color=NAVY, width=2):
        self.draw.line([(float(x) * SCALE, float(y) * SCALE) for x, y in points],
                       fill=color, width=max(1, round(width * SCALE)), joint='curve')

    def circle(self, x, y, radius, color):
        self.draw.ellipse(tuple(v * SCALE for v in
                               (x-radius, y-radius, x+radius, y+radius)), fill=color)

    def arrow(self, start, end):
        self.line([start, end], width=1.6)
        direction = np.array(start, dtype=float) - np.array(end, dtype=float)
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        for sign in [-1, 1]:
            tip = np.array(end) + direction * 11 + sign * normal * 4
            self.line([end, tip], width=1.6)


def background(font_path):
    image = Image.new('RGB', (WIDTH*SCALE, HEIGHT*SCALE), 'white')
    c = Canvas(image, font_path)
    c.circle(188, 73, 9, ORANGE)
    c.text((204, 58), 'Index-finger base', 22, ORANGE)
    for value in range(0, 81, 10):
        y = 524 - value * 5.44
        c.line([(584, y), (1255, y)], GRID, 1)
        c.text((568, y), str(value), 20, '#111111', 'rm')
    for value in [1.0, 0.8, 0.6, 0.4, 0.2]:
        x, _ = chart_point(value, 0)
        c.line([(x, 61), (x, 524)], GRID, 1)
        c.line([(x, 524), (x, 531)], '#111111', 1.5)
        c.text((x, 540), f'{value:.1f}', 20, '#111111', 'mt')
    c.line([(584, 61), (584, 524), (1255, 524)], '#111111', 1.5)
    c.text((920, 572), 'Normalized distance l = L(t) / L0', 22, '#111111', 'mt')
    axis = Image.new('RGBA', (265*SCALE, 32*SCALE), (255, 255, 255, 0))
    label = Canvas(axis, font_path)
    label.text((0, 0), 'Finger aperture d (mm)', 22, '#111111')
    axis = axis.rotate(90, expand=True)
    image.paste(axis, (507*SCALE, 185*SCALE), axis)
    # c.text((800, 614), 'Schematic animation (simulated data)', 17, '#888888')
    c.text((835, 430), 'Start point', 22)
    c.arrow((820, 465), chart_point(1.0, 5.0))
    return image


def render(progress, base, font_path):
    image = base.copy()
    c = Canvas(image, font_path)
    distance, aperture, joint_y, left_x, right_x, tip_y = motion(progress)
    sample = np.linspace(0.0, progress, max(2, int(progress*800)+1))
    history_l, history_d, _, history_left, history_right, history_y = motion(sample)
    for xs in [history_left, history_right]:
        c.line(zip(xs, history_y), TRACE, 2)
    # Root-to-object distance uses the same joint coordinates as l(t).
    for y in np.arange(joint_y, OBJECT_Y, 15):
        c.line([(JOINT_X, y), (JOINT_X, min(y+9, OBJECT_Y))], ORANGE, 2.5)
    c.line([(left_x, tip_y), (JOINT_X, joint_y), (right_x, tip_y)], TEAL, 8)
    c.circle(left_x, tip_y, 4, TEAL)
    c.circle(right_x, tip_y, 4, TEAL)
    c.line([(left_x, tip_y), (right_x, tip_y)], RED, 3)
    c.circle(JOINT_X, joint_y, 11, ORANGE)
    # The object is 30 schematic pixels wide, matching the final aperture.
    c.draw.rectangle((278*SCALE, 452*SCALE, 308*SCALE, 482*SCALE), fill='#333333')
    c.circle(JOINT_X, OBJECT_Y, 4, 'white')
    points = [chart_point(l, d) for l, d in zip(history_l, history_d)]
    c.line(points, TEAL, 4)
    x, y = chart_point(distance, aperture)
    c.circle(x, y, 9, RED)
    c.text((1033, 64), f'l = {distance:.2f}  d = {aperture:.0f} mm', 22)
    c.text((65, 521), f'l = {distance:.2f}', 22, ORANGE)
    if progress >= PEAK_PROGRESS:
        c.text((714, 62), 'Maximum aperture', 22)
        c.arrow((862, 97), chart_point(0.5, 72.0))
    if progress >= 0.98:
        c.text((1073, 438), 'Close to grasp', 22)
        c.arrow((1165, 430), chart_point(*motion(1.0)[:2]))
    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--duration', type=float, default=4.0)
    parser.add_argument('--no-open', action='store_true')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'Report/Animations/From_Scratch')
    args = parser.parse_args()
    if not 1 <= args.fps <= 120 or args.duration < 2:
        parser.error('FPS must be 1 to 120 and duration must be at least 2 seconds')
    font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/arial.ttf'
    if not font_path.is_file():
        parser.error(f'Font missing: {font_path}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / 'grasp_schematic_from_scratch.mp4'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                             args.fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError('Could not open MP4 writer')
    base = background(font_path)
    count = round(args.fps * args.duration)
    try:
        for index in range(count):
            # Initial and final holds, continuous easing through the full action.
            phase = index / max(1, count-1)
            progress = float(ease((phase-0.06)/0.82))
            image = render(progress, base, font_path)
            writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
            if index == count-1:
                image.save(args.output_dir / 'final_frame.png')
            if index % args.fps == 0:
                print(f'Rendering {index//args.fps}/{args.duration:g} seconds', flush=True)
    finally:
        writer.release()
    check = cv2.VideoCapture(str(video_path))
    actual_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if actual_count != count:
        raise RuntimeError(f'Expected {count} frames, found {actual_count}')
    print(f'Saved {actual_count} frames at {args.fps} FPS', flush=True)
    print(str(video_path.resolve()).encode('ascii', errors='backslashreplace').decode('ascii'))
    if not args.no_open:
        os.startfile(str(video_path.resolve()))


if __name__ == '__main__':
    main()
