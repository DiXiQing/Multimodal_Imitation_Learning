"""Compare causal radial velocity estimates with a live overhead camera."""
import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Thread

import cv2
import numpy as np
from scipy.signal import savgol_coeffs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'New/Result/RealTime_Velocity_Comparison'
CAMERA = 1
FPS = 30
EXTRA_LEFT_CROP = 100
MAX_GAP_S = .10
METHODS = (
    ('backward', 'Raw backward difference', '#888888'),
    ('mean4', 'Moving average (4)', '#3485b8'),
    ('mean6', 'Moving average (6)', '#e59a32'),
    ('span5', 'Distance change over 5 frames', '#519e58'),
    ('sg4', 'One-sided quadratic SG (4)', '#8e63b0'),
    ('fit5', 'Linear fit (5 frames)', '#d05048'),
    ('fit7', 'Linear fit (7 frames)', '#00856e'),
)
DEFAULT_VISIBLE_METHODS = {'mean4', 'fit5', 'fit7'}
SG_WEIGHTS = savgol_coeffs(4, 2, pos=3, use='dot')

PREVIEW_SECONDS = 8

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

def detect_hand(frame, hands):
    """Detect the hand in source coordinates, including near the crop edges."""
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None, None
    landmarks = result.multi_hand_landmarks[0]
    h, w = frame.shape[:2]
    node = landmarks.landmark[5]
    mcp = np.array([node.x*w, node.y*h])
    if not np.isfinite(mcp).all() or not np.all((mcp >= 0) & (mcp < [w, h])):
        return landmarks, None
    return landmarks, mcp

def size_settings(size):
    dataset = ROOT / 'Data/1101_D' / f'{size*10}mm'
    candidates = sorted(p for p in dataset.glob('*/settings.json')
                        if p.parent.name.startswith('grasp_') or p.parent.name[:8].isdigit())
    if not candidates:
        raise ValueError(f'No detection settings for {size*10} mm')
    path = candidates[0]
    return path, json.loads(path.read_text(encoding='utf-8-sig'))


class VelocityComparison:
    def __init__(self):
        self.frames = deque(maxlen=7)
        self.raw = deque(maxlen=6)

    def step(self, timestamp, distance):
        result = dict.fromkeys((key for key, _, _ in METHODS), np.nan)
        if not np.isfinite(distance) or distance <= 0:
            self.frames.clear()
            self.raw.clear()
            return result
        if self.frames and not 0 < timestamp-self.frames[-1][0] <= MAX_GAP_S:
            self.frames.clear()
            self.raw.clear()
        if self.frames:
            previous_time, previous_distance = self.frames[-1]
            result['backward'] = (distance-previous_distance)/(timestamp-previous_time)
            self.raw.append(result['backward'])
        self.frames.append((timestamp, distance))
        if len(self.raw) >= 4:
            recent = np.asarray(self.raw)[-4:]
            result['mean4'] = float(recent.mean())
            result['sg4'] = float(recent @ SG_WEIGHTS)
            past_time, past_distance = self.frames[-5]
            result['span5'] = (distance-past_distance)/(timestamp-past_time)
        if len(self.raw) == 6:
            result['mean6'] = float(np.mean(self.raw))
        samples = np.asarray(self.frames)
        for key, window in (('fit5', 5), ('fit7', 7)):
            if len(samples) >= window:
                times, distances = samples[-window:].T
                relative_times = times - times[-1]
                centered_times = relative_times - relative_times.mean()
                centered_distances = distances - distances[-1]
                result[key] = float(
                    (centered_times @ centered_distances) / (centered_times @ centered_times)
                )
        return result


def main():
    import tkinter as tk
    from tkinter import messagebox, ttk

    import mediapipe as mp
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from PIL import Image, ImageTk

    presets = {size: size_settings(size)[1] for size in (2, 3, 4)}
    camera_settings = presets[3]
    left, top, right, bottom = (int(camera_settings.get('crop_margins_px', {}).get(k, 0))
                                for k in ('left', 'top', 'right', 'bottom'))
    margins = left + EXTRA_LEFT_CROP, top, right, bottom
    width = int(camera_settings.get('camera_source_width', camera_settings['width']+left+right))
    height = int(camera_settings.get('camera_source_height', camera_settings['height']+top+bottom))
    scale = float(camera_settings['pixel_calibration']['mm_per_px'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('A valid mm/px calibration is required.')

    ui = tk.Tk()
    ui.title('实时速度比较')
    ui.geometry('1120x570')
    ui.minsize(900, 510)
    ui.columnconfigure(0, weight=1)
    ui.rowconfigure(1, weight=1)

    toolbar = ttk.Frame(ui, padding=(12, 10))
    toolbar.grid(row=0, column=0, sticky='ew')
    ttk.Label(toolbar, text='方块尺寸').pack(side='left', padx=(0, 8))
    selected_size = tk.IntVar(value=3)
    status = tk.StringVar(value='正在打开摄像头…')
    note = tk.StringVar(value='可直接看实时曲线，或点击开始记录保存一次抓取。')
    ttk.Label(ui, textvariable=status, padding=(12, 4)).grid(row=2, column=0, sticky='w')
    ttk.Label(ui, textvariable=note, padding=(12, 4)).grid(row=3, column=0, sticky='w')

    panels = ttk.Panedwindow(ui, orient='horizontal')
    panels.grid(row=1, column=0, sticky='nsew', padx=12)
    preview_panel = ttk.Frame(panels, width=460)
    preview_panel.pack_propagate(False)
    chart_panel = ttk.Frame(panels)
    panels.add(preview_panel, weight=4)
    panels.add(chart_panel, weight=5)
    preview_widget = ttk.Label(preview_panel, anchor='center')
    preview_widget.pack(fill='both', expand=True)

    figure = Figure(figsize=(5.6, 4.2), layout='tight')
    ax = figure.add_subplot()
    ax.set(xlabel='Time (s)', ylabel='Radial velocity (mm/s)', xlim=(0, 1), ylim=(-100, 100))
    ax.axhline(0, color='#b2b2b2', linewidth=.8)
    ax.grid(alpha=.2)
    lines = {
        key: ax.plot([], [], color=color, label=label, visible=key in DEFAULT_VISIBLE_METHODS)[0]
        for key, label, color in METHODS
    }
    chart = FigureCanvasTkAgg(figure, master=chart_panel)
    chart.get_tk_widget().pack(fill='both', expand=True)

    live_rows = deque()
    rows = []
    recording = False
    recording_origin = locked = latest = None
    start_time = 0.
    chart_updated = 0.
    estimator = VelocityComparison()

    def redraw(show_rows=None):
        samples = list((rows if recording else live_rows) if show_rows is None else show_rows)
        times = [row['elapsed_s'] for row in samples]
        for key, line in lines.items():
            line.set_data(times, [row[key] for row in samples])
        ax.relim(visible_only=True)
        ax.autoscale(enable=True, axis='y')
        start = times[0] if times else 0.
        ax.set_xlim(start, max(start+1., times[-1]) if times else 1.)
        title = f'{selected_size.get()*10} mm | '
        ax.set_title(title + ('Recording' if recording else 'Live preview'))
        chart.draw_idle()

    def set_line_visibility(key, enabled):
        lines[key].set_visible(enabled.get())
        redraw()

    methods_panel = ttk.Frame(chart_panel, padding=(4, 4))
    methods_panel.pack(fill='x')
    visibility = {}
    style = ttk.Style(ui)
    for index, (key, label, color) in enumerate(METHODS):
        enabled = tk.BooleanVar(value=key in DEFAULT_VISIBLE_METHODS)
        visibility[key] = enabled
        method_style = f'{key}.TCheckbutton'
        style.configure(method_style, foreground=color)
        control = ttk.Checkbutton(methods_panel, text=label, variable=enabled, style=method_style,
                                  command=lambda key=key, enabled=enabled: set_line_visibility(key, enabled))
        control.grid(row=index//2, column=index%2, sticky='w', padx=(0, 8))

    def clear_preview():
        nonlocal latest, estimator, start_time
        latest = None
        estimator = VelocityComparison()
        live_rows.clear()
        start_time = time.perf_counter()
        redraw()

    def switch_size():
        nonlocal locked
        locked = None
        lock_button.config(text='锁定方块中心 (L)')
        note.set('已切换方块尺寸，正在重新检测。')
        clear_preview()

    size_buttons = []
    for size in (2, 3, 4):
        color = presets[size].get('object_color', 'black')
        name = {'black': '黑', 'blue': '蓝', 'red': '红'}.get(color, color)
        button = ttk.Radiobutton(toolbar, text=f'{size*10} mm · {name}', variable=selected_size,
                                 value=size, command=switch_size)
        button.pack(side='left', padx=(0, 8))
        size_buttons.append(button)

    def start():
        nonlocal recording, recording_origin, rows, start_time, estimator
        if recording:
            return
        if latest is None or not latest[0]:
            note.set('开始失败。请先让食指根部和方块中心都出现在画面中。')
            return
        recording = True
        recording_origin = latest[1].copy()
        start_time = time.perf_counter()
        rows = []
        estimator = VelocityComparison()
        for button in size_buttons:
            button.config(state='disabled')
        start_button.config(state='disabled')
        lock_button.config(state='disabled')
        stop_button.config(state='normal')
        note.set('正在记录，抓取完成后点击结束并保存。')
        redraw()

    def save():
        nonlocal recording
        if not recording:
            return
        recording = False
        for button in (*size_buttons, start_button, lock_button):
            button.config(state='normal')
        stop_button.config(state='disabled')
        if not rows:
            note.set('本次没有采集到帧。')
            clear_preview()
            return
        folder = OUTPUT / f'trial_{datetime.now():%Y%m%d_%H%M%S_%f}'
        folder.mkdir(parents=True)
        with (folder/'velocity.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        redraw(rows)
        ax.set_title(f'{selected_size.get()*10} mm')
        visible_lines = [line for line in lines.values() if line.get_visible()]
        if visible_lines:
            ax.legend(handles=visible_lines, fontsize=8)
        figure.savefig(folder/'velocity_comparison.png', dpi=180)
        if ax.get_legend():
            ax.get_legend().remove()
        note.set(f'已保存 {len(rows)} 帧 · {folder.name}')
        print(f'Saved: {folder}', flush=True)
        clear_preview()

    def lock():
        nonlocal locked
        if recording:
            return
        if locked is not None:
            locked = None
            lock_button.config(text='锁定方块中心 (L)')
            note.set('已解除方块中心锁定。')
        elif latest is not None and latest[1] is not None:
            locked = latest[1].copy()
            lock_button.config(text='解除中心锁定 (L)')
            note.set('方块中心已锁定。')
        else:
            note.set('未找到方块中心，请先确认尺寸和方块位置。')
            return
        clear_preview()

    start_button = ttk.Button(toolbar, text='开始记录 (N)', command=start)
    start_button.pack(side='left', padx=(12, 4))
    stop_button = ttk.Button(toolbar, text='结束并保存 (S)', command=save, state='disabled')
    stop_button.pack(side='left', padx=4)
    lock_button = ttk.Button(toolbar, text='锁定方块中心 (L)', command=lock)
    lock_button.pack(side='left', padx=4)

    def close():
        try:
            save()
        except OSError as error:
            messagebox.showerror('保存失败', str(error), parent=ui)
        finally:
            ui.destroy()

    ttk.Button(toolbar, text='退出 (Q)', command=close).pack(side='right')
    ui.protocol('WM_DELETE_WINDOW', close)
    ui.bind('<Key>', lambda event: {'n': start, 's': save, 'l': lock, 'q': close}.get(
        event.char.lower(), lambda: None)())
    clear_preview()

    cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            messagebox.showerror('摄像头无法打开', f'无法打开摄像头 {CAMERA}。请关闭正在使用摄像头的录制或实时推理窗口。', parent=ui)
            ui.destroy()
            return
        for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height),
                            (cv2.CAP_PROP_FPS, FPS)):
            cap.set(prop, value)
        results = SimpleQueue()
        stopped = Event()
        active_size = [selected_size.get()]

        def capture():
            with mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=.5,
                                          min_tracking_confidence=.5) as hands:
                while not stopped.is_set():
                    capture_frame(hands)

        def capture_frame(hands):
            began = time.perf_counter()
            ok, frame = cap.read()
            timestamp = time.perf_counter()
            if not ok or frame.shape[:2] != (height, width):
                actual = f'{frame.shape[1]}×{frame.shape[0]}' if ok else '读取失败'
                raise RuntimeError(f'当前画面：{actual}，预期 {width}×{height}。')

            size = active_size[0]
            preset = presets[size]
            landmarks, mcp = detect_hand(frame, hands)
            cropped = crop_camera_frame(frame, margins)
            (_, box), _ = detect_cube(cropped, int(preset.get('v', 60)), int(preset.get('s', 120)),
                                     int(preset.get('area', 500)), color=preset.get('object_color', 'black'),
                                     color_s_min=int(preset.get('color_s_min', 140)),
                                     color_v_min=int(preset.get('color_v_min', 50)),
                                     max_area=int(preset.get('max_area', 640000)))
            if landmarks is not None:
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            results.put((size, timestamp, frame, mcp, box))
            stopped.wait(max(0., 1/FPS - (time.perf_counter()-began)))

        def worker():
            try:
                capture()
            except Exception as error:
                results.put(error)

        def update_frame():
            nonlocal latest, chart_updated
            active_size[0] = selected_size.get()
            last_preview = None
            while True:
                try:
                    result = results.get_nowait()
                except Empty:
                    break
                if isinstance(result, Exception):
                    messagebox.showerror('摄像头画面异常', str(result), parent=ui)
                    close()
                    return
                size, timestamp, frame, mcp, box = result
                if size != active_size[0] or timestamp < start_time:
                    continue
                center = locked
                if box is not None:
                    x, y, w, h = box
                    x, y = x+margins[0], y+margins[1]
                    if center is None:
                        center = np.array([x+w/2, y+h/2])
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (78, 220, 100), 2)
                if mcp is not None:
                    cv2.circle(frame, tuple(mcp.astype(int)), 6, (255, 210, 35), -1)
                if center is not None:
                    cv2.circle(frame, tuple(center.astype(int)), 4, (78, 220, 100), -1)

                valid = center is not None and mcp is not None
                shifted = valid and recording and np.linalg.norm(center-recording_origin) > 40
                valid = valid and not shifted
                latest = (valid, center)
                distance = float(np.linalg.norm(mcp-center))*scale if valid else np.nan
                if valid:
                    cv2.line(frame, tuple(center.astype(int)), tuple(mcp.astype(int)), (0, 240, 255), 2)
                hand_status = '已识别' if mcp is not None else '未识别'
                cube_status = '已锁定' if locked is not None else '已识别' if center is not None else '未识别'
                detail = f'距离 {distance:.1f} mm' if valid else '方块中心偏移超过 40 px' if shifted else '距离暂不可用'
                last_preview = frame, f'食指根部 {hand_status}    方块 {cube_status}    {detail}    标定 {scale:g} mm/px'

                sample = dict(elapsed_s=timestamp-start_time, distance_mm=distance,
                              **estimator.step(timestamp, distance))
                if recording:
                    rows.append(sample)
                else:
                    live_rows.append(sample)
                    while sample['elapsed_s'] - live_rows[0]['elapsed_s'] > PREVIEW_SECONDS:
                        live_rows.popleft()
            if last_preview is not None:
                frame, message = last_preview
                status.set(message)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((max(1, preview_widget.winfo_width()-4),
                                 max(1, preview_widget.winfo_height()-4)), Image.Resampling.BILINEAR)
                preview_widget.image = ImageTk.PhotoImage(image, master=ui)
                preview_widget.config(image=preview_widget.image)
                if timestamp-chart_updated > .20:
                    redraw()
                    chart_updated = timestamp
            ui.after(20, update_frame)

        thread = Thread(target=worker, daemon=True)
        thread.start()
        try:
            ui.after(20, update_frame)
            ui.mainloop()
        finally:
            stopped.set()
            thread.join()
    finally:
        cap.release()


if __name__ == '__main__':
    main()
