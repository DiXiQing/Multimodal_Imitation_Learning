"""Camera-only adaptation of Record_mediapipe_continuous.py (Tk continuous trials)."""
import argparse
import csv
import json
import time
import threading
import queue
import tempfile
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from realtime_distance_speed import SpeedEstimator


def calibrate_cube_scale(mask, box, side_mm):
    """Approximate isotropic top-face scale from a rotated square contour."""
    if box is None or not np.isfinite(side_mm) or side_mm <= 0:
        raise ValueError('请先检测到完整方块，并填写正确的尺寸(cm)。')
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[c for c in contours if tuple(cv2.boundingRect(c))==tuple(box)]
    if not candidates:raise ValueError('未找到对应方块轮廓，请重新测算。')
    contour=max(candidates,key=cv2.contourArea)
    rect=cv2.minAreaRect(contour);a,b=map(float,rect[1])
    if min(a,b)<10 or max(a,b)/min(a,b)>1.3:
        raise ValueError('轮廓过小或不像完整正方形，请移开手并调整检测参数。')
    if cv2.contourArea(contour)/(a*b)<.8:
        raise ValueError('方块轮廓不完整，请无遮挡时重新测算。')
    x,y,w,h=box
    if x<=0 or y<=0 or x+w>=mask.shape[1] or y+h>=mask.shape[0]:
        raise ValueError('方块位于画面边缘，无法可靠测算。')
    return dict(mm_per_px=float(side_mm/((a+b)/2)),side_mm=float(side_mm),
        measured_edges_px=[a,b],method='Mean sides of rotated minimum-area rectangle of detected colour contour',
        limitation='Approximate scale at cube top-face height; perspective/hand-height differences remain.',
        calibration_time_unix_s=time.time())


class StartPositionGuide:
    """Radial distance only; readiness requires uninterrupted in-range samples."""
    def __init__(self, target=250., tolerance=5., hold=.5):
        self.target, self.tolerance, self.hold = target, tolerance, hold
        self.since = self.previous = None

    def update(self, distance, now):
        if self.previous is not None and not 0 < now-self.previous <= .2:
            self.since = None
        self.previous = now
        if distance is None or not np.isfinite(distance):
            self.since = None
            return 'missing', 0.
        if abs(distance-self.target) > self.tolerance:
            self.since = None
            return ('near' if distance < self.target else 'far'), 0.
        if self.since is None:
            self.since = now
        elapsed = now-self.since
        return ('ready' if elapsed >= self.hold else 'holding'), elapsed


def hand_reference_points(landmarks, shape):
    """Index MCP (display 2) and wrist (display 1), in cropped-image pixels."""
    h, w = shape[:2]
    def point(i):
        if landmarks is None:
            return None
        mark = landmarks.landmark[i]
        xy = np.array([mark.x*w, mark.y*h])
        return xy if np.isfinite(xy).all() and 0 <= xy[0] < w and 0 <= xy[1] < h else None
    return point(5), point(0)


def draw_hand_reference_candidates(display, landmarks):
    """Visual comparison only; does not change recorded measurement definitions."""
    h, w = display.shape[:2]
    def point(i):
        mark = landmarks.landmark[i]
        return np.array([mark.x*w, mark.y*h])
    candidates = [
        ('Wrist (0)', point(0), (0, 165, 255)),
        ('Index MCP (5)', point(5), (255, 120, 0)),
        ('Middle MCP (9)', point(9), (255, 0, 255)),
        ('Ring MCP (13)', point(13), (255, 255, 0)),
        ('Little MCP (17)', point(17), (80, 255, 80)),
        ('Palm mean (0,5,9,13,17)', np.mean([point(i) for i in (0, 5, 9, 13, 17)], axis=0), (180, 80, 255)),
    ]
    for n, (name, xy, color) in enumerate(candidates, 1):
        center = tuple(np.rint(xy).astype(int))
        cv2.circle(display, center, 8, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(display, center, 6, color, -1, cv2.LINE_AA)
        cv2.putText(display, str(n), (center[0]+9, center[1]-7), 0, .55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, str(n), (center[0]+9, center[1]-7), 0, .55, color, 1, cv2.LINE_AA)
        label = f'{n}: {name}'
        origin = (10, 52+(n-1)*19)
        cv2.putText(display, label, origin, 0, .44, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, label, origin, 0, .44, color, 1, cv2.LINE_AA)


class SpokenCounter:
    """Repeated spoken movement counts followed by two return counts."""
    def __init__(self, target_s):
        self.target_s=int(target_s)
        self.events=queue.SimpleQueue()
        self.cancel=threading.Event()
        self.worker=threading.Thread(target=self.run,daemon=True)
        self.process=None

    def start(self):
        self.worker.start()

    def schedule(self):
        return ([(0.,'开始','start')]+[(float(i),str(i),'count') for i in range(1,self.target_s+1)]
                +[(self.target_s+1.,'回位','finish'),(self.target_s+2.,'1','return_count'),(self.target_s+3.,'2','return_count')])

    def run(self):
        import subprocess
        import base64
        import shutil
        script="""$ErrorActionPreference='Stop'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding
Add-Type -AssemblyName System.Speech
$v=New-Object System.Speech.Synthesis.SpeechSynthesizer
$chinese=@($v.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'zh-*' })
if ($chinese.Count -gt 0) { $v.SelectVoice($chinese[0].VoiceInfo.Name) }
$v.Rate=5
while (($line=[Console]::ReadLine()) -ne $null) {
    if ($chinese.Count -eq 0) {
        if ($line -eq '开始') { $line='Start' }
        elseif ($line -eq '回位') { $line='Return' }
    }
    # Complete each phrase. A new cue must never cut off the previous number.
    $v.Speak($line)
}
$v.Dispose()
"""
        try:
            encoded=base64.b64encode(script.encode('utf-16-le')).decode('ascii')
            self.process=subprocess.Popen([shutil.which('pwsh.exe') or 'powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',encoded],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW)
            # Wait for the speech engine before starting the first cycle.
            if self.cancel.wait(2):return
            while not self.cancel.is_set():
                origin=time.perf_counter()
                for offset,phrase,cue in self.schedule():
                    if self.cancel.wait(max(0,origin+offset-time.perf_counter())):return
                    if self.process.poll() is not None:raise RuntimeError('Speech process exited')
                    self.process.stdin.write(phrase+'\n');self.process.stdin.flush()
                    self.events.put(dict(cue=cue,unix_time_s=time.time(),target_movement_s=self.target_s))
                if self.cancel.wait(max(0,origin+self.target_s+4.-time.perf_counter())):return
        except (OSError,RuntimeError):
            self.events.put(dict(cue='audio_error',unix_time_s=time.time(),target_movement_s=self.target_s))
        finally:
            if self.process is not None:
                if self.process.poll() is None:self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
                if self.process.stdin:self.process.stdin.close()

    def stop(self):
        self.cancel.set()
        self.worker.join(timeout=3)


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


def approach_axis(object_center, fingertip_midpoint):
    """Positive direction points from the object toward the initial hand position."""
    delta = np.asarray(fingertip_midpoint,dtype=float)-np.asarray(object_center,dtype=float)
    length = float(np.linalg.norm(delta))
    if delta.shape != (2,) or not np.isfinite(delta).all() or length < 1:
        raise ValueError('起始指尖中点与物体中心太近，无法确定接近方向；请将手放回初始位置。')
    return delta/length


def signed_hand_distance(object_center, fingertip_midpoint, axis):
    return float(np.dot(np.asarray(fingertip_midpoint)-np.asarray(object_center),axis))


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


class Trial:
    def __init__(self, root, size, width, height, fps, settings):
        self.color=settings.get('object_color','black')
        self.object_id=settings.get('object_id',f'{self.color}_{size:g}cm')
        self.size=size
        self.mm_per_px=(settings.get('pixel_calibration') or {}).get('mm_per_px')
        axis = settings.get('signed_distance_axis_xy')
        self.signed_axis = np.asarray(axis,dtype=float) if axis is not None else None
        self.path = root / f'TRIAL_Color_{self.color}_Size_{size:g}cm_{datetime.now():%Y%m%d_%H%M%S_%f}'
        self.frames = self.path / 'frames'
        self.frames.mkdir(parents=True, exist_ok=False)
        self.file = None
        self.video = None
        self.frame_count = 0
        try:
            self.video = cv2.VideoWriter(str(self.path/'recorded_video.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not self.video.isOpened():
                raise RuntimeError('Cannot create video')
            self.file = (self.path/'camera.csv').open('w', newline='', encoding='utf-8-sig')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['frame','t_camera','elapsed_s','finger_width','object_area',
                'object_x','object_y','thumb_x','thumb_y','index_x','index_y','hand_center_x',
                'hand_center_y','hand_object_distance','valid','object_center_source',
                'object_color','size_cm','object_id','hand_object_signed_distance',
                'index_mcp_x','index_mcp_y','wrist_x','wrist_y',
                'index_mcp_object_distance','wrist_object_distance',
                'index_mcp_distance_valid','wrist_distance_valid',
                'index_mcp_velocity_px_s','index_mcp_approach_speed_px_s','speed_window_span_s',
                'mm_per_px','finger_width_mm','hand_object_distance_mm','hand_object_signed_distance_mm',
                'index_mcp_object_distance_mm','wrist_object_distance_mm','index_mcp_velocity_mm_s','index_mcp_approach_speed_mm_s'])
            (self.path/'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
            self.started = time.perf_counter()
        except Exception:
            self.close()
            raise

    def save(self, raw, display, timestamp, aperture, area, a, thumb, index, source,
             index_mcp=None, wrist=None, velocity=None, speed_span=None):
        ok, encoded = cv2.imencode('.jpg', raw)
        if not ok:
            raise RuntimeError('Cannot encode frame')
        encoded.tofile(self.frames/f'frame_{self.frame_count:06d}.jpg')
        c = (thumb+index)/2 if thumb is not None else None
        valid = a is not None and c is not None
        xy = lambda p: list(p) if p is not None else ['', '']
        mm=lambda value: float(value)*self.mm_per_px if self.mm_per_px is not None and value is not None and np.isfinite(value) else ''
        self.writer.writerow([self.frame_count, timestamp, time.perf_counter()-self.started,
            aperture if c is not None else '', area, *xy(a), *xy(thumb), *xy(index), *xy(c),
            float(np.linalg.norm(a-c)) if valid else '', int(valid), source,
            self.color,self.size,self.object_id,
            signed_hand_distance(a,c,self.signed_axis) if valid and self.signed_axis is not None else '',
            *xy(index_mcp), *xy(wrist),
            float(np.linalg.norm(a-index_mcp)) if a is not None and index_mcp is not None else '',
            float(np.linalg.norm(a-wrist)) if a is not None and wrist is not None else '',
            int(a is not None and index_mcp is not None), int(a is not None and wrist is not None),
            float(velocity) if velocity is not None and np.isfinite(velocity) else '',
            -float(velocity) if velocity is not None and np.isfinite(velocity) else '',
            float(speed_span) if speed_span is not None and np.isfinite(speed_span) else '',
            self.mm_per_px if self.mm_per_px is not None else '',mm(aperture),
            mm(float(np.linalg.norm(a-c)) if valid else None),
            mm(signed_hand_distance(a,c,self.signed_axis) if valid and self.signed_axis is not None else None),
            mm(float(np.linalg.norm(a-index_mcp)) if a is not None and index_mcp is not None else None),
            mm(float(np.linalg.norm(a-wrist)) if a is not None and wrist is not None else None),
            mm(velocity),mm(-velocity if velocity is not None else None)])
        self.video.write(display)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.file.flush()

    def close(self):
        if self.video is not None:
            self.video.release()
            self.video = None
        if self.file is not None:
            self.file.close()
            self.file = None
        print(f'[Saved] {self.path} | frames={self.frame_count}', flush=True)


def main():
    import tkinter as tk
    from tkinter import messagebox, ttk
    from PIL import Image, ImageTk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=1000)
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--start-distance', type=float, default=250.)
    parser.add_argument('--start-tolerance', type=float, default=5.)
    parser.add_argument('--start-hold', type=float, default=0., help='Seconds to wait in range; default 0 gives immediate audio feedback')
    parser.add_argument('--color', choices=['black','red','blue'], default='black')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'Data'/'TopView')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('FPS must be positive')
    if (not all(np.isfinite(v) and v > 0 for v in (args.start_distance,args.start_tolerance))
            or not np.isfinite(args.start_hold) or args.start_hold < 0):
        parser.error('Distance/tolerance must be positive; hold time must be nonnegative.')
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
    except (ImportError, AttributeError) as exc:
        raise SystemExit('Use the same Python environment as Record_mediapipe_continuous.py; '
                         'MediaPipe with mp.solutions.hands is required.') from exc
    window = tk.Tk()
    window.title('俯视连续抓取录制 — 无 IMU / 黑、红、蓝物体')
    screen_w,screen_h=window.winfo_screenwidth(),window.winfo_screenheight()
    window.geometry(f'{min(1100,screen_w-60)}x{min(760,screen_h-100)}+20+20')
    window.resizable(True,True)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        window.destroy()
        raise RuntimeError('Cannot open camera; try --camera 0')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                          min_detection_confidence=.6, min_tracking_confidence=.6)
    speed_estimator = SpeedEstimator(window=5, max_gap=.10)
    start_guide = StartPositionGuide(args.start_distance,args.start_tolerance,args.start_hold)
    guide_ready_latched = False
    guide_last_sound = 0.
    calibration = None
    calibration_signature = None
    calibration_sample = None
    capture_frame_id = 0
    trial = None
    latest = None
    signed_axis = None
    source_dimensions = None
    crop_margins = (0, 0, 0, 0)
    locked = None
    closing = False
    status = tk.StringVar(value='先选择目标颜色和尺寸，检查绿框是否正确，再开始。')
    info = tk.StringVar()
    tk.Label(window, textvariable=status).pack()
    tk.Label(window, textvariable=info).pack()
    guide_text = tk.StringVar(value='起始位置引导：等待检测食指根部和物体')
    guide_label = tk.Label(window, textvariable=guide_text, fg='#b06000', font=('',11))
    guide_label.pack()
    mm_info=tk.StringVar(value='毫米：未标定，请先点击右侧“测算 mm/px”。')
    tk.Label(window,textvariable=mm_info,fg='#145e9c',font=('',11),wraplength=1000).pack()

    controls = tk.Frame(window)
    controls.pack(pady=4)
    actions=tk.Frame(window)
    actions.pack(pady=4)
    body=tk.Frame(window)
    body.pack(fill='both',expand=True,padx=8,pady=6)
    preview_container=tk.Frame(body)
    preview_container.pack(side='left',fill='both',expand=True)
    preview = tk.Label(preview_container)
    preview.pack(fill='both',expand=True)
    # Parameters scroll independently; preview height cannot push sliders off-screen.
    parameter_container=tk.Frame(body,width=350)
    parameter_container.pack(side='right',fill='y',padx=(8,0))
    canvas=tk.Canvas(parameter_container,width=330,highlightthickness=0)
    scrollbar=ttk.Scrollbar(parameter_container,orient='vertical',command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right',fill='y')
    canvas.pack(side='left',fill='both',expand=True)
    parameters=tk.Frame(canvas)
    parameter_window=canvas.create_window((0,0),window=parameters,anchor='nw')
    parameters.bind('<Configure>',lambda event:canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>',lambda event:canvas.itemconfigure(parameter_window,width=event.width))
    def scroll_parameters(event):
        canvas.yview_scroll(-int(event.delta/120),'units')
    def bind_scroll(widget):
        widget.bind('<MouseWheel>',scroll_parameters)
        for child in widget.winfo_children():bind_scroll(child)
    tk.Label(controls, text='目标颜色').pack(side='left')
    color = tk.StringVar(value=args.color)
    color_selector = ttk.Combobox(controls,textvariable=color,values=['black','red','blue'],state='readonly',width=7)
    color_selector.pack(side='left',padx=4)
    tk.Label(controls, text='尺寸(cm，球体填直径)').pack(side='left')
    size = tk.StringVar(value='3')
    size_entry=tk.Entry(controls, textvariable=size, width=5)
    size_entry.pack(side='left')
    start_values=tk.LabelFrame(parameters,text='起始距离引导（食指根部到物体）')
    start_values.pack(fill='x',padx=5,pady=5)
    start_target_text=tk.StringVar(value=f'{args.start_distance:g}')
    start_tolerance_text=tk.StringVar(value=f'{args.start_tolerance:g}')
    start_unit=tk.StringVar(value='px')
    applied_start_unit='px'
    for title,var in [('目标距离',start_target_text),('允许误差 ±',start_tolerance_text)]:
        row=tk.Frame(start_values);row.pack(fill='x')
        tk.Label(row,text=title,width=12,anchor='w').pack(side='left')
        tk.Entry(row,textvariable=var,width=9).pack(side='left')
    ttk.Combobox(start_values,textvariable=start_unit,values=['px','mm'],state='readonly',width=7).pack()
    def apply_start_target():
        nonlocal applied_start_unit,guide_ready_latched
        try:
            if trial is not None:raise ValueError('请结束录制后修改起始距离。')
            target=float(start_target_text.get());tolerance=float(start_tolerance_text.get())
            if not all(np.isfinite(v) and v>0 for v in (target,tolerance)):raise ValueError('请输入正数。')
            if start_unit.get()=='mm' and calibration is None:raise ValueError('请先测算 mm/px，再使用毫米引导。')
            applied_start_unit=start_unit.get()
            start_guide.target=target;start_guide.tolerance=tolerance
            start_guide.since=start_guide.previous=None;guide_ready_latched=False
        except ValueError as exc:messagebox.showerror('起始距离',str(exc))
    tk.Button(start_values,text='应用起始距离',command=apply_start_target).pack(fill='x')
    tk.Label(start_values,text='进入目标范围立即响；停留时每秒提示，离开即停止。',wraplength=295).pack()
    def ready_sound():
        def play():
            try:
                import winsound
                winsound.Beep(1000,180)
            except (ImportError,RuntimeError):pass
        threading.Thread(target=play,daemon=True).start()
    calibration_text=tk.StringVar(value='未标定：仅保存 px；将方块完整放入画面，手移开后测算。')
    tk.Label(parameters,textvariable=calibration_text,wraplength=300,justify='left').pack(fill='x',padx=5,pady=4)
    def current_calibration_signature():
        # Scale belongs to the camera geometry, not the selected object.
        return (tuple(crop_margins),source_dimensions)
    def measure_scale():
        nonlocal calibration,calibration_signature
        if trial is not None:
            messagebox.showinfo('标定','请先结束录制，再测算比例；同一次录制中比例固定。')
            return
        try:
            if calibration_sample is None or time.time()-calibration_sample['timestamp']>.5:
                raise ValueError('没有最新的方块图像，请检查检测画面。')
            measured=calibrate_cube_scale(calibration_sample['mask'],calibration_sample['box'],float(size.get())*10)
            calibration=measured
            calibration_signature=current_calibration_signature()
            calibration_text.set(f"已标定：{measured['mm_per_px']:.5f} mm/px；边长 {measured['side_mm']:g} mm。切换颜色/尺寸保留比例；摄像头移动后请重新测算。")
        except (ValueError,TypeError) as exc:messagebox.showerror('测算失败',str(exc))
    tk.Button(parameters,text='用当前方块测算 mm/px（尺寸使用上方 cm）',command=measure_scale).pack(fill='x',padx=5,pady=4)

    values = {}
    parameter_scales=[]
    for label, name, default, maximum in [('黑色亮度上限 V', 'v', 60, 255),
            ('黑色饱和度上限 S', 's', 120, 255),
            ('红/蓝：饱和度下限 S', 'color_s_min', 140, 255),
            ('红/蓝：亮度下限 V', 'color_v_min', 50, 255),
            ('最小面积 px²', 'area', 500, 3000)]:
        value = tk.IntVar(value=default)
        scale=tk.Scale(parameters, label=label, from_=10 if name in ('area','max_area') else 0, to=maximum,
                 orient='horizontal', variable=value, length=290)
        scale.pack(fill='x',padx=5,pady=3)
        parameter_scales.append(scale)
        values[name] = value
    values['max_area'] = tk.IntVar(value=args.width*args.height)
    area_row = tk.Frame(parameters)
    area_row.pack(fill='x',padx=5,pady=6)
    tk.Label(area_row,text='最大面积 px²').pack(side='left')
    max_area_text = tk.StringVar(value=str(values['max_area'].get()))
    max_area_entry = tk.Entry(area_row,textvariable=max_area_text,width=10)
    max_area_entry.pack(side='left',padx=5)
    def apply_max_area(event=None):
        try:
            limit = int(max_area_text.get().strip())
            if limit < values['area'].get():
                raise ValueError('最大面积必须不小于最小面积')
        except ValueError:
            status.set('最大面积请输入不小于最小面积的整数；当前生效值：'+str(values['max_area'].get()))
            return 'break'
        values['max_area'].set(limit)
        status.set(f'最大面积已应用：{limit} px²（与物体面积显示采用相同的外接框面积）')
        return 'break'
    max_area_button = tk.Button(area_row,text='应用',command=apply_max_area)
    max_area_button.pack(side='left')
    max_area_entry.bind('<Return>',apply_max_area)
    parameter_scales.extend([max_area_entry,max_area_button])
    crop_box = tk.LabelFrame(parameters,text='视野裁剪（像素，不拉伸）')
    crop_box.pack(fill='x',padx=5,pady=6)
    crop_text = [tk.StringVar(value='0') for _ in range(4)]
    for label,variable in zip(['裁掉左侧','裁掉顶部','裁掉右侧','裁掉底部'],crop_text):
        row = tk.Frame(crop_box)
        row.pack(fill='x')
        tk.Label(row,text=label).pack(side='left')
        entry = tk.Entry(row,textvariable=variable,width=8)
        entry.pack(side='right',padx=4,pady=2)
        parameter_scales.append(entry)
    crop_info = tk.StringVar(value='当前不裁剪。输入数值后点击应用。')
    tk.Label(crop_box,textvariable=crop_info,wraplength=285,justify='left').pack(fill='x')

    def apply_crop(reset=False):
        nonlocal crop_margins,locked,latest,signed_axis
        if trial is not None:
            return
        try:
            margins = (0,0,0,0) if reset else tuple(int(v.get().strip()) for v in crop_text)
            if source_dimensions is None:
                raise ValueError('等待摄像头预览后再应用裁剪')
            w,h = source_dimensions
            # Same validation as the real processing path, using the actual camera dimensions.
            cropped = crop_camera_frame(np.empty((h,w,3),dtype=np.uint8),margins)
        except ValueError as exc:
            messagebox.showerror('裁剪范围',str(exc))
            return
        crop_margins = margins
        for variable,value in zip(crop_text,margins):variable.set(str(value))
        locked = None
        latest = None
        signed_axis = None
        crop_info.set(f'原画面 {w}×{h} → 保存 {cropped.shape[1]}×{cropped.shape[0]}；左上原点随裁剪移动。')
        status.set('裁剪已应用，旧中心锁定已清除；请重新确认物体和手部。')

    for label,command in [('应用裁剪',apply_crop),('恢复完整视野',lambda:apply_crop(True))]:
        button = tk.Button(crop_box,text=label,command=command)
        button.pack(side='left',padx=4,pady=4)
        parameter_scales.append(button)
    show_mask = tk.BooleanVar(value=False)
    tk.Checkbutton(parameters, text='显示颜色掩膜（白色为候选区域）', variable=show_mask).pack(pady=8)
    metronome=None
    metro_config=None
    metro_events=[]
    guide_target=tk.StringVar(value='3')
    metro_info=tk.StringVar(value='开始 → 1、2、3 → 回位 → 1、2 → 开始，循环播报。')
    metro_box=tk.LabelFrame(parameters,text='循环语音计数')
    metro_box.pack(fill='x',padx=5,pady=6)
    target_row=tk.Frame(metro_box);target_row.pack(fill='x')
    tk.Label(target_row,text='接近计数').pack(side='left')
    for duration in ('1','2','3'):
        tk.Radiobutton(target_row,text=duration+' 秒',variable=guide_target,value=duration).pack(side='left')
    tk.Label(metro_box,textvariable=metro_info,wraplength=290,justify='left').pack(fill='x',pady=5)

    def cancel_guide(text='语音已停止。'):
        nonlocal metronome
        if metronome is not None:metronome.stop()
        metronome=None
        metro_button.config(text='开启循环语音（空格）')
        metro_info.set(text)

    def toggle_metronome(event=None):
        nonlocal metronome,metro_config
        if event is not None and isinstance(window.focus_get(),(tk.Entry,ttk.Entry)):return
        if metronome is not None:
            cancel_guide();return 'break'
        target=int(guide_target.get())
        metro_config=dict(mode='spoken_count_loop',target_movement_s=target,return_counts=2)
        metronome=SpokenCounter(target);metronome.start()
        metro_button.config(text='停止循环语音（空格）')
        metro_info.set('正在启动语音……\n开始 → '+ '、'.join(str(i) for i in range(1,target+1))+' → 回位 → 1、2 → 开始')
        return 'break'

    metro_button=tk.Button(metro_box,text='开启循环语音（空格）',command=toggle_metronome)
    metro_button.pack(pady=4)
    window.bind('<space>',toggle_metronome)

    def poll_metronome():
        if closing:return
        if metronome is not None:
            while not metronome.events.empty():
                event=metronome.events.get()
                if trial is not None:metro_events.append(event)
                if event['cue']=='audio_error':cancel_guide('语音启动失败，请检查 Windows 语音功能。');break
                if event['cue']=='start':metro_info.set('开始接近：'+str(metro_config['target_movement_s'])+' 秒')
                elif event['cue']=='finish':metro_info.set('回位：1、2；等待下一声“开始”。')
        window.after(30,poll_metronome)
    window.after(30,poll_metronome)
    bind_scroll(parameter_container)

    def reset_detection(*_):
        nonlocal locked,latest,signed_axis
        if trial is None:
            locked=None;latest=None;signed_axis=None
            status.set(f'目标：{color.get()}；已清除旧中心锁定，请重新确认绿框。')
    color.trace_add('write',reset_detection)
    for value in values.values():value.trace_add('write',reset_detection)

    def stop(event=None):
        nonlocal trial
        cancel_guide()
        if trial is not None:
            with (trial.path/'metronome_events.csv').open('w',newline='',encoding='utf-8-sig') as f:
                fields = ['cue','unix_time_s','target_movement_s']
                writer = csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                writer.writerows(metro_events)
            trial.close()
            trial = None
        color_selector.config(state='readonly')
        size_entry.config(state='normal')
        for scale in parameter_scales:
            scale.config(state='normal')
        status.set('已结束本次，预览继续；可以移动物体后开始下一次。')

    def start(event=None):
        nonlocal trial,signed_axis
        if trial is not None:
            return
        try:
            object_size = float(size.get())
            if not np.isfinite(object_size) or object_size <= 0:
                raise ValueError('物体尺寸必须大于 0')
            if latest is None or latest['box'] is None or latest['thumb'] is None:
                raise ValueError('需同时检测到所选颜色物体和手部；检查掩膜并调整阈值。')
            initial_center = (latest['thumb']+latest['index'])/2
            new_axis = approach_axis(latest['object_center'],initial_center)
            settings = {name: value.get() for name, value in values.items()}
            settings['signed_distance_axis_xy'] = new_axis.tolist()
            settings['signed_distance_initial_object_center_xy'] = latest['object_center'].tolist()
            settings['signed_distance_initial_hand_center_xy'] = initial_center.tolist()
            settings['signed_distance_definition'] = '(current fingertip midpoint - current object center) dot fixed initial axis; positive on initial-hand side, negative after crossing the perpendicular zero line. Pixels; not signed Euclidean distance.'
            settings['metronome_at_recording_start'] = metro_config if metronome is not None else None
            settings['approach_guide'] = dict(mode='spoken_count_loop',target_movement_s=int(guide_target.get()),return_counts=2)
            settings['camera_source_width'],settings['camera_source_height'] = source_dimensions
            settings['crop_margins_px'] = dict(zip(('left','top','right','bottom'),crop_margins))
            settings['coordinate_origin'] = 'Top-left of cropped image; add crop left/top for source coordinates.'
            settings.update(camera=args.camera, width=latest['width'], height=latest['height'],
                object_color=color.get(),object_id=f'{color.get()}_{object_size:g}cm',
                hsv_ranges=color_ranges(color.get(),settings['v'],settings['s'],settings['color_s_min'],settings['color_v_min']),
                fps=args.fps, size_cm=object_size, center_source='locked' if locked is not None else 'detected',
                locked_center=locked.tolist() if locked is not None else None,
                base_workflow='Record_mediapipe_continuous.py', fingertip_method='MediaPipe landmarks 4,8')
            settings['distance_measurements'] = {
                'hand_object_distance': 'Euclidean distance from landmarks 4/8 midpoint to object centre (legacy)',
                'index_mcp_object_distance': 'Euclidean distance from index MCP landmark 5 (display node 2) to object centre',
                'wrist_object_distance': 'Euclidean distance from wrist landmark 0 (display node 1) to object centre',
                'units': 'px', 'primary_display': 'index_mcp_object_distance',
                'hand_object_signed_distance': 'Legacy signed midpoint projection retained separately'}
            settings['speed_estimation'] = dict(distance_column='index_mcp_object_distance',
                method='Least-squares slope over trailing 5 consecutive valid distance/time samples',
                window_frames=5, max_gap_s=.10, units='px/s',
                velocity_sign='Distance derivative; negative approaching',
                approach_speed_sign='Negative distance derivative; positive approaching')
            settings['pixel_calibration'] = dict(calibration) if calibration is not None else None
            settings['start_position_guide'] = dict(target_distance=start_guide.target,
                tolerance=start_guide.tolerance,units=applied_start_unit,hold_s=args.start_hold,direction_constraint=False,
                ready_at_recording_start=latest.get('start_position_ready',False),
                initial_index_mcp_distance_px=latest.get('index_mcp_distance'),enforce_recording_start=False)
            trial = Trial(args.output, object_size, latest['width'], latest['height'], args.fps, settings)
            speed_estimator.reset()
            signed_axis = new_axis
            metro_events.clear()
            color_selector.config(state='disabled')
            size_entry.config(state='disabled')
            for scale in parameter_scales:
                scale.config(state='disabled')
            status.set('录制中：'+trial.path.name)
        except Exception as exc:
            messagebox.showerror('无法开始', str(exc))

    def lock_center(event=None):
        nonlocal locked
        if trial is not None:
            return
        if locked is not None:
            locked = None
            status.set('已恢复每帧实时检测中心。')
        elif latest is not None and latest['box'] is not None:
            x,y,w,h = latest['box']
            locked = np.array([x+w/2, y+h/2])
            status.set('中心已锁定；物体必须保持不动。按 L 可解除。')
        else:
            status.set('未检测到所选颜色物体，无法锁定。')

    def close(event=None):
        nonlocal closing
        if closing:
            return
        closing = True
        if metronome is not None:
            metronome.stop()
        stop()
        cap.release()
        hands.close()
        window.destroy()

    for text, fn in [('下一次实验 / 开始 (N)',start), ('结束本次 (S)',stop),
                     ('锁定 / 解除中心 (L)',lock_center), ('保存退出 (Q)',close)]:
        tk.Button(actions, text=text, command=fn).pack(side='left', padx=3)
    for key, fn in [('n',start),('s',stop),('l',lock_center),('q',close)]:
        window.bind('<'+key+'>',fn)
        window.bind('<'+key.upper()+'>',fn)
    window.protocol('WM_DELETE_WINDOW',close)

    def update():
        nonlocal latest,source_dimensions,capture_frame_id,calibration_sample,calibration,calibration_signature,guide_ready_latched,guide_last_sound
        if closing:
            return
        began = time.perf_counter()
        try:
            ret, raw = cap.read()
            if not ret:
                raise RuntimeError('Camera read failed')
            timestamp = time.time()
            capture_frame_id += 1
            source_dimensions = (raw.shape[1],raw.shape[0])
            raw = crop_camera_frame(raw,crop_margins)
            display = raw.copy()
            thumb, index, landmarks = detect_fingers(raw,hands)
            index_mcp, wrist = hand_reference_points(landmarks, raw.shape)
            (area,box), mask = detect_cube(raw, values['v'].get(), values['s'].get(), values['area'].get(),
                color=color.get(),color_s_min=values['color_s_min'].get(),color_v_min=values['color_v_min'].get(),
                max_area=values['max_area'].get())
            calibration_sample=dict(mask=mask.copy(),box=box,timestamp=timestamp)
            if trial is None and calibration is not None and calibration_signature!=current_calibration_signature():
                calibration=None;calibration_signature=None
                calibration_text.set('裁剪或画面分辨率已变化，请重新测算 mm/px。')
            a = locked.copy() if locked is not None else None
            if box is not None:
                x,y,w,h = box
                cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
                if locked is None:
                    a = np.array([x+w/2,y+h/2])
            if landmarks is not None:
                mp.solutions.drawing_utils.draw_landmarks(display,landmarks,mp_hands.HAND_CONNECTIONS)
            if index_mcp is not None:
                cv2.circle(display,tuple(np.rint(index_mcp).astype(int)),4,(255,120,0),-1,cv2.LINE_AA)
            aperture = float(np.linalg.norm(thumb-index)) if thumb is not None else None
            if a is not None:
                cv2.circle(display,tuple(a.astype(int)),5,(0,0,255),-1)
            distance = None
            if thumb is not None and a is not None:
                c = (thumb+index)/2
                distance = float(np.linalg.norm(a-c))
            index_mcp_distance = float(np.linalg.norm(a-index_mcp)) if a is not None and index_mcp is not None else None
            wrist_distance = float(np.linalg.norm(a-wrist)) if a is not None and wrist is not None else None
            guide_scale=trial.mm_per_px if trial is not None else (calibration or {}).get('mm_per_px')
            guide_distance=index_mcp_distance
            if applied_start_unit=='mm':
                guide_distance=index_mcp_distance*guide_scale if index_mcp_distance is not None and guide_scale is not None else None
            guide_state, held = start_guide.update(guide_distance,time.perf_counter())
            sound_now=time.perf_counter()
            if guide_state=='ready':
                if not guide_ready_latched or sound_now-guide_last_sound>=1.:
                    ready_sound();guide_last_sound=sound_now
                guide_ready_latched=True
            elif guide_state not in ('ready','holding'):
                guide_ready_latched=False
            messages = {'missing':'等待检测食指根部和物体', 'near':'太近，请后退',
                'far':'太远，请靠近','holding':f'距离合适，请保持 {held:.1f}/{args.start_hold:g} 秒',
                'ready':'起始位置就绪'}
            guide_text.set(f'目标 {start_guide.target:g}±{start_guide.tolerance:g} {applied_start_unit} | '+messages[guide_state]+('（录制中，仅提示）' if trial is not None else ''))
            guide_label.config(fg='#16853b' if guide_state in ('holding','ready') else '#b06000')
            if index_mcp is not None and guide_state in ('holding','ready'):
                cv2.circle(display,tuple(np.rint(index_mcp).astype(int)),4,(40,200,60),-1,cv2.LINE_AA)
            if index_mcp_distance is not None:
                cv2.line(display,tuple(a.astype(int)),tuple(np.rint(index_mcp).astype(int)),(255,120,0),1,cv2.LINE_AA)
            _, velocity, speed_span, _ = speed_estimator.step(
                index_mcp_distance if index_mcp_distance is not None else np.nan,
                timestamp, capture_frame_id)
            latest = dict(box=box, thumb=thumb, index=index, object_center=a,timestamp=timestamp,
                          start_position_ready=guide_state=='ready' and trial is None,
                          index_mcp_distance=index_mcp_distance,
                          width=raw.shape[1], height=raw.shape[0])
            signed_distance = None
            if a is not None and thumb is not None and signed_axis is not None:
                c = (thumb+index)/2
                signed_distance = signed_hand_distance(a,c,signed_axis)
            source = 'locked' if locked is not None else ('detected' if box is not None else 'missing')
            if trial is not None:
                cv2.putText(display,'REC',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
                trial.save(raw,display,timestamp,aperture,area,a,thumb,index,source,index_mcp,wrist,velocity,speed_span)
            fw = f'{aperture:.1f}' if aperture is not None else '未检测'
            dist = f'{distance:.1f}' if distance is not None else '无效'
            signed_text = f'{signed_distance:+.1f}px' if signed_distance is not None else '开始录制时确定方向'
            mcp_text = f'{index_mcp_distance:.1f}' if index_mcp_distance is not None else '无效'
            wrist_text = f'{wrist_distance:.1f}' if wrist_distance is not None else '无效'
            speed_text = f'{-velocity:+.1f}' if np.isfinite(velocity) else '等待5帧'
            info.set(f'{color.get()} | 开合={fw}px | 食指根(2)={mcp_text}px | 接近速度={speed_text}px/s | 手腕(1)={wrist_text}px | 旧中点={dist}px | Frames={trial.frame_count if trial else 0}')
            active_scale=trial.mm_per_px if trial is not None else (calibration or {}).get('mm_per_px')
            if active_scale is not None:
                fmt=lambda value: f'{value*active_scale:.1f}' if value is not None and np.isfinite(value) else '--'
                mm_info.set(f"毫米：开合={fmt(aperture)} mm | 食指根={fmt(index_mcp_distance)} mm | 手腕={fmt(wrist_distance)} mm | 旧中点={fmt(distance)} mm | 接近速度={fmt(-velocity)} mm/s")
            else:
                mm_info.set('毫米：未标定，请先点击右侧“测算 mm/px”。')

            if show_mask.get():
                shown = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
            else:
                shown = cv2.cvtColor(display,cv2.COLOR_BGR2RGB)
            # Only resize GUI preview; measurements and saved frames keep original resolution.
            image = Image.fromarray(shown)
            image.thumbnail((max(1,preview_container.winfo_width()-8),
                             max(1,preview_container.winfo_height()-8)))
            photo = ImageTk.PhotoImage(image)
            preview.config(image=photo)
            preview.image=photo
        except Exception as exc:
            messagebox.showerror('录制错误',str(exc))
            close()
            return
        window.after(max(1,int(1000*(1/args.fps-(time.perf_counter()-began)))),update)
    try:
        update()
        window.mainloop()
    finally:
        if not closing:
            close()


if __name__ == '__main__':
    main()
