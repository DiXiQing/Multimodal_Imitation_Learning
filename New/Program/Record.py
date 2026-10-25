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
                'object_color','size_cm','object_id','hand_object_signed_distance'])
            (self.path/'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
            self.started = time.perf_counter()
        except Exception:
            self.close()
            raise

    def save(self, raw, display, timestamp, aperture, area, a, thumb, index, source):
        ok, encoded = cv2.imencode('.jpg', raw)
        if not ok:
            raise RuntimeError('Cannot encode frame')
        encoded.tofile(self.frames/f'frame_{self.frame_count:06d}.jpg')
        c = (thumb+index)/2 if thumb is not None else None
        valid = a is not None and c is not None
        xy = lambda p: list(p) if p is not None else ['', '']
        self.writer.writerow([self.frame_count, timestamp, time.perf_counter()-self.started,
            aperture if c is not None else '', area, *xy(a), *xy(thumb), *xy(index), *xy(c),
            float(np.linalg.norm(a-c)) if valid else '', int(valid), source,
            self.color,self.size,self.object_id,
            signed_hand_distance(a,c,self.signed_axis) if valid and self.signed_axis is not None else ''])
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
    parser.add_argument('--color', choices=['black','red','blue'], default='black')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'Data'/'TopView')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('FPS must be positive')
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
            trial = Trial(args.output, object_size, latest['width'], latest['height'], args.fps, settings)
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
        nonlocal latest,source_dimensions
        if closing:
            return
        began = time.perf_counter()
        try:
            ret, raw = cap.read()
            if not ret:
                raise RuntimeError('Camera read failed')
            timestamp = time.time()
            source_dimensions = (raw.shape[1],raw.shape[0])
            raw = crop_camera_frame(raw,crop_margins)
            display = raw.copy()
            thumb, index, landmarks = detect_fingers(raw,hands)
            (area,box), mask = detect_cube(raw, values['v'].get(), values['s'].get(), values['area'].get(),
                color=color.get(),color_s_min=values['color_s_min'].get(),color_v_min=values['color_v_min'].get(),
                max_area=values['max_area'].get())
            a = locked.copy() if locked is not None else None
            if box is not None:
                x,y,w,h = box
                cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
                if locked is None:
                    a = np.array([x+w/2,y+h/2])
            if landmarks is not None:
                mp.solutions.drawing_utils.draw_landmarks(display,landmarks,mp_hands.HAND_CONNECTIONS)
            aperture = float(np.linalg.norm(thumb-index)) if thumb is not None else None
            if a is not None:
                cv2.circle(display,tuple(a.astype(int)),5,(0,0,255),-1)
            distance = None
            if thumb is not None and a is not None:
                c = (thumb+index)/2
                distance = float(np.linalg.norm(a-c))
                cv2.line(display,tuple(a.astype(int)),tuple(c.astype(int)),(0,255,255),2)
            latest = dict(box=box, thumb=thumb, index=index, object_center=a,timestamp=timestamp,
                          width=raw.shape[1], height=raw.shape[0])
            signed_distance = None
            if a is not None and thumb is not None and signed_axis is not None:
                c = (thumb+index)/2
                signed_distance = signed_hand_distance(a,c,signed_axis)
                # Display the fixed projection axis and its perpendicular zero reference.
                h,w = display.shape[:2]
                perpendicular = np.array([-signed_axis[1],signed_axis[0]])
                endpoints = [tuple(np.rint(a+sign*perpendicular*max(w,h)).astype(int)) for sign in [-1,1]]
                cv2.line(display,endpoints[0],endpoints[1],(200,100,255),1,cv2.LINE_AA)
                cv2.arrowedLine(display,tuple(np.rint(a).astype(int)),
                    tuple(np.rint(a+signed_axis*60).astype(int)),(200,100,255),2,tipLength=.18)
                label = f'Signed distance: {signed_distance:+.1f} px'
                cv2.putText(display,label,(10,h-12),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(display,label,(10,h-12),cv2.FONT_HERSHEY_SIMPLEX,.55,(200,100,255),1,cv2.LINE_AA)
            source = 'locked' if locked is not None else ('detected' if box is not None else 'missing')
            if trial is not None:
                cv2.putText(display,'REC',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
                trial.save(raw,display,timestamp,aperture,area,a,thumb,index,source)
            fw = f'{aperture:.1f}' if aperture is not None else '未检测'
            dist = f'{distance:.1f}' if distance is not None else '无效'
            signed_text = f'{signed_distance:+.1f}px' if signed_distance is not None else '开始录制时确定方向'
            info.set(f'{color.get()} | 指间距 B={fw}px | 手物距离={dist}px | 有符号距离={signed_text} | 物体面积={area:.0f}px² | 中心={source} | Frames={trial.frame_count if trial else 0}')
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
