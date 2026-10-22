"""Live camera inference using three automatically selected five-frame distance MLPs; no IMU."""
import argparse
import csv
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if __name__ == '__main__':
    print('[Startup] Loading camera and MLP libraries...', flush=True)

import cv2
import numpy as np
import torch
from record_topview_black_cube import detect_cube, detect_fingers

from inference_window_mlp import load_model, predict, DistanceWindow, ExperimentTrial

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATHS = {
    'blue': r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TopView\TRIAL_Color_blue_Size_3cm_20261018_151544_535958\grasp_segments\Result\distance_mlp_window5_20261018_152555_247675\models\distance_window5.pth',
    'red': ROOT/'New/Data/2/Result/distance_mlp_window5_20261017_213924_764497/models/distance_window5.pth',
    'black': r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TopView\TRIAL_Color_black_Size_3cm_20261018_150704_338887\grasp_segments\Result\distance_mlp_window5_20261018_153227_711853\models\distance_window5.pth',
}
CONFIRM_FRAMES = 3
DETECTION_SETTINGS_PATH = ROOT/'New/Program/inference_three_targets_detection.json'


def detection_snapshot(bundles):
    return {color:{key:int(b['settings'].get(key,default)) for key,default in
                   [('v',60),('s',120),('area',500),('color_s_min',140),('color_v_min',50)]}
            for color,b in bundles.items()}


def apply_detection_settings(bundles, values):
    validated = {}
    for color,b in bundles.items():
        validated[color] = {}
        for key,value in values.get(color,{}).items():
            if key not in ('v','s','area','color_s_min','color_v_min'):
                continue
            number = int(value)
            if number != float(value) or not 0 <= number <= (100000 if key=='area' else 255):
                raise ValueError(f'Invalid detection setting: {color}.{key}={value}')
            validated[color][key] = number
    for color,settings in validated.items():
        bundles[color]['settings'].update(settings)


class DetectionControls:
    """Recorder-style Tk sliders, with independent colors and scrollable pages."""
    def __init__(self, initial, defaults):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title('三目标实时推理 — 检测参数')
        self.root.geometry(f'390x{min(620,self.root.winfo_screenheight()-100)}+20+20')
        self.closed = False
        self.root.protocol('WM_DELETE_WINDOW',self.request_close)
        tk.Label(self.root,text='选择下面的颜色页只用于调参，不会强制选择模型。',wraplength=360).pack(padx=8,pady=6)
        tk.Label(self.root,text='参数实时生效、自动保存；关闭本窗口会结束推理。',wraplength=360).pack(padx=8)
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both',expand=True,padx=8,pady=6)
        self.colors = list(initial)
        self.values = {}
        for color in self.colors:
            page = tk.Frame(self.tabs)
            self.tabs.add(page,text={'blue':'蓝色','red':'红色','black':'黑色'}[color])
            canvas = tk.Canvas(page,highlightthickness=0)
            scrollbar = ttk.Scrollbar(page,orient='vertical',command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
            content = tk.Frame(canvas)
            item = canvas.create_window((0,0),window=content,anchor='nw')
            content.bind('<Configure>',lambda event,c=canvas:c.configure(scrollregion=c.bbox('all')))
            canvas.bind('<Configure>',lambda event,c=canvas,i=item:c.itemconfigure(i,width=event.width))
            specs = [('黑色亮度上限 V','v'),('黑色饱和度上限 S','s')] if color=='black' else [
                ('颜色饱和度下限 S','color_s_min'),('颜色亮度下限 V','color_v_min')]
            self.values[color] = {k:tk.IntVar(value=v) for k,v in initial[color].items()}
            for label,key in specs + [('最小面积 px²','area')]:
                scale = tk.Scale(content,label=label,from_=0 if key!='area' else 10,
                    to=255 if key!='area' else max(3000,initial[color]['area']),
                    orient='horizontal',variable=self.values[color][key],length=310)
                scale.pack(fill='x',padx=8,pady=5)
            tk.Button(content,text='恢复该颜色的模型默认参数',
                command=lambda c=color:self.reset(c,defaults[c])).pack(pady=10)
            def bind_scroll(widget,c=canvas):
                widget.bind('<MouseWheel>',lambda event:c.yview_scroll(-int(event.delta/120),'units'))
                for child in widget.winfo_children():bind_scroll(child,c)
            bind_scroll(page)
        self.mask = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root,text='显示当前调参颜色的掩膜',variable=self.mask).pack(pady=5)

    def request_close(self):
        self.closed = True

    def reset(self,color,values):
        for key,value in values.items():
            self.values[color][key].set(value)

    def poll(self):
        self.root.update_idletasks()
        self.root.update()
        return not self.closed

    def snapshot(self):
        return {c:{k:v.get() for k,v in values.items()} for c,values in self.values.items()}

    def mask_color(self):
        return self.colors[self.tabs.index(self.tabs.select())] if self.mask.get() else None

    def close(self):
        self.root.destroy()


def load_models(paths):
    bundles = {}
    for color, path in paths.items():
        print(f'[Startup] Loading {color} model...', flush=True)
        p, model, settings, network = load_model(path)
        if settings.get('object_color', 'black') != color:
            raise ValueError(f'{color}: checkpoint object color does not match mapping.')
        bundles[color] = dict(path=p, model=model, settings=copy.deepcopy(settings), network=network)
    geometries = {(b['settings']['width'], b['settings']['height'], b['settings']['fps']) for b in bundles.values()}
    if len(geometries) != 1:
        raise ValueError('All three models must share image resolution and recording FPS.')
    return bundles


class PlacementROI:
    """Drag a box over live frames, without freezing capture in selectROI."""
    def __init__(self):
        self.start = self.end = None
        self.dragging = False

    def on_mouse(self, event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = self.end = (x,y)
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x,y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.end = (x,y)
            self.dragging = False

    def selection(self):
        if self.start is None or self.end is None or self.dragging:
            return None
        x,y = self.start;xx,yy = self.end
        return min(x,xx),min(y,yy),abs(xx-x),abs(yy-y)


class TargetSelector:
    def __init__(self, count=CONFIRM_FRAMES):
        self.required = count
        self.color = None
        self.count = 0

    def update(self, color):
        if color is None:
            self.color, self.count = None, 0
            return None
        self.count = self.count + 1 if self.color == color else 1
        self.color = color
        return color if self.count >= self.required else None


def validate_roi(roi, width, height):
    x,y,w,h = map(int, roi)
    if x < 0 or y < 0 or w < 5 or h < 5 or x+w > width or y+h > height:
        raise ValueError('ROI must lie inside the camera image, with width/height >= 5.')
    return x,y,w,h


def classify_target(raw, bundles, roi):
    """Classify by stored HSV thresholds inside a fixed placement ROI only.

    More than one plausible color is ambiguous; never fall back to black.
    The one-object protocol and ROI prevent dark sleeves/backgrounds and red
    reference marks elsewhere on the table from choosing a model.
    """
    x0,y0,rw,rh = roi
    crop = raw[y0:y0+rh, x0:x0+rw]
    candidates = {}
    combined_mask = np.zeros(raw.shape[:2],np.uint8)
    for color,bundle in bundles.items():
        s = bundle['settings']
        (area,box),mask = detect_cube(crop,s['v'],s['s'],s['area'],color=color,
            color_s_min=s.get('color_s_min',140),color_v_min=s.get('color_v_min',50))
        combined_mask[y0:y0+rh,x0:x0+rw] = cv2.bitwise_or(combined_mask[y0:y0+rh,x0:x0+rw],mask)
        if box is None:
            continue
        x,y,w,h = box
        # Reject slivers, cropped contours and large dark background regions.
        if min(w,h) < 5 or not .45 <= w/h <= 2.2 or area > .65*rw*rh:
            continue
        if x <= 0 or y <= 0 or x+w >= rw or y+h >= rh:
            continue
        candidates[color] = dict(area=float(area),box=(x+x0,y+y0,w,h))
    if len(candidates) != 1:
        return None, None, combined_mask, candidates
    color = next(iter(candidates))
    return color,candidates[color]['box'],combined_mask,candidates


def draw_candidates(image,candidates):
    """Retain all color candidates on screen, even when classification is ambiguous."""
    colors = {'blue':(255,100,0),'red':(0,0,255),'black':(50,50,50)}
    for i,(color,candidate) in enumerate(candidates.items()):
        x,y,w,h = candidate['box']
        ink = colors[color]
        cv2.rectangle(image,(x,y),(x+w,y+h),ink,2)
        label = f'{color.upper()} {w}x{h} box={candidate["area"]:.0f}px2'
        # Stack labels away from small overlapping boxes so all remain readable.
        yy = max(16,y-8)-i*19
        if yy < 16:yy = y+h+20+i*19
        (tw,th),_ = cv2.getTextSize(label,0,.45,1)
        xx = max(0,min(x,image.shape[1]-tw-4))
        cv2.rectangle(image,(xx,yy-th-3),(xx+tw+3,yy+3),(255,255,255),-1)
        cv2.putText(image,label,(xx,yy),0,.45,ink,1,cv2.LINE_AA)


def summarize_trials(out):
    """Summarize recorded trials only; preview frames never enter experiment metrics."""
    rows = []
    for folder in sorted(out.glob('trial_*')):
        if not (folder/'summary.json').exists():
            continue
        with (folder/'predictions.csv').open(encoding='utf-8-sig') as f:
            frames = list(csv.DictReader(f))
        for color in MODEL_PATHS:
            selected = [r for r in frames if r['selected_model_color']==color]
            valid = [r for r in selected if r['valid']=='1']
            if not selected:continue
            y = np.array([float(r['measured_aperture_px']) for r in valid])
            errors = np.array([float(r['error_px']) for r in valid])
            denominator = float(np.sum((y-y.mean())**2)) if len(y) else 0
            rows.append(dict(trial=folder.name,color=color,
                model_path=selected[0]['model_path'],total_frames=len(selected),
                predicted_frames=len(valid),prediction_coverage=len(valid)/len(selected),
                mae_px=float(np.mean(abs(errors))) if len(valid) else None,
                rmse_px=float(np.sqrt(np.mean(errors**2))) if len(valid) else None,
                r2=float(1-np.sum(errors**2)/denominator) if denominator>1e-12 else None))
    if not rows:
        return
    with (out/'trial_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(13,4.5))
    for ax,key,title in zip(axes,['mae_px','rmse_px','r2'],['MAE (px)','RMSE (px)','R2']):
        colors,means = [],[]
        for color in MODEL_PATHS:
            values = [r[key] for r in rows if r['color']==color and r[key] is not None]
            if values:
                colors.append(color);means.append(float(np.mean(values)))
        bars = ax.bar(colors,means,color=colors)
        ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Recorded sessions | metrics grouped by selected target (not individual grasps)')
    fig.tight_layout();fig.savefig(out/'target_metrics.png',dpi=170);plt.close(fig)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    for color,path in MODEL_PATHS.items():
        parser.add_argument('--'+color+'-model', default=str(path))
    parser.add_argument('--roi', nargs=4, type=int, metavar=('X','Y','W','H'), help='Fixed object placement region in raw image pixels')
    parser.add_argument('--detection-settings',type=Path,default=DETECTION_SETTINGS_PATH,help='Saved adjustable detection parameters')
    parser.add_argument('--no-controls',action='store_true',help='Disable slider window')
    parser.add_argument('--camera', type=int, help='Default: recording camera index')
    parser.add_argument('--output', type=Path, default=ROOT/'Data/TopView/Inference')
    parser.add_argument('--video', type=Path, help='Replay a recorded video instead of a camera')
    parser.add_argument('--no-preview', action='store_true', help='Replay/verification without opening a window')
    parser.add_argument('--max-frames', type=int, default=0, help='0: unlimited')
    parser.add_argument('--auto-record',action='store_true',help='Start a trial automatically (also useful for video replay)')
    args = parser.parse_args()
    try:
        bundles = load_models({c:getattr(args,c+'_model') for c in MODEL_PATHS})
        detection_defaults = detection_snapshot(bundles)
        if args.detection_settings.exists():
            apply_detection_settings(bundles,json.loads(args.detection_settings.read_text(encoding='utf-8-sig')))
        settings = bundles['blue']['settings']
        torch.set_num_threads(min(4, torch.get_num_threads()))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.no_preview and args.roi is None:
        parser.error('--no-preview requires --roi X Y W H.')
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise SystemExit('Run in the gripper environment used for recording (mp.solutions.hands required).')
    camera = settings.get('camera', 1) if args.camera is None else args.camera
    print(f'[Startup] Opening {args.video if args.video else "camera " + str(camera)}...', flush=True)
    cap = cv2.VideoCapture(str(args.video)) if args.video else cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise SystemExit('Cannot open camera/video. Try --camera 0 or check --video path.')
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
        cap.set(cv2.CAP_PROP_FPS, settings['fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print('[Startup] Camera/video opened. Initializing hand tracking...', flush=True)
    out = args.output/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    config = dict(models={c:dict(model_path=str(b['path']),model=b['model'],recording_settings=b['settings']) for c,b in bundles.items()},
                  initial_detection_settings=detection_snapshot(bundles),
                  camera=camera, video=str(args.video) if args.video else None,
                  input='five consecutive distances, oldest to current; checkpoint normalization', output='finger_width in pixels',
                  confirmation_frames=CONFIRM_FRAMES,
                  error_definition='predicted minus measured', target='HSV color classification inside fixed placement ROI; one object at a time; ambiguous colors suppress prediction')
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for color,b in bundles.items():
        print(f'{color}: {b["path"]}')
    print(f'Expected image: {settings["width"]} x {settings["height"]}; keep camera pose and zoom identical to recording.')
    print('N: start trial | S: stop and save trial | L: lock/unlock center | R: placement ROI | M: mask | Q: save and quit')
    print('Results:', out)
    print('Five-frame causal window; missing detections/gaps reset it. Match recording FPS:', settings['fps'])
    window = DistanceWindow(.1)
    selector = TargetSelector()
    active_color = None
    roi = tuple(args.roi) if args.roi is not None else None
    locked_color = None
    locked = None
    print('Select a small ROI around the fixed object position; exclude sleeves and colored reference marks.')
    show_mask = False
    errors = []
    frames = valid_frames = outside_frames = 0
    started = time.perf_counter()
    hands = None
    trial = None
    trial_number = 0
    auto_record = args.auto_record
    roi_editor = PlacementROI()
    window_name = 'Live five-frame MLP - automatic blue/red/black'
    first_frame = True
    controls = None
    detection_values = detection_snapshot(bundles)
    parameter_revision = 0
    parameter_events = [dict(revision=0,frame=0,elapsed_s=0,settings=copy.deepcopy(detection_values))]
    (out/'detection_parameter_events.json').write_text(json.dumps(parameter_events,indent=2),encoding='utf-8')
    try:
        if not args.no_preview and not args.no_controls:
            controls = DetectionControls(detection_values,detection_defaults)
        hands = mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,
                                        min_detection_confidence=.6,min_tracking_confidence=.6)
        print('[Startup] Waiting for first camera/video frame...', flush=True)
        with (out/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
            columns = ['frame','unix_time_s','elapsed_s','video_time_s','object_x','object_y','hand_center_x','hand_center_y',
                       'distance_px','normalized_distance','measured_aperture_px','predicted_aperture_px','error_px',
                       'valid','outside_training_range','object_center_source','detection_valid',
                       'window_count','window_start_frame','window_duration_s','window_distances_px','status','inference_ms',
                       'detected_color','selected_model_color','trial_target_color','candidate_colors','model_path','parameter_revision','candidate_details']
            writer = csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            while True:
                if controls is not None:
                    if not controls.poll():
                        break
                    updated = controls.snapshot()
                    if updated != detection_values:
                        apply_detection_settings(bundles,updated)
                        detection_values = copy.deepcopy(updated)
                        parameter_revision += 1
                        window.clear()
                        selector = TargetSelector()
                        active_color = locked_color = locked = None
                        parameter_events.append(dict(revision=parameter_revision,frame=frames,
                            elapsed_s=time.perf_counter()-started,settings=copy.deepcopy(updated)))
                        (out/'detection_parameter_events.json').write_text(json.dumps(parameter_events,indent=2),encoding='utf-8')
                        args.detection_settings.parent.mkdir(parents=True,exist_ok=True)
                        args.detection_settings.write_text(json.dumps(updated,indent=2),encoding='utf-8')
                ok, raw = cap.read()
                if not ok:
                    if args.video:break
                    raise RuntimeError('Camera frame acquisition failed')
                h,w = raw.shape[:2]
                if first_frame:
                    print('[Startup] First frame received. ' + ('Live preview ready; drag placement box and press Enter.' if roi is None else 'Using configured ROI; starting detection.'), flush=True)
                    first_frame = False
                if (w,h) != (settings['width'],settings['height']):
                    raise RuntimeError(f'Image resolution {w}x{h} differs from training {settings["width"]}x{settings["height"]}. Pixel distances would not match.')
                timestamp = time.time();elapsed = time.perf_counter()-started
                sample_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if args.video else elapsed
                if roi is None:
                    preview = raw.copy()
                    if roi_editor.start is not None and roi_editor.end is not None:
                        cv2.rectangle(preview,roi_editor.start,roi_editor.end,(255,180,0),2)
                    cv2.putText(preview,'LIVE: drag object placement box, ENTER to confirm',(10,25),0,.55,(0,180,0),2)
                    cv2.putText(preview,'Q / ESC: quit | image remains live while selecting',(10,50),0,.5,(0,180,0),1)
                    cv2.imshow(window_name,preview)
                    cv2.setMouseCallback(window_name,roi_editor.on_mouse)
                    key = cv2.waitKey(1)&255
                    if key in (ord('q'),27):
                        break
                    if key in (10,13,32) and roi_editor.selection() is not None:
                        try:
                            roi = validate_roi(roi_editor.selection(),w,h)
                        except ValueError as exc:
                            print(str(exc),flush=True)
                            continue
                        config['placement_roi'] = roi
                        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
                        window.clear()
                        print(f'[Ready] Placement ROI: {roi}. N to record.',flush=True)
                    continue
                roi = validate_roi(roi,w,h)
                config['placement_roi'] = roi
                detected_color,box,mask,candidates = classify_target(raw,bundles,roi)
                confirmed_color = selector.update(detected_color)
                if confirmed_color != active_color:
                    window.clear()
                    if confirmed_color != locked_color:
                        locked = None
                        locked_color = None
                    active_color = confirmed_color
                bundle = bundles.get(active_color)
                if bundle is not None:
                    model,network = bundle['model'],bundle['network']
                    window.max_gap = model['max_frame_gap_s']
                thumb,index,landmarks = detect_fingers(raw,hands)
                a = locked.copy() if locked is not None and active_color == locked_color else None
                if a is None and box is not None:
                    x,y,bw,bh = box;a = np.array([x+bw/2,y+bh/2])
                c = (thumb+index)/2 if thumb is not None else None
                measured = float(np.linalg.norm(thumb-index)) if c is not None else None
                valid = a is not None and c is not None and active_color is not None
                if auto_record:
                    window.clear()
                source = 'locked' if locked is not None else ('detected' if box is not None else 'missing')
                row = dict.fromkeys(columns,'')
                row.update(frame=frames,unix_time_s=timestamp,elapsed_s=elapsed,
                           video_time_s=cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if args.video else '',
                           valid=0,detection_valid=int(valid),object_center_source=source,
                           detected_color=detected_color or '',selected_model_color=active_color or '',
                           parameter_revision=parameter_revision,
                           trial_target_color=active_color or '',candidate_colors=json.dumps(list(candidates)),
                           candidate_details=json.dumps(candidates),
                           model_path=str(bundle['path']) if bundle else '')
                if a is not None:row.update(object_x=a[0],object_y=a[1])
                if c is not None:row.update(hand_center_x=c[0],hand_center_y=c[1],measured_aperture_px=measured)
                target_line = f'Target: {active_color or "unconfirmed"} | Continuous session'
                lines = [target_line, 'CONFLICT: '+', '.join(candidates) if len(candidates)>1 else 'Waiting for confirmed target and hand...']
                distance = float(np.linalg.norm(a-c)) if valid else None
                distances = window.push(distance, sample_time, frames)
                row.update(window_count=len(window.values), status='ambiguous_target' if len(candidates)>1 else 'missing_detection' if not valid else 'warming_up')
                if valid:
                    row.update(distance_px=distance, normalized_distance=(distance-model['x_min'])/model['scale'])
                    lines = [f'{active_color.upper()} | Distance: {distance:.1f} px | Center: {source}',
                             f'Measured B: {measured:.1f} px',f'Warming up: {len(window.values)}/5 frames']
                if distances is not None:
                    infer_started = time.perf_counter()
                    estimate = predict(network,model,distances)
                    inference_ms = (time.perf_counter()-infer_started)*1000
                    outside = any(not model['x_min'] <= d <= model['x_max'] for d in distances)
                    error = estimate-measured
                    row.update(valid=1,status='ready', predicted_aperture_px=estimate,error_px=error,
                               outside_training_range=int(outside),inference_ms=inference_ms,
                               window_start_frame=window.values[0][2],
                               window_duration_s=window.values[-1][1]-window.values[0][1],
                               window_distances_px=json.dumps(distances))
                    valid_frames += 1;outside_frames += int(outside)
                    if not outside:errors.append(error)
                    lines = [f'{active_color.upper()} | Distance: {distance:.1f} px | Window: 5/5',
                             f'Measured B: {measured:.1f} px',f'Predicted B: {estimate:.1f} px | Window span: {row["window_duration_s"]:.3f}s',
                             f'Error: {error:+.1f} px | MLP: {inference_ms:.2f} ms',
                             'OUTSIDE TRAINING RANGE (any window input)' if outside else 'Within training distance range']
                writer.writerow(row);frames += 1
                if frames%30 == 0:f.flush()
                view = raw.copy()
                rx,ry,rw,rh = roi
                cv2.rectangle(view,(rx,ry),(rx+rw,ry+rh),(255,180,0),1)
                if landmarks is not None:
                    mp.solutions.drawing_utils.draw_landmarks(view,landmarks,mp.solutions.hands.HAND_CONNECTIONS)
                draw_candidates(view,candidates)
                if a is not None:cv2.circle(view,tuple(a.astype(int)),5,(0,0,255),-1)
                if valid:cv2.line(view,tuple(a.astype(int)),tuple(c.astype(int)),(0,255,255),2)
                panel = np.full((h+170,max(w,760),3),245,np.uint8);panel[:h,:w]=view
                for i,line in enumerate(lines):cv2.putText(panel,line,(12,h+25+i*25),0,.55,(30,30,30),1,cv2.LINE_AA)
                cv2.putText(panel,'N: record | S: stop | L: lock | R: ROI | M: mask | Q: quit',(12,h+155),0,.5,(80,80,80),1)
                if auto_record:
                    trial_number+=1
                    trial_config=dict(config,target_color='continuous_multi_target',model_path='See per-frame model_path in predictions.csv',object_center_source=source,
                        recording_mode='One continuous session; multiple object changes and grasps; no automatic grasp counting.',
                        parameter_revision=parameter_revision,detection_settings=copy.deepcopy(detection_values),
                        locked_center=locked.tolist() if locked is not None else None)
                    trial=ExperimentTrial(out,trial_number,(w,h),(panel.shape[1],panel.shape[0]),settings['fps'],columns,trial_config)
                    auto_record=False
                    print(f'[REC] session trial_{trial_number:03d} | continuous blue/red/black',flush=True)
                if trial is not None:
                    cv2.putText(panel,f'REC {trial_number:03d} | frame {len(trial.rows)}',(12,25),0,.65,(0,0,255),2)
                    trial.save(raw,panel,row)
                if not args.no_preview:
                    preview=panel.copy()
                    if show_mask:preview[:h,:w]=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
                    mask_color = controls.mask_color() if controls is not None else None
                    if mask_color is not None:
                        s = bundles[mask_color]['settings']
                        _,tuning_mask = detect_cube(raw[ry:ry+rh,rx:rx+rw],s['v'],s['s'],s['area'],
                            color=mask_color,color_s_min=s.get('color_s_min',140),color_v_min=s.get('color_v_min',50))
                        preview[:h,:w] = 0
                        preview[ry:ry+rh,rx:rx+rw] = cv2.cvtColor(tuning_mask,cv2.COLOR_GRAY2BGR)
                    if show_mask or mask_color is not None:
                        draw_candidates(preview,candidates)
                    cv2.imshow(window_name,preview)
                    key = cv2.waitKey(1)&255
                    if key in (ord('q'),27):break
                    if key == ord('m'):show_mask = not show_mask
                    if key == ord('n') and trial is None:
                        window.clear()
                        auto_record=True
                    if key == ord('s') and trial is not None:
                        finished=trial;trial=None;finished.close()
                        window.clear()
                    if key == ord('l') and trial is None and active_color is not None:
                        window.clear()
                        if locked is not None:
                            locked = None
                            locked_color = None
                        elif box is not None:
                            x,y,bw,bh=box;locked=np.array([x+bw/2,y+bh/2])
                            locked_color = active_color
                    if key == ord('r') and trial is None:
                        roi = None
                        roi_editor = PlacementROI()
                        window.clear()
                        selector = TargetSelector()
                        locked = None
                        locked_color = None
                        active_color = None
                if args.max_frames>0 and frames>=args.max_frames:break
    finally:
        cap.release()
        if controls is not None:controls.close()
        if hands is not None:hands.close()
        if not args.no_preview:cv2.destroyAllWindows()
        if trial is not None:trial.close()
        summary = dict(frames=frames,valid_frames=valid_frames,outside_training_range_frames=outside_frames,
            in_range_frames=len(errors),in_range_mae_px=float(np.mean(np.abs(errors))) if errors else None,
            in_range_rmse_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            note='Live descriptive errors, not an independent held-out evaluation.')
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        summarize_trials(out)
        print(f'Saved: {out} | frames={frames}, valid={valid_frames}, outside range={outside_frames}')

if __name__ == '__main__':main()
