"""Live comparison of the three joint size-conditioned models."""
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
OUTPUT = ROOT / 'New/Result/RealTime_Joint_Three_Models'
DEFAULT_MODELS = ROOT / 'Data/1101_D/Result/Joint_Combined_Function_MLP5MC_LSTMFull_20261104_133406'
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
        self.net = nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Dropout(.1),
                                 nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self, x):
        return self.net(x).reshape(-1)

class JointLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 32, batch_first=True)
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
        self.set_size(args.size_cm)
        self.low, self.span = fn['train_x_min'], fn['train_x_range']
        self.mlp, self.lstm = JointMLP(), JointLSTM()
        self.scales = []
        for model, name in [(self.mlp,'mlp5.pth'),(self.lstm,'lstm_full.pth')]:
            ck = torch.load(args.models/name,map_location='cpu',weights_only=True)
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
        base, slope = np.asarray(spec['coefficients']).reshape(2, -1)
        self.function = BSpline(spec['knots'], base + (size - 3) * slope, spec['degree'])
        self.reset()

    def reset(self,clear_initial=True):
        self.history = deque(maxlen=5)
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
        x = self.normalized_distance = distance/self.initial_distance
        self.previous = timestamp,frame
        self.history.append(x)
        start = time.perf_counter()
        function = float(self.function((x-self.low)/self.span))
        self.last_timing['function_ms'] = (time.perf_counter()-start)*1000
        mlp = np.nan
        if len(self.history)==5:
            start = time.perf_counter()
            features = torch.tensor([*self.history,self.size],dtype=torch.float32)
            result = self.mlp(features.expand(self.mlp_samples,-1))
            mean,std = self.scales[0]
            mlp = torch.quantile(result,.5).item()*std+mean
            self.last_timing['mlp_5f_ms'] = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        hidden,self.state = self.lstm.lstm(torch.tensor([[[x,self.size]]],dtype=torch.float32),self.state)
        result = self.lstm.head(hidden[0].expand(self.lstm_samples,-1)).flatten()
        mean,std = self.scales[1]
        lstm = torch.quantile(result,.5).item()*std+mean
        self.last_timing['lstm_full_ms'] = (time.perf_counter()-start)*1000
        return dict(zip(KEYS,(function,mlp,lstm))),len(self.history)==5,gap


def check_models(engine):
    distances = np.linspace(320,45,30)
    for i,distance in enumerate(distances):
        values,common,_ = engine.step(distance,100+i/30,i)
        assert common == (i>=4)
        assert all(np.isfinite([values[KEYS[0]],values[KEYS[2]]]))
        if common:
            assert np.isfinite(values[KEYS[1]])
    with torch.inference_mode():
        x=torch.tensor(np.column_stack((distances/distances[0],np.full(30,engine.size))),
                       dtype=torch.float32)[None]
        _,whole_state=engine.lstm.lstm(x)
        for streamed,whole in zip(engine.state,whole_state):
            torch.testing.assert_close(streamed,whole,atol=1e-6,rtol=1e-5)
    _,common,gap=engine.step(50.,102.,30)
    assert gap and not common and engine.initial_distance==320
    engine.step(None,103.,31)
    assert engine.state is None and not engine.history
    engine.reset()
    assert engine.initial_distance is None
    for size in (2, 3, 4):
        engine.set_size(size)
        assert engine.initial_distance is None and engine.state is None
        prediction, _, _ = engine.step(320., 100., 0)
        assert np.isfinite(prediction[KEYS[0]]) and np.isfinite(prediction[KEYS[2]])
    print('PASS: model loading, causal streaming, warmup, dropout and size resets.')


def write_csv(path, rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def score_rows(rows):
    common=[r for r in rows if r['common_valid'] and all(np.isfinite(r[k]) for k in ['finger_width_mm',*KEYS])]
    result=[]
    for key,label in zip(KEYS,LABELS):
        if not common:continue
        y=np.array([r['finger_width_mm'] for r in common]);p=np.array([r[key] for r in common]);e=p-y
        denominator=np.sum((y-y.mean())**2)
        result.append(dict(model=label,frames=len(y),mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),
                           r2=float(1-(e*e).sum()/denominator) if denominator>1e-12 else None))
    return result

def plot_scores(path,scores):
    if not scores:return
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (mm)','RMSE (mm)','R²']):
        bars=ax.bar(LABELS,[r[key] if r[key] is not None else np.nan for r in scores],color=COLORS)
        ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3)
        ax.tick_params(axis='x',labelsize=7);ax.set_title(title);ax.margins(y=.25);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(Path(path).with_suffix('.'+ext),dpi=180)
    plt.close(fig)

class LiveTrial:
    def __init__(self,session,frame_shape,fps,metadata):
        self.path=session/f'trial_{datetime.now():%Y%m%d_%H%M%S_%f}';self.path.mkdir()
        self.rows=[];self.start_time=None;self.closed=False
        self.metadata=metadata
        self.file=(self.path/'predictions.csv').open('w',newline='',encoding='utf-8-sig');self.writer=None
        h,w=frame_shape[:2]
        self.raw_video=cv2.VideoWriter(str(self.path/'raw_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        self.video=cv2.VideoWriter(str(self.path/'annotated_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not self.video.isOpened() or not self.raw_video.isOpened():
            self.video.release();self.raw_video.release();self.file.close();raise RuntimeError('Cannot create trial videos.')
        (self.path/'settings.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

    def append(self,row,raw,display):
        if self.start_time is None:self.start_time=row['t_camera']
        row=dict(frame=len(self.rows),elapsed_s=row['t_camera']-self.start_time,**row)
        if self.writer is None:
            self.writer=csv.DictWriter(self.file,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row);self.file.flush();self.rows.append(row)
        self.raw_video.write(raw);self.video.write(display)

    def close(self):
        if self.closed:return
        self.closed=True;self.file.close();self.raw_video.release();self.video.release()
        scores=score_rows(self.rows);write_csv(self.path/'metrics_common_frames.csv',scores)
        plot_scores(self.path/'metrics_comparison',scores)
        dt=np.diff([r['t_camera'] for r in self.rows])
        diagnostics=dict(saved_frames=len(self.rows),valid_frames=sum(r['valid'] for r in self.rows),
            common_frames=sum(r['common_valid'] for r in self.rows),input_gap_resets=sum(r['reset_on_gap'] for r in self.rows),
            observed_fps=float(1/np.mean(dt)) if len(dt) and np.mean(dt)>0 else None,
            mean_inference_ms=float(np.mean([r['inference_ms'] for r in self.rows])) if self.rows else None,
            p95_inference_ms=float(np.percentile([r['inference_ms'] for r in self.rows],95)) if self.rows else None,
            mean_detection_ms=float(np.mean([r['detection_ms'] for r in self.rows])) if self.rows else None,
            timing_scope='processing_before_save_ms excludes CSV/video writing, chart and display; observed_fps includes frame loop and pacing')
        (self.path/'timing_and_coverage.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
        fig,ax=plt.subplots(figsize=(11,4.5))
        if self.rows:
            t=[r['elapsed_s'] for r in self.rows]
            measured=[r['finger_width_mm'] if r['valid'] else np.nan for r in self.rows]
            ax.plot(t,measured,color='#222222',lw=2,label='Measured')
            for key,label,color in zip(KEYS,LABELS,COLORS):ax.plot(t,[r[key] for r in self.rows],color=color,label=label,lw=1.4)
            ax.legend()
        ax.set(xlabel='Time from trial start (s)',ylabel='Aperture (mm)');ax.grid(alpha=.2);fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(self.path/f'prediction_comparison.{ext}',dpi=180)
        plt.close(fig)
        return scores

def chart(rows):
    image=np.full((300,720,3),255,np.uint8)
    cv2.putText(image,'Measured / Function / MLP / LSTM',(14,22),0,.55,(40,40,40),1,cv2.LINE_AA)
    if len(rows)<2:return image
    times=np.array([r['elapsed_s'] for r in rows]);last=max(float(times[-1]),1.)
    data=[np.array([r['finger_width_mm'] if r['valid'] else np.nan for r in rows])]+[np.array([r[k] for r in rows]) for k in KEYS]
    finite=np.concatenate(data);finite=finite[np.isfinite(finite)]
    upper=max(110.,float(finite.max())+5) if len(finite) else 110.
    lower=min(0.,float(finite.min())-5) if len(finite) else 0.
    for values,color in zip(data,[(30,30,30),(184,133,52),(50,154,229),(88,158,81)]):
        for i in range(1,len(times)):
            if not np.isfinite(values[i-1:i+1]).all():continue
            a=(int(45+times[i-1]/last*650),int(260-(values[i-1]-lower)/(upper-lower)*210))
            b=(int(45+times[i]/last*650),int(260-(values[i]-lower)/(upper-lower)*210))
            cv2.line(image,a,b,color,1,cv2.LINE_AA)
    cv2.line(image,(45,50),(45,260),(120,120,120),1);cv2.line(image,(45,260),(695,260),(120,120,120),1)
    cv2.putText(image,f'0 - {last:.1f} s | Aperture {lower:.0f}..{upper:.0f} mm',(45,286),0,.5,(50,50,50),1)
    return image

def size_settings(size):
    dataset = ROOT / 'Data/1101_D' / f'{size*10}mm'
    candidates = sorted(p for p in dataset.glob('*/settings.json')
                        if p.parent.name.startswith('grasp_') or p.parent.name[:8].isdigit())
    if not candidates:
        raise ValueError(f'No detection settings for {size*10} mm')
    path = candidates[0]
    return path, json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,default=DEFAULT_MODELS)
    parser.add_argument('--size-cm',type=int,choices=(2,3,4),default=3)
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--mm-per-px',type=float)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--camera',type=int,default=1)
    parser.add_argument('--fps',type=float,default=30.)
    parser.add_argument('--max-object-shift',type=float,default=40.)
    parser.add_argument('--mlp-samples',type=int,default=128)
    parser.add_argument('--lstm-samples',type=int,default=1,help='1 uses normal inference; >1 uses output-layer MC median')
    parser.add_argument('--check-models',action='store_true')
    args=parser.parse_args()
    if min(args.fps,args.max_object_shift,args.mlp_samples,args.lstm_samples)<=0:
        parser.error('FPS, shift limit and sample counts must be positive')
    torch.set_num_threads(1)
    engine=ComparisonEngine(args)
    if args.check_models:
        check_models(engine)
        return
    presets = {size: size_settings(size) for size in (2, 3, 4)}
    engine.settings_file, engine.settings = presets[args.size_cm]
    if args.settings:
        engine.settings_file = args.settings
        engine.settings = json.loads(args.settings.read_text(encoding='utf-8-sig'))
        presets[args.size_cm] = (args.settings, engine.settings)
    calibration=engine.settings.get('pixel_calibration') or {}
    engine.mm_per_px=args.mm_per_px if args.mm_per_px is not None else float(calibration.get('mm_per_px',np.nan))
    if not np.isfinite(engine.mm_per_px) or engine.mm_per_px<=0:
        parser.error('Provide valid mm/px calibration')
    engine.settings['pixel_calibration']=dict(calibration,mm_per_px=engine.mm_per_px)
    LABELS[1]='MLP 5f (MC median)' if args.mlp_samples>1 else 'MLP 5f'
    LABELS[2]='LSTM Full (MC median)' if args.lstm_samples>1 else 'LSTM Full'
    import tkinter as tk
    from tkinter import messagebox,ttk
    import mediapipe as mp
    settings=engine.settings
    margins=tuple(int(settings.get('crop_margins_px',{}).get(k,0)) for k in ('left','top','right','bottom'))
    expected=(int(settings['width']),int(settings['height']))
    color=settings.get('object_color','black')
    session=args.output/f'three_model_live_{datetime.now():%Y%m%d_%H%M%S_%f}'
    session.mkdir(parents=True,exist_ok=False)
    (session/'model_sources.json').write_text(json.dumps(dict(models=str(args.models.resolve()),initial_size_cm=args.size_cm,mlp_samples=args.mlp_samples,lstm_samples=args.lstm_samples,files=['function.json','mlp5.pth','lstm_full.pth'],distance='index_mcp_object_distance_mm / trial initial distance',settings_source=str(engine.settings_file),mm_per_px=engine.mm_per_px,common_frame_rule='Five consecutive valid inputs after every reset; LSTM observes all valid inputs, including warmup',units='mm',video_timing='Nominal FPS video; use CSV timestamps for actual time'),indent=2),encoding='utf-8')
    cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW)
    if not cap.isOpened():raise RuntimeError('Cannot open camera.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(settings.get('camera_source_width',expected[0]+margins[0]+margins[2])))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(settings.get('camera_source_height',expected[1]+margins[1]+margins[3])))
    cap.set(cv2.CAP_PROP_FPS,args.fps)
    hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=.5,min_tracking_confidence=.5)
    ui=tk.Tk();ui.title('三模型实时对比');ui.geometry('440x760')
    status=tk.StringVar(value='手放回起点，确认物体和手部检测，再按 N 开始。')
    size_label=tk.StringVar(value=f'{engine.size*10} mm | {color}')
    tk.Label(ui,textvariable=size_label,font=('',13)).pack(pady=6)
    tk.Label(ui,text=f'Calibration: {engine.mm_per_px:.6g} mm/px',wraplength=395).pack()
    tk.Label(ui,textvariable=status,wraplength=395,justify='left').pack(fill='x',padx=8)
    target=tk.StringVar(value='1')
    row=tk.Frame(ui);row.pack()
    for text in ('1','2','3'):tk.Radiobutton(row,text=text+' 秒条件',variable=target,value=text).pack(side='left')
    values={key:tk.IntVar(value=int(settings.get(key,default))) for key,default in [('v',60),('s',120),('area',500),('max_area',640000),('color_s_min',140),('color_v_min',50)]}
    controls=[]
    for label,key in [('黑色亮度上限','v'),('黑色饱和度上限','s'),('颜色饱和度下限','color_s_min'),('颜色亮度下限','color_v_min'),('最小面积','area')]:
        control=tk.Scale(ui,label=label,variable=values[key],from_=0,to=3000 if key=='area' else 255,orient='horizontal',length=350);control.pack();controls.append(control)
    row=tk.Frame(ui);row.pack();tk.Label(row,text='最大物体面积 px²').pack(side='left')
    entry=tk.Entry(row,textvariable=values['max_area'],width=12);entry.pack(side='left');controls.append(entry)
    mask_option=tk.BooleanVar(value=False);tk.Checkbutton(ui,text='显示检测掩膜',variable=mask_option).pack()
    trial=None;axis=None;origin=None;locked=None;latest=None;closing=False;completed=[];last_plot=np.full((300,720,3),255,np.uint8)
    detection={k:v.get() for k,v in values.items()}

    selected_size = tk.IntVar(value=engine.size)

    def switch_size():
        nonlocal color, locked, latest, detection
        if trial is not None:
            selected_size.set(engine.size)
            return
        size = selected_size.get()
        path, preset = presets[size]
        engine.set_size(size)
        color = preset.get('object_color', 'black')
        for key, var in values.items():
            if key in preset:
                var.set(int(preset[key]))
        detection = {key: var.get() for key, var in values.items()}
        locked = latest = None
        size_label.set(f'{size*10} mm | {color}')
        status.set('已切换尺寸。检测到手和方块后，按 N 开始。')

    size_row = tk.Frame(ui)
    size_row.pack(after=ui.winfo_children()[0], pady=4)
    for size in (2, 3, 4):
        button = tk.Radiobutton(size_row, text=f'{size*10} mm', value=size,
                                variable=selected_size, command=switch_size)
        button.pack(side='left', padx=5)
        controls.append(button)

    def stop():
        nonlocal trial,axis,origin,last_plot
        if trial is not None:
            finished=trial;trial=None
            scores=finished.close();last_plot=chart(finished.rows)
            for r in scores:completed.append(dict(trial=finished.path.name,target_s=finished.metadata['target_movement_s'],**r))
            write_csv(session/'trial_metrics.csv',completed)
            all_rows=[]
            for path in session.glob('trial_*/predictions.csv'):
                condition=json.loads((path.parent/'settings.json').read_text(encoding='utf-8'))['target_movement_s']
                with path.open(encoding='utf-8-sig') as f:
                    for r in csv.DictReader(f):
                        all_rows.append(dict(target_s=condition,common_valid=r['common_valid']=='True',**{k:float(r[k]) for k in ['finger_width_mm',*KEYS]}))
            overall=score_rows(all_rows)
            write_csv(session/'overall_metrics.csv',overall);plot_scores(session/'overall_comparison',overall)
            by_condition=[]
            for condition in (1,2,3):
                by_condition.extend(dict(target_s=condition,**score) for score in score_rows([r for r in all_rows if r['target_s']==condition]))
            write_csv(session/'condition_metrics.csv',by_condition)
            status.set(f'已保存：{finished.path.name}；回位后按 N 开始下一次。')
        engine.reset();axis=None;origin=None
        for widget in controls:widget.config(state='normal')

    def start():
        nonlocal trial,axis,origin
        if trial is not None:return
        if latest is None or not latest['valid'] or time.time()-latest['time']>.3:
            status.set('需要同时检测到手和物体。');return
        try:
            axis=None;origin=latest['a'].copy();engine.reset()
            metadata=dict(size_cm=engine.size,object_color=color,detection_settings_source=str(presets[engine.size][0]),mlp_samples=args.mlp_samples,lstm_samples=args.lstm_samples,recording_settings=settings,detection=detection.copy(),crop_margins_px=margins,
                model_directory=str(args.models.resolve()),target_movement_s=int(target.get()),
                distance_definition='Euclidean MediaPipe landmark 5 to object centre',normalization='distance / first recorded valid distance',pixel_calibration=settings['pixel_calibration'],initial_object_center_xy=origin.tolist(),object_center_locked=locked is not None,
                common_frame_rule='Five consecutive valid frames after trial/gap reset',max_frame_gap_s=engine.gap,
                object_jump_reject_px=args.max_object_shift,units='mm')
            trial=LiveTrial(session,latest['shape'],args.fps,metadata)
            for widget in controls:widget.config(state='disabled')
            status.set('录制预测中；自然完成一次抓取后按 S 结束。')
        except Exception as exc:messagebox.showerror('无法开始',str(exc))

    def lock():
        nonlocal locked
        if trial is not None:return
        if locked is not None:locked=None;status.set('已解除物体中心锁定。')
        elif latest is not None and latest['valid']:locked=latest['a'].copy();status.set('物体中心已锁定；请保持物体不动。')

    def close():
        nonlocal closing
        closing=True
    for label,fn in [('开始一次 (N)',start),('结束并保存 (S)',stop),('锁定/解除物体中心 (L)',lock),('退出 (Q)',close)]:tk.Button(ui,text=label,command=fn).pack(fill='x',padx=20,pady=3)
    ui.protocol('WM_DELETE_WINDOW',close)
    def shortcut(event):
        if isinstance(ui.focus_get(),(tk.Entry,ttk.Entry)):return
        fn={'n':start,'s':stop,'l':lock,'q':close}.get(event.char.lower())
        if fn:fn()
    ui.bind('<Key>',shortcut)
    last_chart=0.;frame_id=0
    try:
        while not closing:
            began=time.perf_counter();ui.update_idletasks();ui.update()
            if closing:break
            capture_start=time.perf_counter();ok,raw=cap.read();timestamp=time.time();capture_ms=(time.perf_counter()-capture_start)*1000
            if not ok:raise RuntimeError('Camera read failed.')
            raw=crop_camera_frame(raw,margins)
            if (raw.shape[1],raw.shape[0])!=expected:raise RuntimeError(f'Camera cropped size {raw.shape[1]}x{raw.shape[0]} differs from model {expected}. No automatic resizing is applied.')
            if trial is None:
                try:
                    candidate={k:v.get() for k,v in values.items()}
                    if candidate['max_area']>0 and candidate['area']>=0 and all(0<=candidate[k]<=255 for k in ('v','s','color_s_min','color_v_min')):detection=candidate
                except (tk.TclError,ValueError):pass
            detection_start=time.perf_counter();thumb,index,landmarks=detect_fingers(raw,hands)
            (area,box),mask=detect_cube(raw,detection['v'],detection['s'],detection['area'],color=color,color_s_min=detection['color_s_min'],color_v_min=detection['color_v_min'],max_area=detection['max_area'])
            detection_ms=(time.perf_counter()-detection_start)*1000
            a=locked.copy() if locked is not None else np.array([box[0]+box[2]/2,box[1]+box[3]/2]) if box is not None else None
            c=np.array([landmarks.landmark[5].x*raw.shape[1],landmarks.landmark[5].y*raw.shape[0]]) if landmarks is not None else None
            aperture=float(np.linalg.norm(thumb-index))*engine.mm_per_px if thumb is not None and index is not None else np.nan
            valid=a is not None and c is not None and np.isfinite(aperture)
            reason='' if valid else 'missing hand/object'
            if valid and trial is not None and np.linalg.norm(a-origin)>args.max_object_shift:valid=False;reason='object centre jump'
            latest=dict(valid=valid,a=a,c=c,time=timestamp,shape=raw.shape)
            display=raw.copy()
            if landmarks is not None:mp.solutions.drawing_utils.draw_landmarks(display,landmarks,mp.solutions.hands.HAND_CONNECTIONS)
            if box is not None:cv2.rectangle(display,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),1)
            if a is not None:cv2.circle(display,tuple(a.astype(int)),4,(0,0,255),-1)
            if a is not None and c is not None:cv2.line(display,tuple(a.astype(int)),tuple(c.astype(int)),(0,255,255),1)
            prediction=dict(zip(KEYS,[np.nan]*3));common=False;gap=False;distance_mm=np.nan;latency=np.nan
            if trial is not None:
                distance_mm=float(np.linalg.norm(a-c))*engine.mm_per_px if valid else np.nan
                inference_start=time.perf_counter();prediction,common,gap=engine.step(distance_mm,timestamp,frame_id);latency=(time.perf_counter()-inference_start)*1000
            labels=[f'Measured: {aperture:.1f} mm']+[f'{name}: {prediction[key]:.1f}' if np.isfinite(prediction[key]) else f'{name}: --' for name,key in zip(['Function','MLP','LSTM'],KEYS)]
            for j,text in enumerate(labels):
                cv2.putText(display,text,(10,25+j*23),0,.55,(0,0,0),3,cv2.LINE_AA);cv2.putText(display,text,(10,25+j*23),0,.55,(255,255,255),1,cv2.LINE_AA)
            if trial is not None:
                cv2.putText(display,f'REC | MCP {distance_mm:.1f} mm | infer {latency:.1f} ms',(10,display.shape[0]-16),0,.48,(0,0,255),1,cv2.LINE_AA)
                xy=lambda p: [float(p[0]),float(p[1])] if p is not None else [np.nan,np.nan]
                ax,ay=xy(a);cx,cy=xy(c);tx,ty=xy(thumb);ix,iy=xy(index)
                row=dict(t_camera=timestamp,source_frame=frame_id,valid=valid,invalid_reason=reason,
                    finger_width=aperture/engine.mm_per_px,finger_width_mm=aperture,mm_per_px=engine.mm_per_px,index_mcp_object_distance_mm=distance_mm,initial_distance_mm=engine.initial_distance,distance_over_initial=engine.normalized_distance,
                    index_mcp_object_distance=float(np.linalg.norm(a-c)) if a is not None and c is not None else np.nan,
                    object_x=ax,object_y=ay,thumb_x=tx,thumb_y=ty,index_x=ix,index_y=iy,index_mcp_x=cx,index_mcp_y=cy,
                    common_valid=common and valid,reset_on_gap=gap,inference_ms=latency,capture_ms=capture_ms,detection_ms=detection_ms,processing_before_save_ms=(time.perf_counter()-began)*1000,**engine.last_timing,**prediction)
                trial.append(row,raw,display)
                if time.perf_counter()-last_chart>.15:last_plot=chart(trial.rows);last_chart=time.perf_counter()
            cv2.imshow('Three-model camera | N start S stop L lock Q quit',cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR) if mask_option.get() else display)
            cv2.imshow('Live aperture comparison',last_plot)
            key=cv2.waitKey(1)&255
            if key in (ord('n'),ord('N')):start()
            elif key in (ord('s'),ord('S')):stop()
            elif key in (ord('l'),ord('L')):lock()
            elif key in (ord('q'),ord('Q'),27):close()
            frame_id+=1
            delay=1/args.fps-(time.perf_counter()-began)
            if delay>0:time.sleep(delay)
    finally:
        try:stop()
        finally:cap.release();hands.close();cv2.destroyAllWindows();ui.destroy()
    print('Saved session:',session)

if __name__ == '__main__':
    main()
