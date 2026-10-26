"""Live function / window MLP / continuous LSTM comparison. Never trains.

N starts one complete trial from the initial posture; S ends it. L locks the
stationary object centre before a trial. Q exits and saves. --check-models loads
and validates models on synthetic inputs without opening a camera or audio.
"""
import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Record import detect_cube, detect_fingers, crop_camera_frame, approach_axis, signed_hand_distance
from train_function_mlp_lstm_comparison import FunctionApproximation, DistanceMLP, ContinuousDistanceLSTM

ROOT=Path(__file__).resolve().parents[2]
MODEL_DIR=ROOT/'Data/D4/Result/fixed_function_mlp_lstm_window5_20261026_205550_700402/models'
OUTPUT=ROOT/'New/Result/RealTime'
CAMERA_INDEX = 1  # Camera number: change to 1, 2, ... when running from the IDE.
KEYS=['predicted_function_px','predicted_mlp_px','predicted_lstm_px']
LABELS=['Function approximation','MLP (5 frames)','LSTM (full history)']
COLORS=['#3485b8','#e59a32','#519e58']


def write_csv(path, rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class ComparisonEngine:
    """One causal state per trial. Input gaps clear both history and LSTM state."""
    def __init__(self, folder):
        folder=Path(folder)
        fn=json.loads((folder/'function_approximation.json').read_text(encoding='utf-8-sig'))
        mlp=torch.load(folder/'distance_window5.pth',map_location='cpu',weights_only=True)
        lstm=torch.load(folder/'distance_lstm_continuous.pth',map_location='cpu',weights_only=True)
        if fn.get('schema')!='distance_function_v1' or mlp.get('schema')!='distance_mlp_v1' or lstm.get('schema')!='distance_lstm_continuous_v1':
            raise ValueError('Expected function JSON, 5-frame MLP and continuous LSTM checkpoints.')
        if mlp.get('window_size')!=5 or mlp.get('input_size')!=5 or lstm.get('input_size')!=1:
            raise ValueError('This live comparison requires a 5-frame MLP and one-feature continuous LSTM.')
        self.saved=[fn,mlp,lstm]
        if any(s.get('distance_column')!='hand_object_signed_distance' for s in self.saved):
            raise ValueError('All three checkpoints must use hand_object_signed_distance.')
        signatures=[tuple(s['recording_settings'].get(k) for k in ('object_color','size_cm','width','height','crop_margins_px')) for s in self.saved]
        if any(sig!=signatures[0] for sig in signatures):
            raise ValueError('Model recording classes, image sizes or camera crops do not agree.')
        for s in self.saved:
            scale=s['scaler']
            if not all(np.isfinite(scale[k]) for k in ('x_min','x_scale','y_mean','y_scale')) or min(scale['x_scale'],scale['y_scale'])<=0:
                raise ValueError('Invalid checkpoint scaler.')
        self.settings=fn['recording_settings']
        self.gap=min(float(s['max_frame_gap_s']) for s in self.saved)
        if not np.isfinite(self.gap) or self.gap<=0:raise ValueError('Invalid checkpoint maximum frame gap.')
        self.function=FunctionApproximation(fn['function'])
        self.mlp=DistanceMLP(5,mlp['hidden_size']);self.mlp.load_state_dict(mlp['state_dict']);self.mlp.eval()
        self.lstm=ContinuousDistanceLSTM(lstm['hidden_size']);self.lstm.load_state_dict(lstm['state_dict']);self.lstm.eval()
        self.reset()

    def reset(self):
        self.history=deque(maxlen=5);self.state=None;self.previous=None

    def step(self,distance,timestamp,frame):
        if distance is None or not np.isfinite(distance):
            self.reset();return dict(zip(KEYS,[np.nan]*3)),False,False
        gap=self.previous is not None and (frame-self.previous[1]!=1 or not 0<timestamp-self.previous[0]<=self.gap)
        if gap:self.reset()
        self.previous=(timestamp,frame);self.history.append(float(distance))
        values=[]
        for j,saved in enumerate(self.saved):
            scale=saved['scaler']
            with torch.inference_mode():
                if j==0:
                    value=float(self.function.predict(np.array([(distance-scale['x_min'])/scale['x_scale']]))[0])
                elif j==1:
                    if len(self.history)<5:values.append(np.nan);continue
                    x=torch.from_numpy(((np.array(self.history)-scale['x_min'])/scale['x_scale']).astype(np.float32)[None,:])
                    value=float(self.mlp(x)[0])
                else:
                    x=torch.tensor([[[(distance-scale['x_min'])/scale['x_scale']]]],dtype=torch.float32)
                    prediction,self.state=self.lstm(x,self.state)
                    value=float(prediction[0,0])
                values.append(value*scale['y_scale']+scale['y_mean'])
        if not np.isfinite(values[0]) or not np.isfinite(values[2]) or (len(self.history)==5 and not np.isfinite(values[1])):
            raise ValueError('Nonfinite model prediction.')
        return dict(zip(KEYS,values)),len(self.history)==5,gap


def score_rows(rows):
    common=[r for r in rows if r['common_valid'] and all(np.isfinite(r[k]) for k in ['finger_width',*KEYS])]
    result=[]
    for key,label in zip(KEYS,LABELS):
        if not common:continue
        y=np.array([r['finger_width'] for r in common]);p=np.array([r[key] for r in common]);e=p-y
        denominator=np.sum((y-y.mean())**2)
        result.append(dict(model=label,frames=len(y),mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),
                           r2=float(1-(e*e).sum()/denominator) if denominator>1e-12 else None))
    return result


def plot_scores(path,scores):
    if not scores:return
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
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
            mean_inference_ms=float(np.mean([r['inference_ms'] for r in self.rows])) if self.rows else None)
        (self.path/'timing_and_coverage.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
        fig,ax=plt.subplots(figsize=(11,4.5))
        if self.rows:
            t=[r['elapsed_s'] for r in self.rows]
            measured=[r['finger_width'] if r['valid'] else np.nan for r in self.rows]
            ax.plot(t,measured,color='#222222',lw=2,label='Measured')
            for key,label,color in zip(KEYS,LABELS,COLORS):ax.plot(t,[r[key] for r in self.rows],color=color,label=label,lw=1.4)
            ax.legend()
        ax.set(xlabel='Time from trial start (s)',ylabel='Aperture (px)');ax.grid(alpha=.2);fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(self.path/f'prediction_comparison.{ext}',dpi=180)
        plt.close(fig)
        return scores


def chart(rows):
    image=np.full((300,720,3),255,np.uint8)
    cv2.putText(image,'Measured / Function / MLP / LSTM',(14,22),0,.55,(40,40,40),1,cv2.LINE_AA)
    if len(rows)<2:return image
    times=np.array([r['elapsed_s'] for r in rows]);last=max(float(times[-1]),1.)
    data=[np.array([r['finger_width'] if r['valid'] else np.nan for r in rows])]+[np.array([r[k] for r in rows]) for k in KEYS]
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
    cv2.putText(image,f'0 - {last:.1f} s | Aperture {lower:.0f}..{upper:.0f} px',(45,286),0,.5,(50,50,50),1)
    return image


def check_models(engine):
    engine.reset()
    for i,d in enumerate(np.linspace(250,-10,30)):
        values,common,_=engine.step(float(d),100+i/30,i)
        assert common==(i>=4)
        if common:assert all(np.isfinite(list(values.values())))
    first=engine.saved[2]['scaler'];x=torch.tensor((np.linspace(250,-10,30)-first['x_min'])/first['x_scale'],dtype=torch.float32).reshape(1,-1,1)
    with torch.inference_mode():whole,_=engine.lstm(x)
    expected=float(whole[0,-1])*first['y_scale']+first['y_mean']
    assert abs(values[KEYS[2]]-expected)<1e-3
    engine.reset();assert engine.state is None and not engine.history
    for i in range(5):engine.step(200.,100+i/30,i)
    _,common,gap=engine.step(190.,101.,5);assert gap and not common
    engine.step(None,102.,6);assert engine.state is None and not engine.history
    print('PASS: model loading, scalers, five-frame warmup, continuous streaming, trial/gap resets. No camera, audio or training started.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,default=MODEL_DIR)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--camera',type=int,default=CAMERA_INDEX)
    parser.add_argument('--fps',type=float,default=30.)
    parser.add_argument('--max-object-shift',type=float,default=40.,help='Reject centre jumps relative to trial start (px); object must stay fixed')
    parser.add_argument('--check-models',action='store_true')
    args=parser.parse_args()
    if args.fps<=0 or args.max_object_shift<=0:parser.error('FPS and object shift threshold must be positive.')
    torch.set_num_threads(2)
    engine=ComparisonEngine(args.models)
    if args.check_models:check_models(engine);return
    import tkinter as tk
    from tkinter import messagebox,ttk
    import mediapipe as mp
    settings=engine.settings
    margins=tuple(int(settings.get('crop_margins_px',{}).get(k,0)) for k in ('left','top','right','bottom'))
    expected=(int(settings['width']),int(settings['height']))
    color=settings.get('object_color','black')
    session=args.output/f'three_model_live_{datetime.now():%Y%m%d_%H%M%S_%f}'
    session.mkdir(parents=True,exist_ok=False)
    (session/'model_sources.json').write_text(json.dumps(dict(models=str(args.models.resolve()),files=['function_approximation.json','distance_window5.pth','distance_lstm_continuous.pth'],distance='hand_object_signed_distance',common_frame_rule='Five consecutive valid inputs after every reset; LSTM observes all valid inputs, including warmup',units='px',video_timing='Nominal FPS video; use CSV timestamps for actual time'),indent=2),encoding='utf-8')
    cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW)
    if not cap.isOpened():raise RuntimeError('Cannot open camera.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(settings.get('camera_source_width',expected[0]+margins[0]+margins[2])))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(settings.get('camera_source_height',expected[1]+margins[1]+margins[3])))
    cap.set(cv2.CAP_PROP_FPS,args.fps)
    hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=.5,min_tracking_confidence=.5)
    ui=tk.Tk();ui.title('三模型实时对比');ui.geometry('420x650')
    status=tk.StringVar(value='手放回起点，确认物体和手部检测，再按 N 开始。')
    tk.Label(ui,text=f'{color} / {settings["size_cm"]:g} cm | 有符号距离',font=('',13)).pack(pady=6)
    tk.Label(ui,textvariable=status,wraplength=395,justify='left').pack(fill='x',padx=8)
    target=tk.StringVar(value='1')
    row=tk.Frame(ui);row.pack()
    for text in ('1','2','3'):tk.Radiobutton(row,text=text+' 秒条件',variable=target,value=text).pack(side='left')
    values={key:tk.IntVar(value=int(settings.get(key,default))) for key,default in [('v',60),('s',120),('area',500),('max_area',640000),('color_s_min',140),('color_v_min',50)]}
    controls=[]
    for label,key in ([('黑色亮度上限','v'),('黑色饱和度上限','s')] if color=='black' else [('颜色饱和度下限','color_s_min'),('颜色亮度下限','color_v_min')])+[('最小面积','area')]:
        control=tk.Scale(ui,label=label,variable=values[key],from_=0,to=3000 if key=='area' else 255,orient='horizontal',length=350);control.pack();controls.append(control)
    row=tk.Frame(ui);row.pack();tk.Label(row,text='最大物体面积 px²').pack(side='left')
    entry=tk.Entry(row,textvariable=values['max_area'],width=12);entry.pack(side='left');controls.append(entry)
    mask_option=tk.BooleanVar(value=False);tk.Checkbutton(ui,text='显示检测掩膜',variable=mask_option).pack()
    trial=None;axis=None;origin=None;locked=None;latest=None;closing=False;completed=[];last_plot=np.full((300,720,3),255,np.uint8)
    detection={k:v.get() for k,v in values.items()}

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
                        all_rows.append(dict(target_s=condition,common_valid=r['common_valid']=='True',**{k:float(r[k]) for k in ['finger_width',*KEYS]}))
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
            axis=approach_axis(latest['a'],latest['c']);origin=latest['a'].copy();engine.reset()
            metadata=dict(recording_settings=settings,detection=detection.copy(),crop_margins_px=margins,
                model_directory=str(args.models.resolve()),target_movement_s=int(target.get()),
                signed_distance_axis_xy=axis.tolist(),initial_object_center_xy=origin.tolist(),object_center_locked=locked is not None,
                common_frame_rule='Five consecutive valid frames after trial/gap reset',max_frame_gap_s=engine.gap,
                object_jump_reject_px=args.max_object_shift,units='px')
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
            ok,raw=cap.read();timestamp=time.time()
            if not ok:raise RuntimeError('Camera read failed.')
            raw=crop_camera_frame(raw,margins)
            if (raw.shape[1],raw.shape[0])!=expected:raise RuntimeError(f'Camera cropped size {raw.shape[1]}x{raw.shape[0]} differs from model {expected}. No automatic resizing is applied.')
            if trial is None:
                try:
                    candidate={k:v.get() for k,v in values.items()}
                    if candidate['max_area']>0 and candidate['area']>=0 and all(0<=candidate[k]<=255 for k in ('v','s','color_s_min','color_v_min')):detection=candidate
                except (tk.TclError,ValueError):pass
            thumb,index,landmarks=detect_fingers(raw,hands)
            (area,box),mask=detect_cube(raw,detection['v'],detection['s'],detection['area'],color=color,color_s_min=detection['color_s_min'],color_v_min=detection['color_v_min'],max_area=detection['max_area'])
            a=locked.copy() if locked is not None else np.array([box[0]+box[2]/2,box[1]+box[3]/2]) if box is not None else None
            c=(thumb+index)/2 if thumb is not None else None
            aperture=float(np.linalg.norm(thumb-index)) if thumb is not None else np.nan
            valid=a is not None and c is not None
            reason='' if valid else 'missing hand/object'
            if valid and trial is not None and np.linalg.norm(a-origin)>args.max_object_shift:valid=False;reason='object centre jump'
            latest=dict(valid=valid,a=a,c=c,time=timestamp,shape=raw.shape)
            display=raw.copy()
            if landmarks is not None:mp.solutions.drawing_utils.draw_landmarks(display,landmarks,mp.solutions.hands.HAND_CONNECTIONS)
            if box is not None:cv2.rectangle(display,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),1)
            if a is not None:cv2.circle(display,tuple(a.astype(int)),4,(0,0,255),-1)
            if a is not None and c is not None:cv2.line(display,tuple(a.astype(int)),tuple(c.astype(int)),(0,255,255),1)
            prediction=dict(zip(KEYS,[np.nan]*3));common=False;gap=False;signed=np.nan;latency=np.nan
            if trial is not None:
                signed=signed_hand_distance(a,c,axis) if valid else np.nan
                inference_start=time.perf_counter();prediction,common,gap=engine.step(signed,timestamp,frame_id);latency=(time.perf_counter()-inference_start)*1000
            labels=[f'Measured: {aperture:.1f} px']+[f'{name}: {prediction[key]:.1f}' if np.isfinite(prediction[key]) else f'{name}: --' for name,key in zip(['Function','MLP','LSTM'],KEYS)]
            for j,text in enumerate(labels):
                cv2.putText(display,text,(10,25+j*23),0,.55,(0,0,0),3,cv2.LINE_AA);cv2.putText(display,text,(10,25+j*23),0,.55,(255,255,255),1,cv2.LINE_AA)
            if trial is not None:
                cv2.putText(display,f'REC | signed {signed:+.1f} px | infer {latency:.1f} ms',(10,display.shape[0]-16),0,.48,(0,0,255),1,cv2.LINE_AA)
                xy=lambda p: [float(p[0]),float(p[1])] if p is not None else [np.nan,np.nan]
                ax,ay=xy(a);cx,cy=xy(c);tx,ty=xy(thumb);ix,iy=xy(index)
                row=dict(t_camera=timestamp,source_frame=frame_id,valid=valid,invalid_reason=reason,
                    finger_width=aperture,hand_object_signed_distance=signed,
                    hand_object_distance=float(np.linalg.norm(a-c)) if a is not None and c is not None else np.nan,
                    object_x=ax,object_y=ay,thumb_x=tx,thumb_y=ty,index_x=ix,index_y=iy,hand_center_x=cx,hand_center_y=cy,
                    common_valid=common and valid,reset_on_gap=gap,inference_ms=latency,**prediction)
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


if __name__=='__main__':main()
