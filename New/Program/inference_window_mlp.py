"""Live camera inference using the causal five-frame distance MLP; no IMU."""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from record_topview_black_cube import detect_cube, detect_fingers

ROOT = Path(__file__).resolve().parents[2]
# Change this to another saved model, or use --model with an absolute path.
MODEL_PATH = ROOT/'New/Data/2/Result/distance_mlp_window5_20261017_213924_764497/models/distance_window5.pth'

from collections import deque
import torch
from torch import nn
from inference import ExperimentTrial as BaseExperimentTrial


class ExperimentTrial(BaseExperimentTrial):
    """Keep original videos/plots and report warm-up separately from detection loss."""
    def close(self):
        if self.closed:
            return
        super().close()
        path = self.path/'summary.json'
        summary = json.loads(path.read_text(encoding='utf-8'))
        summary.update(
            valid_definition='Five-frame window ready; warm-up and detection loss have no prediction.',
            missing_detection_frames=sum(r['detection_valid'] == 0 for r in self.rows),
            warmup_frames=sum(r['status'] == 'warming_up' for r in self.rows))
        for key, rows in [('all_valid', [r for r in self.rows if r['valid'] == 1]),
                          ('within_training_range', [r for r in self.rows if r['valid'] == 1 and r['outside_training_range'] == 0])]:
            y = np.asarray([r['measured_aperture_px'] for r in rows], dtype=float)
            errors = np.asarray([r['error_px'] for r in rows], dtype=float)
            denominator = float(np.sum((y-y.mean())**2)) if len(y) else 0
            summary[key]['r2'] = float(1-np.sum(errors**2)/denominator) if denominator > 1e-12 else None
        path.write_text(json.dumps(summary,indent=2),encoding='utf-8')


class DistanceMLP(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(.10),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model(path):
    path = Path(str(path).strip().strip('"' + chr(39)))
    if path.is_dir():
        path = (path/'distance_window5.pth') if (path/'distance_window5.pth').exists() else path/'models/distance_window5.pth'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema') != 'distance_mlp_v1' or saved.get('input_size') != 5 or saved.get('window_size') != 5:
        raise ValueError('Expected final five-frame distance MLP checkpoint.')
    scaler = saved['scaler']
    if not all(np.isfinite(scaler[k]) for k in ('x_min','x_scale','y_mean','y_scale')) or scaler['x_scale'] <= 0 or scaler['y_scale'] <= 0:
        raise ValueError('Invalid checkpoint normalization parameters.')
    if not np.isfinite(saved['max_frame_gap_s']) or saved['max_frame_gap_s'] <= 0:
        raise ValueError('Invalid checkpoint frame gap.')
    network = DistanceMLP(5, saved['hidden_size'])
    network.load_state_dict(saved['state_dict'], strict=True)
    if not all(torch.isfinite(v).all() for v in network.state_dict().values()):
        raise ValueError('Non-finite model parameters.')
    network.eval()
    metadata = {k:v for k,v in saved.items() if k != 'state_dict'}
    metadata.update(x_min=scaler['x_min'], x_max=scaler['x_min']+scaler['x_scale'], scale=scaler['x_scale'])
    return path.resolve(), metadata, saved['recording_settings'], network


class DistanceWindow:
    def __init__(self, max_gap):
        self.values = deque(maxlen=5)
        self.max_gap = max_gap
        self.last_time = None
        self.last_frame = None

    def clear(self):
        self.values.clear()
        self.last_time = self.last_frame = None

    def push(self, distance, timestamp, frame):
        if distance is None or not np.isfinite(distance) or not np.isfinite(timestamp):
            self.clear()
            return None
        if self.last_time is not None and (frame != self.last_frame+1 or not 0 < timestamp-self.last_time <= self.max_gap):
            self.clear()
        self.values.append((float(distance),float(timestamp),int(frame)))
        self.last_time, self.last_frame = timestamp, frame
        return [x[0] for x in self.values] if len(self.values)==5 else None


def predict(network, model, distances):
    scaler = model['scaler']
    x = (np.asarray(distances,dtype=np.float32)-scaler['x_min'])/scaler['x_scale']
    with torch.inference_mode():
        value = network(torch.from_numpy(x).reshape(1,5)).item()
    result = value*scaler['y_scale']+scaler['y_mean']
    if not np.isfinite(result):
        raise RuntimeError('Non-finite MLP prediction')
    return float(result)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(MODEL_PATH), help='Five-frame .pth model or result directory')
    parser.add_argument('--camera', type=int, help='Default: recording camera index')
    parser.add_argument('--output', type=Path, default=ROOT/'Data/TopView/Inference')
    parser.add_argument('--video', type=Path, help='Replay a recorded video instead of a camera')
    parser.add_argument('--no-preview', action='store_true', help='Replay/verification without opening a window')
    parser.add_argument('--max-frames', type=int, default=0, help='0: unlimited')
    parser.add_argument('--auto-record',action='store_true',help='Start a trial automatically (also useful for video replay)')
    args = parser.parse_args()
    try:
        path, model, settings, network = load_model(args.model)
        torch.set_num_threads(min(4, torch.get_num_threads()))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise SystemExit('Run in the gripper environment used for recording (mp.solutions.hands required).')
    camera = settings.get('camera', 1) if args.camera is None else args.camera
    cap = cv2.VideoCapture(str(args.video)) if args.video else cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise SystemExit('Cannot open camera/video. Try --camera 0 or check --video path.')
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
        cap.set(cv2.CAP_PROP_FPS, settings['fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    out = args.output/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    config = dict(model_path=str(path), model=model, recording_settings=settings,
                  camera=camera, video=str(args.video) if args.video else None,
                  input='five consecutive distances, oldest to current; checkpoint normalization', output='finger_width in pixels',
                  error_definition='predicted minus measured', target='manually selected model; no automatic object classification')
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Model:', path)
    print(f'Training distance: {model["x_min"]:.2f} to {model["x_max"]:.2f} px')
    print(f'Expected image: {settings["width"]} x {settings["height"]}; keep camera pose and zoom identical to recording.')
    print('N: start trial | S: stop and save trial | L: lock/unlock center | M: mask | Q: save and quit')
    print('Results:', out)
    print('Five-frame causal window; missing detections/gaps reset it. Match recording FPS:', settings['fps'])
    window = DistanceWindow(model['max_frame_gap_s'])
    locked = None
    if settings.get('center_source') == 'locked' and settings.get('locked_center') is not None:
        print('Training used locked center. Reposition object and press L to lock current center if needed.')
    show_mask = False
    errors = []
    frames = valid_frames = outside_frames = 0
    started = time.perf_counter()
    hands = None
    trial = None
    trial_number = 0
    auto_record = args.auto_record
    try:
        hands = mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,
                                        min_detection_confidence=.6,min_tracking_confidence=.6)
        with (out/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
            columns = ['frame','unix_time_s','elapsed_s','video_time_s','object_x','object_y','hand_center_x','hand_center_y',
                       'distance_px','normalized_distance','measured_aperture_px','predicted_aperture_px','error_px',
                       'valid','outside_training_range','object_center_source','detection_valid',
                       'window_count','window_start_frame','window_duration_s','window_distances_px','status','inference_ms']
            writer = csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            while True:
                ok, raw = cap.read()
                if not ok:
                    if args.video:break
                    raise RuntimeError('Camera frame acquisition failed')
                h,w = raw.shape[:2]
                if (w,h) != (settings['width'],settings['height']):
                    raise RuntimeError(f'Image resolution {w}x{h} differs from training {settings["width"]}x{settings["height"]}. Pixel distances would not match.')
                timestamp = time.time();elapsed = time.perf_counter()-started
                sample_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if args.video else elapsed
                (area,box),mask = detect_cube(raw,settings['v'],settings['s'],settings['area'],
                    color=settings.get('object_color','black'),color_s_min=settings.get('color_s_min',140),
                    color_v_min=settings.get('color_v_min',50))
                thumb,index,landmarks = detect_fingers(raw,hands)
                a = locked.copy() if locked is not None else None
                if a is None and box is not None:
                    x,y,bw,bh = box;a = np.array([x+bw/2,y+bh/2])
                c = (thumb+index)/2 if thumb is not None else None
                measured = float(np.linalg.norm(thumb-index)) if c is not None else None
                valid = a is not None and c is not None
                source = 'locked' if locked is not None else ('detected' if box is not None else 'missing')
                row = dict.fromkeys(columns,'')
                row.update(frame=frames,unix_time_s=timestamp,elapsed_s=elapsed,
                           video_time_s=cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if args.video else '',
                           valid=0,detection_valid=int(valid),object_center_source=source)
                if a is not None:row.update(object_x=a[0],object_y=a[1])
                if c is not None:row.update(hand_center_x=c[0],hand_center_y=c[1],measured_aperture_px=measured)
                lines = ['Waiting for object and hand...',f'Center: {source}']
                distance = float(np.linalg.norm(a-c)) if valid else None
                distances = window.push(distance, sample_time, frames)
                row.update(window_count=len(window.values), status='missing_detection' if not valid else 'warming_up')
                if valid:
                    row.update(distance_px=distance, normalized_distance=(distance-model['x_min'])/model['scale'])
                    lines = [f'Distance: {distance:.1f} px | Center: {source}',
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
                    lines = [f'Distance: {distance:.1f} px | Window: 5/5 | Center: {source}',
                             f'Measured B: {measured:.1f} px',f'Predicted B: {estimate:.1f} px | Window span: {row["window_duration_s"]:.3f}s',
                             f'Error: {error:+.1f} px | MLP: {inference_ms:.2f} ms',
                             'OUTSIDE TRAINING RANGE (any window input)' if outside else 'Within training distance range']
                writer.writerow(row);frames += 1
                if frames%30 == 0:f.flush()
                view = raw.copy()
                if landmarks is not None:
                    mp.solutions.drawing_utils.draw_landmarks(view,landmarks,mp.solutions.hands.HAND_CONNECTIONS)
                if box is not None:
                    x,y,bw,bh=box;cv2.rectangle(view,(x,y),(x+bw,y+bh),(0,255,0),2)
                if a is not None:cv2.circle(view,tuple(a.astype(int)),5,(0,0,255),-1)
                if valid:cv2.line(view,tuple(a.astype(int)),tuple(c.astype(int)),(0,255,255),2)
                panel = np.full((h+170,max(w,760),3),245,np.uint8);panel[:h,:w]=view
                for i,line in enumerate(lines):cv2.putText(panel,line,(12,h+25+i*25),0,.55,(30,30,30),1,cv2.LINE_AA)
                cv2.putText(panel,'N: record | S: stop | L: lock center | M: mask | Q: quit',(12,h+155),0,.5,(80,80,80),1)
                if auto_record:
                    trial_number+=1
                    trial_config=dict(config,object_center_source=source,
                        locked_center=locked.tolist() if locked is not None else None)
                    trial=ExperimentTrial(out,trial_number,(w,h),(panel.shape[1],panel.shape[0]),settings['fps'],columns,trial_config)
                    auto_record=False
                    print(f'[REC] trial_{trial_number:03d}',flush=True)
                if trial is not None:
                    cv2.putText(panel,f'REC {trial_number:03d} | frame {len(trial.rows)}',(12,25),0,.65,(0,0,255),2)
                    trial.save(raw,panel,row)
                if not args.no_preview:
                    preview=panel.copy()
                    if show_mask:preview[:h,:w]=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
                    cv2.imshow('Live five-frame MLP - manually selected target',preview)
                    key = cv2.waitKey(1)&255
                    if key in (ord('q'),27):break
                    if key == ord('m'):show_mask = not show_mask
                    if key == ord('n') and trial is None:
                        if valid:
                            window.clear()
                            auto_record=True
                        else:print('Cannot start: detect both object and hand first.',flush=True)
                    if key == ord('s') and trial is not None:
                        finished=trial;trial=None;finished.close()
                        window.clear()
                    if key == ord('l') and trial is None:
                        window.clear()
                        if locked is not None:locked = None
                        elif box is not None:
                            x,y,bw,bh=box;locked=np.array([x+bw/2,y+bh/2])
                if args.max_frames>0 and frames>=args.max_frames:break
    finally:
        cap.release()
        if hands is not None:hands.close()
        if not args.no_preview:cv2.destroyAllWindows()
        if trial is not None:trial.close()
        summary = dict(frames=frames,valid_frames=valid_frames,outside_training_range_frames=outside_frames,
            in_range_frames=len(errors),in_range_mae_px=float(np.mean(np.abs(errors))) if errors else None,
            in_range_rmse_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            note='Live descriptive errors, not an independent held-out evaluation.')
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(f'Saved: {out} | frames={frames}, valid={valid_frames}, outside range={outside_frames}')

if __name__ == '__main__':main()
