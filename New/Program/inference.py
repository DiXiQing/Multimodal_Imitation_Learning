"""Live camera inference using the camera-only regression model; no IMU."""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import BSpline
from record_topview_black_cube import detect_cube, detect_fingers

ROOT = Path(__file__).resolve().parents[2]
# Change this to another saved model, or use --model with an absolute path.
MODEL_PATH = ROOT/'Data/TopView/TRIAL_Size_4cm_20261017_142000_558582/grasp_segments/Result/regression_35train_7test_20261017_142637_596810/models/best_model.json'


def load_model(path):
    path = Path(str(path).strip().strip('\"\''))
    if path.is_dir():
        path = path/'models'/'best_model.json' if (path/'models'/'best_model.json').exists() else path/'model.json'
    model = json.loads(path.read_text(encoding='utf-8-sig'))
    if model['kind'] not in ('polynomial', 'spline', 'mean', 'median_spline'):
        raise ValueError('Unsupported model kind')
    if not np.isfinite(model['scale']) or model['scale'] <= 0:
        raise ValueError('Invalid normalization scale')
    if not np.isfinite(model['coefficients']).all():
        raise ValueError('Invalid model coefficients')
    settings = model.get('recording_settings')
    if settings is None:
        for parent in path.resolve().parents:
            if (parent/'settings.json').is_file():
                settings = json.loads((parent/'settings.json').read_text(encoding='utf-8-sig'))
                break
    if settings is None:
        raise ValueError('Cannot find recording settings.json above model. Keep model in its original result directory.')
    return path.resolve(), model, settings


def predict(model, distance):
    z = (distance-model['x_min'])/model['scale']
    if model['kind'] == 'polynomial':
        return float(np.polynomial.polynomial.polyval(z, model['coefficients']))
    if model['kind'] == 'mean':
        return float(model['coefficients'][0])
    return float(BSpline(model['knots'], model['coefficients'], model['degree'], extrapolate=True)(z))


class ExperimentTrial:
    """Save exactly one CSV row per frame in both videos, including invalid frames."""
    def __init__(self, root, number, raw_size, panel_size, fps, columns, config):
        self.path = root/f'trial_{number:03d}'
        self.path.mkdir()
        self.rows = []
        self.raw = self.annotated = self.file = None
        self.started = time.perf_counter()
        self.closed = False
        try:
            self.raw = cv2.VideoWriter(str(self.path/'raw_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,raw_size)
            self.annotated = cv2.VideoWriter(str(self.path/'prediction_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,panel_size)
            if not self.raw.isOpened() or not self.annotated.isOpened():
                raise RuntimeError('Cannot create experiment video')
            self.file = (self.path/'predictions.csv').open('w',newline='',encoding='utf-8-sig')
            self.writer = csv.DictWriter(self.file,fieldnames=['trial_frame','trial_elapsed_s',*columns])
            self.writer.writeheader()
            (self.path/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        except Exception:
            for video in (self.raw,self.annotated):
                if video is not None:video.release()
            if self.file is not None:self.file.close()
            raise

    def save(self, raw, panel, row):
        saved=dict(trial_frame=len(self.rows),trial_elapsed_s=time.perf_counter()-self.started,**row)
        self.writer.writerow(saved)
        self.raw.write(raw)
        self.annotated.write(panel)
        self.rows.append(saved)
        if len(self.rows)%30==0:self.file.flush()

    def close(self):
        if self.closed:return
        self.closed=True
        self.raw.release();self.annotated.release();self.file.close()
        valid=[r for r in self.rows if r['valid']==1]
        in_range=[r for r in valid if r['outside_training_range']==0]
        def score(rows):
            errors=np.array([r['error_px'] for r in rows],dtype=float)
            return dict(frames=len(rows),mae_px=float(np.mean(abs(errors))) if len(errors) else None,
                        rmse_px=float(np.sqrt(np.mean(errors**2))) if len(errors) else None)
        summary=dict(total_frames=len(self.rows),invalid_frames=len(self.rows)-len(valid),
                     outside_training_range_frames=len(valid)-len(in_range),
                     all_valid=score(valid),within_training_range=score(in_range),
                     video_timing='Fixed FPS; use CSV timestamps for actual acquisition time.')
        (self.path/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        if self.rows:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            t=[r['trial_elapsed_s'] for r in self.rows]
            actual=[r['measured_aperture_px'] if r['measured_aperture_px']!='' else np.nan for r in self.rows]
            predicted=[r['predicted_aperture_px'] if r['valid']==1 else np.nan for r in self.rows]
            fig,ax=plt.subplots(figsize=(11,4.5))
            ax.plot(t,actual,color='#20242a',label='Measured',lw=2)
            ax.plot(t,predicted,color='#d95035',label='Predicted',lw=1.8)
            outside=[r for r in valid if r['outside_training_range']==1]
            if outside:ax.scatter([r['trial_elapsed_s'] for r in outside],[r['predicted_aperture_px'] for r in outside],marker='x',color='orange',label='Outside training range')
            ax.set(xlabel='Trial time (s)',ylabel='Aperture (px)',title=self.path.name+' | live inference')
            ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(self.path/'prediction_plot.png',dpi=170);plt.close(fig)
        print(f'[Saved] {self.path} | {len(self.rows)} frames | invalid={summary["invalid_frames"]}',flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(MODEL_PATH), help='Model JSON or result directory')
    parser.add_argument('--camera', type=int, help='Default: recording camera index')
    parser.add_argument('--output', type=Path, default=ROOT/'Data/TopView/Inference')
    parser.add_argument('--video', type=Path, help='Replay a recorded video instead of a camera')
    parser.add_argument('--no-preview', action='store_true', help='Replay/verification without opening a window')
    parser.add_argument('--max-frames', type=int, default=0, help='0: unlimited')
    parser.add_argument('--auto-record',action='store_true',help='Start a trial automatically (also useful for video replay)')
    args = parser.parse_args()
    try:
        path, model, settings = load_model(args.model)
    except (ValueError, KeyError, OSError) as exc:
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
    out = args.output/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    config = dict(model_path=str(path), model=model, recording_settings=settings,
                  camera=camera, video=str(args.video) if args.video else None,
                  input='hand_object_distance in pixels', output='finger_width in pixels',
                  error_definition='predicted minus measured', target='manually selected model; no automatic object classification')
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Model:', path)
    print(f'Training distance: {model["x_min"]:.2f} to {model["x_max"]:.2f} px')
    print(f'Expected image: {settings["width"]} x {settings["height"]}; keep camera pose and zoom identical to recording.')
    print('N: start trial | S: stop and save trial | L: lock/unlock center | M: mask | Q: save and quit')
    print('Results:', out)
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
                       'valid','outside_training_range','object_center_source']
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
                           valid=int(valid),object_center_source=source)
                if a is not None:row.update(object_x=a[0],object_y=a[1])
                if c is not None:row.update(hand_center_x=c[0],hand_center_y=c[1],measured_aperture_px=measured)
                lines = ['Waiting for object and hand...',f'Center: {source}']
                if valid:
                    distance = float(np.linalg.norm(a-c));estimate = predict(model,distance)
                    outside = not model['x_min'] <= distance <= model['x_max']
                    error = estimate-measured
                    row.update(distance_px=distance,normalized_distance=(distance-model['x_min'])/model['scale'],
                               predicted_aperture_px=estimate,error_px=error,outside_training_range=int(outside))
                    valid_frames += 1;outside_frames += int(outside)
                    if not outside:errors.append(error)
                    lines = [f'Distance: {distance:.1f} px | Center: {source}',
                             f'Measured B: {measured:.1f} px',f'Predicted B: {estimate:.1f} px',
                             f'Error (pred-measured): {error:+.1f} px',
                             'OUTSIDE TRAINING RANGE - extrapolation' if outside else 'Within training distance range']
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
                    cv2.imshow('Live grasp regression - manually selected target',preview)
                    key = cv2.waitKey(1)&255
                    if key in (ord('q'),27):break
                    if key == ord('m'):show_mask = not show_mask
                    if key == ord('n') and trial is None:
                        if valid:auto_record=True
                        else:print('Cannot start: detect both object and hand first.',flush=True)
                    if key == ord('s') and trial is not None:
                        finished=trial;trial=None;finished.close()
                    if key == ord('l') and trial is None:
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
