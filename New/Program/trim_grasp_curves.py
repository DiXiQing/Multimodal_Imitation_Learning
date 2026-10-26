"""Drag a retained frame range on an aperture curve; save backed-up synchronized crops."""
import argparse
import csv
import json
import queue
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT/'New/Result/RealTime/three_model_live_20261026_210818_947238/selected_best3_by_speed'
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trial_csv(folder):
    for name in ('camera.csv','predictions.csv'):
        path=Path(folder)/name
        if path.is_file():return path
    raise ValueError(f'找不到 camera.csv 或 predictions.csv：{folder}')


def discover_trials(base):
    base=Path(base).resolve()
    if any((base/name).is_file() for name in ('camera.csv','predictions.csv')):return [base]
    found=set()
    for name in ('camera.csv','predictions.csv'):
        for path in base.rglob(name):
            if any(p.lower() in ('backup','prepared','original') or p.startswith('.') for p in path.relative_to(base).parts[:-1]):continue
            if path.is_file():found.add(path.parent)
    return sorted(found)


def refresh_live_results(folder,rows):
    """Re-score retained predictions only; do not re-run any model."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    keys=['predicted_function_px','predicted_mlp_px','predicted_lstm_px']
    labels=['Function approximation','MLP (5 frames)','LSTM (full history)']
    colors=['#3485b8','#e59a32','#519e58']
    def number(r,k):
        try:return float(r[k])
        except (KeyError,ValueError,TypeError):return np.nan
    common=[r for r in rows if r.get('common_valid','').lower() in ('true','1') and
            all(np.isfinite(number(r,k)) for k in ['finger_width',*keys])]
    scores=[]
    for key,label in zip(keys,labels):
        if not common:continue
        y=np.array([number(r,'finger_width') for r in common]);p=np.array([number(r,key) for r in common]);e=p-y
        den=np.sum((y-y.mean())**2)
        scores.append(dict(model=label,frames=len(y),mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),r2=float(1-(e*e).sum()/den) if den>1e-12 else None))
    score_path=folder/'metrics_common_frames.csv'
    write_csv(score_path,['model','frames','mae','rmse','r2'],scores)
    fig=Figure(figsize=(11,4.5));FigureCanvasAgg(fig);ax=fig.subplots()
    t=[number(r,'elapsed_s') for r in rows]
    ax.plot(t,[number(r,'finger_width') if r.get('valid','').lower() in ('true','1') else np.nan for r in rows],color='#222222',lw=2,label='Measured')
    for key,label,color in zip(keys,labels,colors):ax.plot(t,[number(r,key) for r in rows],label=label,color=color,lw=1.4)
    ax.set(xlabel='Time from cropped segment start (s)',ylabel='Aperture (px)');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'prediction_comparison.{ext}',dpi=180)
    fig=Figure(figsize=(12,4));FigureCanvasAgg(fig);axes=fig.subplots(1,3)
    for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
        if scores:
            bars=ax.bar(labels,[r[key] if r[key] is not None else np.nan for r in scores],color=colors)
            ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3)
        ax.set_title(title);ax.tick_params(axis='x',labelsize=7);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.25)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'metrics_comparison.{ext}',dpi=180)
    dt=np.diff([number(r,'t_camera') for r in rows])
    info=dict(saved_frames=len(rows),valid_frames=sum(r.get('valid','').lower() in ('true','1') for r in rows),common_frames=len(common),
              observed_fps=float(1/np.mean(dt)) if len(dt) and np.mean(dt)>0 else None,
              note='Recomputed for manually retained interval. Predictions retain their ORIGINAL inference history; no model rerun or state reset.')
    (folder/'timing_and_coverage.json').write_text(json.dumps(info,indent=2),encoding='utf-8')


def crop_trial(folder, first, last, expected_csv):
    """1-based inclusive selection. Prepare/verify first, then swap; no source deletion."""
    folder = Path(folder).resolve()
    csv_path=trial_csv(folder)
    live=csv_path.name=='predictions.csv'
    if csv_path.read_bytes() != expected_csv:
        raise ValueError('数据已被其他程序修改，请重新打开文件夹。')
    fields, rows = read_csv(csv_path)
    if not 1 <= first < last <= len(rows):
        raise ValueError('请选择至少两帧，且范围不能超出原始帧数。')
    if first == 1 and last == len(rows):
        raise ValueError('当前选择包含全部帧，无需裁剪。')
    kept = [dict(r) for r in rows[first-1:last]]
    if live and 'elapsed_s' in fields:
        if 'source_elapsed_s' not in fields:fields.append('source_elapsed_s')
        origin=float(kept[0]['t_camera'])
        for row in kept:
            row.setdefault('source_elapsed_s',row['elapsed_s'])
            row['elapsed_s']=str(float(row['t_camera'])-origin)
    backup = folder.parent/'backup'/f'curve_crop_{datetime.now():%Y%m%d_%H%M%S_%f}_{folder.name}'
    backup.mkdir(parents=True, exist_ok=False)
    stage = backup/'prepared'
    shutil.copytree(folder, stage)
    # Retain source timestamps/IDs and values; regenerate only segment-relative columns.
    if 'segment_frame' in fields:
        for i, row in enumerate(kept):
            row['segment_frame'] = str(i)
    if 'segment_elapsed_s' in fields:
        key = 't_camera' if 't_camera' in fields else 'segment_elapsed_s'
        origin = float(rows[first-1][key])
        for i, row in enumerate(kept):
            row['segment_elapsed_s'] = str(float(rows[first-1+i][key])-origin)
    write_csv(stage/csv_path.name, fields, kept)
    for video in folder.iterdir():
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        cap = cv2.VideoCapture(str(video))
        writer = None
        try:
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
                raise ValueError(f'{video.name} 与 CSV 帧数不一致，已停止；原数据未改动。')
            size = (int(cap.get(3)), int(cap.get(4)))
            # Decode from the beginning to avoid codec/keyframe seek inaccuracies.
            output = stage/(video.stem+'.mp4')
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), size)
            if not writer.isOpened():
                raise RuntimeError('无法创建裁剪视频')
            for i in range(last):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError('视频提前结束')
                if i >= first-1:
                    writer.write(image)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        check = cv2.VideoCapture(str(output))
        count = int(check.get(7))
        check.release()
        if count != len(kept):
            raise RuntimeError('输出视频帧数校验失败')
        if output.name != video.name:
            (stage/video.name).unlink()
    frame_dir = stage/'frames'
    if frame_dir.is_dir():
        keep_names = {f'frame_{int(r["frame"]):06d}.jpg' for r in kept}
        expected = {f'frame_{int(r["frame"]):06d}.jpg' for r in rows}
        actual = {p.name for p in frame_dir.glob('*.jpg')}
        if actual != expected:
            raise ValueError('帧图片文件名/数量与 CSV 不一致，已停止；原数据未改动。')
        for image in frame_dir.glob('*.jpg'):
            if image.name not in keep_names:
                if not image.resolve().is_relative_to(stage.resolve()):
                    raise ValueError('图片路径超出临时副本')
                image.unlink()
    meta_path = stage/'segment.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
        old_start = int(meta['start'])
        start, end = old_start+first-1, old_start+last-1
        meta.update(start=start, end=end, frames=len(kept), source_start_frame=int(kept[0]['frame']),
                    source_end_frame=int(kept[-1]['frame']),
                    duration_s=float(kept[-1]['t_camera'])-float(kept[0]['t_camera']),
                    invalid_frames=sum(r.get('valid') != '1' for r in kept))
        if 'onset' in meta:
            meta['pre_onset_frames'] = max(0, min(end+1, int(meta['onset']))-start)
        if 'stable' in meta:
            meta['post_stable_frames'] = max(0, end-max(start, int(meta['stable'])))
        meta.setdefault('manual_curve_crops', []).append(dict(first=first, last=last,
            original_frames=len(rows), backup=str(backup/'original'), timestamp=datetime.now().isoformat()))
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if live:refresh_live_results(stage,kept)
    crop_note=dict(first=first,last=last,original_frames=len(rows),retained_frames=len(kept),
                   csv_name=csv_path.name,backup=str(backup/'original'),
                   prediction_history='Retained original predictions, not rerun at crop start' if live else None)
    (stage/'manual_crop_info.json').write_text(json.dumps(crop_note,indent=2),encoding='utf-8')
    _, verified = read_csv(stage/csv_path.name)
    assert verified == kept
    # Recheck before replacing: edits made during preparation must not be lost.
    if csv_path.read_bytes() != expected_csv:
        raise ValueError('准备期间数据发生变化，已停止；原数据未改动。')
    original = backup/'original'
    if not original.resolve().is_relative_to(folder.parent.resolve()):
        raise ValueError('备份路径超出数据目录')
    folder.rename(original)
    try:
        stage.rename(folder)
    except Exception:
        original.rename(folder)
        raise
    manifest = dict(trial=folder.name, first=first, last=last, original_frames=len(rows),
                    retained_frames=len(kept), removed_frames=len(rows)-len(kept),
                    backup=str(original), csv_name=csv_path.name,
                    note='Live plots and metrics regenerated; predictions not rerun. Parent selection/ranking stays based on original full trials.' if live else 'Only segment-relative CSV fields regenerated; other retained values unchanged. Existing analysis plots/results require regeneration.')
    (backup/'crop_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def launch(initial):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.widgets import SpanSelector

    window = tk.Tk()
    window.title('曲线裁剪：拖选要保留的帧范围')
    window.geometry('1150x760')
    state = dict(base=None, folders=[], index=0, rows=[], bytes=b'', pending={}, busy=False)
    result_queue = queue.SimpleQueue()
    top = ttk.Frame(window)
    top.pack(fill='x', padx=8, pady=6)
    path_text = tk.StringVar()
    status = tk.StringVar(value='拖选曲线中的保留区间，再点“保存当前裁剪”。不保存不改数据。')
    body = ttk.Frame(window)
    body.pack(fill='both', expand=True)
    listing = tk.Listbox(body, width=33, exportselection=False)
    listing.pack(side='left', fill='y', padx=6)
    figure = Figure(figsize=(8, 5), dpi=100)
    ax = figure.add_subplot(111)
    canvas = FigureCanvasTkAgg(figure, body)
    canvas.get_tk_widget().pack(side='left', fill='both', expand=True)
    controls = ttk.Frame(window)
    controls.pack(fill='x', padx=8, pady=6)
    first_var, last_var = tk.StringVar(), tk.StringVar()
    ttk.Label(window, textvariable=status, wraplength=1100).pack(fill='x', padx=8, pady=8)

    def select_range(left, right):
        if state['busy'] or not state['rows']:
            return
        n = len(state['rows'])
        first = max(1, min(n, int(np.floor(left+.5))))
        last = max(1, min(n, int(np.floor(right+.5))))
        if last <= first:
            return
        folder = state['folders'][state['index']]
        state['pending'][folder] = (first, last)
        first_var.set(str(first))
        last_var.set(str(last))
        selector.extents = (first, last)
        shades[0].set_xy([[.5,0],[first-.5,0],[first-.5,1],[.5,1],[.5,0]])
        shades[1].set_xy([[last+.5,0],[n+.5,0],[n+.5,1],[last+.5,1],[last+.5,0]])
        canvas.draw_idle()
        status.set(f'保留第 {first}–{last} 帧，共 {last-first+1} 帧；删除前 {first-1} 帧、后 {n-last} 帧。尚未保存。')

    shades = []
    selector = SpanSelector(ax, select_range, 'horizontal', useblit=False,
                            props=dict(alpha=.14, facecolor='#3485b8'), interactive=True)

    def show(index):
        nonlocal selector
        if state['busy'] or not state['folders']:
            return
        state['index'] = max(0, min(len(state['folders'])-1, index))
        folder = state['folders'][state['index']]
        csv_path=trial_csv(folder)
        state['bytes'] = csv_path.read_bytes()
        _, state['rows'] = read_csv(csv_path)
        n = len(state['rows'])
        y = np.array([float(r['finger_width']) if r['finger_width'] else np.nan for r in state['rows']])
        selector.disconnect_events()
        ax.clear()
        ax.plot(np.arange(1,n+1), y, color='#222222' if csv_path.name=='predictions.csv' else '#3485b8', lw=1.7,label='Measured')
        if csv_path.name=='predictions.csv':
            for key,label,color in zip(['predicted_function_px','predicted_mlp_px','predicted_lstm_px'],['Function','MLP','LSTM'],['#3485b8','#e59a32','#519e58']):
                pred=np.array([float(r[key]) if r.get(key) else np.nan for r in state['rows']])
                ax.plot(np.arange(1,n+1),pred,color=color,lw=1,label=label)
            ax.legend(fontsize=9)
        ax.set(title=str(folder.relative_to(state['base'])) if folder!=state['base'] else folder.name, xlabel='Frame (row position, 1-based)', ylabel='Fingertip aperture (px)', xlim=(.5,n+.5))
        ax.grid(alpha=.2)
        # Use patches with axis-fraction Y coordinates, robust to autoscaling.
        from matplotlib.patches import Polygon
        shades.clear()
        for _ in range(2):
            patch = Polygon([[.5,0],[.5,0],[.5,1],[.5,1]], transform=ax.get_xaxis_transform(),
                            color='gray', alpha=.2)
            ax.add_patch(patch)
            shades.append(patch)
        selector = SpanSelector(ax, select_range, 'horizontal', useblit=False,
                                props=dict(alpha=.14, facecolor='#3485b8'), interactive=True)
        figure.tight_layout()
        listing.selection_clear(0,'end')
        listing.selection_set(state['index'])
        listing.see(state['index'])
        select_range(*state['pending'].get(folder,(1,n)))

    def choose(path=None):
        if state['busy']:
            return
        path = path or filedialog.askdirectory(initialdir=str(state['base'] or DATA_ROOT))
        if not path:
            return
        base = Path(path)
        folders = discover_trials(base)
        if not folders:
            messagebox.showerror('未找到数据','请选择含 camera.csv 或 predictions.csv 的实验目录，或它们的上级目录；会自动查找 fast/normal/slow，跳过 backup。')
            return
        state.update(base=base, folders=folders, pending={})
        path_text.set(str(base))
        listing.delete(0,'end')
        for p in folders:
            listing.insert('end',str(p.relative_to(base)) if p!=base else p.name)
        show(0)

    def save_current():
        if state['busy'] or not state['folders']:
            return
        try:
            first, last = int(first_var.get()), int(last_var.get())
            if not 1 <= first < last <= len(state['rows']):
                raise ValueError()
        except ValueError:
            messagebox.showerror('帧范围','请输入有效的整数帧范围，至少保留两帧。')
            return
        folder = state['folders'][state['index']]
        snapshot = state['bytes']
        state['busy'] = True
        selector.set_active(False)
        status.set('正在准备视频、帧图片和备份；完成前请勿关闭窗口。')
        def work():
            try:
                result_queue.put(('ok',crop_trial(folder,first,last,snapshot)))
            except Exception as exc:
                result_queue.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll():
        if not result_queue.empty():
            kind, result = result_queue.get()
            state['busy'] = False
            selector.set_active(True)
            if kind == 'ok':
                state['pending'].pop(state['folders'][state['index']],None)
                show(state['index'])
                extra='实时预测图和指标已重算；原始选择排名未改。' if result['csv_name']=='predictions.csv' else '已有分析图需重新生成。'
                status.set(f"已保存：保留 {result['retained_frames']} 帧，删除 {result['removed_frames']} 帧。备份：{result['backup']}。{extra}")
            else:
                status.set('保存失败：'+result)
                messagebox.showerror('保存失败',result)
        window.after(100,poll)

    def close():
        if state['busy']:
            messagebox.showinfo('正在保存','请等待保存完成后再关闭。')
            return
        window.destroy()

    ttk.Button(top,text='选择数据文件夹',command=choose).pack(side='left')
    ttk.Label(top,textvariable=path_text).pack(side='left',padx=8)
    ttk.Button(controls,text='上一组',command=lambda:show(state['index']-1)).pack(side='left')
    ttk.Button(controls,text='下一组',command=lambda:show(state['index']+1)).pack(side='left',padx=4)
    ttk.Label(controls,text='保留起始帧').pack(side='left',padx=(15,3))
    ttk.Entry(controls,textvariable=first_var,width=7).pack(side='left')
    ttk.Label(controls,text='结束帧').pack(side='left',padx=(8,3))
    ttk.Entry(controls,textvariable=last_var,width=7).pack(side='left')
    def apply_range():
        try:
            select_range(float(first_var.get()),float(last_var.get()))
        except ValueError:
            messagebox.showerror('帧范围','请输入数字帧数。')
    ttk.Button(controls,text='应用范围',command=apply_range).pack(side='left',padx=5)
    ttk.Button(controls,text='重置选择',command=lambda:select_range(1,len(state['rows']))).pack(side='left',padx=4)
    ttk.Button(controls,text='保存当前裁剪（自动备份）',command=save_current).pack(side='right')
    listing.bind('<<ListboxSelect>>',lambda event:show(listing.curselection()[0])
                 if listing.curselection() and listing.curselection()[0] != state['index'] else None)
    window.protocol('WM_DELETE_WINDOW',close)
    window.after(100,poll)
    window.after(200,lambda:choose(initial) if initial else choose())
    window.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DATA_ROOT)
    parser.add_argument('--check-data',action='store_true',help='List recognized trials only; do not open GUI or crop')
    args = parser.parse_args()
    if args.check_data:
        folders=discover_trials(args.source)
        for folder in folders:print(str(folder.relative_to(args.source.resolve())),trial_csv(folder).name)
        print(f'Found {len(folders)} trials. No data changed.')
    else:
        launch(args.source)
