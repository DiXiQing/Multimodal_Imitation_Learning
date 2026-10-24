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
DATA_ROOT = ROOT/'Data/D2/fast'
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


def crop_trial(folder, first, last, expected_csv):
    """1-based inclusive selection. Prepare/verify first, then swap; no source deletion."""
    folder = Path(folder).resolve()
    if (folder/'camera.csv').read_bytes() != expected_csv:
        raise ValueError('数据已被其他程序修改，请重新打开文件夹。')
    fields, rows = read_csv(folder/'camera.csv')
    if not 1 <= first < last <= len(rows):
        raise ValueError('请选择至少两帧，且范围不能超出原始帧数。')
    if first == 1 and last == len(rows):
        raise ValueError('当前选择包含全部帧，无需裁剪。')
    kept = [dict(r) for r in rows[first-1:last]]
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
    write_csv(stage/'camera.csv', fields, kept)
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
    _, verified = read_csv(stage/'camera.csv')
    assert verified == kept
    # Recheck before replacing: edits made during preparation must not be lost.
    if (folder/'camera.csv').read_bytes() != expected_csv:
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
                    backup=str(original), note='Only segment-relative CSV fields regenerated; other retained values unchanged. Existing analysis plots/results require regeneration.')
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
        state['bytes'] = (folder/'camera.csv').read_bytes()
        _, state['rows'] = read_csv(folder/'camera.csv')
        n = len(state['rows'])
        y = np.array([float(r['finger_width']) if r['finger_width'] else np.nan for r in state['rows']])
        selector.disconnect_events()
        ax.clear()
        ax.plot(np.arange(1,n+1), y, color='#3485b8', lw=1.7)
        ax.set(title=folder.name, xlabel='Frame', ylabel='Fingertip aperture (px)', xlim=(.5,n+.5))
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
        folders = [base] if (base/'camera.csv').exists() else sorted(p.parent for p in base.glob('*/camera.csv'))
        if not folders:
            messagebox.showerror('未找到数据','请选择直接包含实验子文件夹的目录，例如 D2/fast 或 D2/slow。')
            return
        state.update(base=base, folders=folders, pending={})
        path_text.set(str(base))
        listing.delete(0,'end')
        for p in folders:
            listing.insert('end',p.name)
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
                status.set(f"已保存：保留 {result['retained_frames']} 帧，删除 {result['removed_frames']} 帧。备份：{result['backup']}。已有分析图需重新生成。")
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
    parser.add_argument('--source',type=Path,default=None)
    args = parser.parse_args()
    launch(args.source)
