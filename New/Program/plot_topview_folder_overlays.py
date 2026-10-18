"""Manual top-view curve selection: retained / backup. No training or smoothing."""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import hashlib
import numpy as np
from matplotlib.figure import Figure

DATA_ROOT = Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning\New\Data\D1\3')


def read_trials(base):
    trials = []
    # Immediate trial folders only; also read backup to allow restoring them.
    for home, group in [(base,'usable'),(base/'backup','reserve')]:
        if not home.is_dir():continue
        for folder in sorted(home.iterdir()):
            path = folder/'camera.csv'
            if not folder.is_dir() or not path.is_file() or not (folder/'segment.json').is_file():continue
            with path.open(encoding='utf-8-sig') as f:rows = list(csv.DictReader(f))
            if not rows:continue
            def values(key):
                result = []
                for row in rows:
                    try:result.append(float(row[key]))
                    except (KeyError,ValueError,TypeError):result.append(np.nan)
                return np.asarray(result)
            d,b,frame = values('hand_object_distance'),values('finger_width'),values('frame')
            valid = np.array([r.get('valid','1')=='1' for r in rows])&np.isfinite(d)&np.isfinite(b)
            d[~valid] = np.nan;b[~valid] = np.nan
            if not valid.any():print('[Skip]',folder.name,'no valid measurements');continue
            if not np.isfinite(frame).all() or np.any(np.diff(frame)<=0):
                raise ValueError(f'{folder.name}: invalid source frame sequence')
            trials.append(dict(path=folder.resolve(),label=folder.name,group=group,
                frame=frame-frame[0],distance=d,aperture=b,rows=len(rows),invalid=int((~valid).sum())))
    if not trials:raise ValueError(f'No camera.csv / segment.json trials found: {base}')
    return trials


def plot_lines(fig,trials,groups,view='usable'):
    axes = fig.subplots(3,1)
    colors = __import__('matplotlib').colormaps['turbo'](np.linspace(.03,.97,len(trials)))
    lines = []
    for i,(trial,color) in enumerate(zip(trials,colors)):
        for ax,x,y in [(axes[0],trial['distance'],trial['aperture']),
                       (axes[1],trial['frame'],trial['aperture']),
                       (axes[2],trial['frame'],trial['distance'])]:
            line, = ax.plot(x,y,color=color,lw=1.1,picker=6)
            line.set_visible(view=='all' or groups[i]==view)
            lines.append((line,i))
    axes[0].set(xlabel='Hand-object distance (px)',ylabel='Aperture (px)')
    axes[1].set(xlabel='Source frames from segment start',ylabel='Aperture (px)')
    axes[2].set(xlabel='Source frames from segment start',ylabel='Hand-object distance (px)')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout()
    return axes,lines


def export_plot(trials,groups,view,output,dpi):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    output.mkdir(parents=True,exist_ok=True)
    fig = Figure(figsize=(11,11),dpi=dpi);FigureCanvasAgg(fig)
    axes,_ = plot_lines(fig,trials,groups,view)
    count = sum(view=='all' or groups[i]==view for i in range(len(trials)))
    axes[0].set_title(f'Top view | {count} trials | {view} | raw measurements')
    fig.savefig(output/f'overlay_{view}.png',dpi=dpi)
    fig.savefig(output/f'overlay_{view}.pdf')
    with (output/'trial_mapping.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['trial','folder','group','path','frames','invalid_frames'])
        writer.writeheader()
        writer.writerows(dict(trial=i+1,folder=t['label'],group=groups[i],path=str(t['path']),
            frames=t['rows'],invalid_frames=t['invalid']) for i,t in enumerate(trials))


def save_groups(base,trials,groups,output):
    base=base.resolve();backup=(base/'backup').resolve();plan=[]
    for i,t in enumerate(trials):
        src=t['path'].resolve();dest=((base if groups[i]=='usable' else backup)/src.name).resolve()
        if src==dest:continue
        if src.parent not in (base,backup) or dest.parent not in (base,backup):
            raise ValueError('Move would leave the selected dataset')
        if not src.is_dir() or dest.exists():raise ValueError(f'Source/destination conflict: {src} -> {dest}')
        hashes={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file()}
        plan.append((i,src,dest,hashes))
    if len({p[2] for p in plan})!=len(plan):raise ValueError('Duplicate destination folders')
    completed=[]
    try:
        for i,src,dest,hashes in plan:
            dest.parent.mkdir(exist_ok=True);src.rename(dest);completed.append((i,src,dest,hashes))
            after={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
            if hashes!=after:raise ValueError(f'File verification failed: {dest}')
        output.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        (output/f'manual_selection_{stamp}.json').write_text(json.dumps(dict(
            basis='User manual curve selection; not confirmation of measurement defects.',
            trials=[dict(folder=t['label'],group=groups[i]) for i,t in enumerate(trials)],
            moves=[dict(original=str(src),destination=str(dest),file_hashes=h) for _,src,dest,h in completed]),indent=2),encoding='utf-8')
    except Exception:
        for _,src,dest,_ in reversed(completed):dest.rename(src)
        raise
    for i,src,dest,h in completed:trials[i]['path']=dest
    return len(completed)


def open_selector(base,trials,output,dpi):
    import tkinter as tk
    from tkinter import ttk,messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
    groups={i:t['group'] for i,t in enumerate(trials)};saved=groups.copy();selected=set()
    root=tk.Tk();root.title('俯视抓握曲线筛选：可用 / backup')
    root.geometry(f'{min(1400,root.winfo_screenwidth()-60)}x{min(900,root.winfo_screenheight()-100)}+20+20')
    root.minsize(900,600)
    status=tk.StringVar(value='')
    ttk.Label(root,text=str(base),wraplength=1000).pack(anchor='w',padx=8)
    ttk.Label(root,textvariable=status).pack(fill='x',padx=8,pady=4)
    body=ttk.Frame(root);body.pack(fill='both',expand=True)
    plot=ttk.Frame(body);plot.pack(side='left',fill='both',expand=True)
    panel=ttk.Frame(body,padding=8);panel.pack(side='right',fill='y')
    view=tk.StringVar(value='usable')
    for text,value in [('显示可用','usable'),('显示 backup','reserve'),('显示全部','all')]:
        ttk.Radiobutton(panel,text=text,variable=view,value=value,command=lambda:redraw()).pack(anchor='w')
    boxes={};ids={};headings={}
    for key,title in [('usable','可用'),('reserve','backup')]:
        frame=ttk.LabelFrame(panel,text=title,padding=4);frame.pack(fill='both',expand=True,pady=5);headings[key]=frame
        box=tk.Listbox(frame,selectmode='extended',exportselection=False,width=34,height=10)
        scrollbar=ttk.Scrollbar(frame,command=box.yview);box.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right',fill='y');box.pack(fill='both',expand=True);boxes[key]=box
    fig=Figure(figsize=(10,10),dpi=100);axes,lines=plot_lines(fig,trials,groups)
    canvas=FigureCanvasTkAgg(fig,master=plot);canvas.get_tk_widget().pack(fill='both',expand=True)
    toolbar=NavigationToolbar2Tk(canvas,plot);toolbar.update()
    def update_status():
        names=', '.join(trials[i]['label'] for i in sorted(selected))
        status.set(f'选中 {len(selected)} 组：{names[:130]} | '+('有未保存分类' if groups!=saved else '分类已保存'))
    def redraw():
        for key,box in boxes.items():
            box.delete(0,'end');ids[key]=[i for i in groups if groups[i]==key]
            headings[key].configure(text=f'{"可用" if key=="usable" else "backup"} ({len(ids[key])})')
            for position,i in enumerate(ids[key]):
                box.insert('end',trials[i]['label'])
                if i in selected:box.selection_set(position)
        for line,i in lines:
            line.set_visible(view.get()=='all' or groups[i]==view.get())
            line.set_linewidth(2.8 if i in selected else 1.1)
            line.set_alpha(1 if not selected or i in selected else .15)
        for ax in axes:ax.relim(visible_only=True);ax.autoscale_view()
        axes[0].set_title(f'{sum(view.get()=="all" or g==view.get() for g in groups.values())} trials | raw measurements')
        canvas.draw_idle();update_status()
    def list_select(key):
        chosen={ids[key][position] for position in boxes[key].curselection()}
        if not chosen or chosen==selected:return
        selected.clear();selected.update(chosen)
        for other,box in boxes.items():
            if other!=key:box.selection_clear(0,'end')
        for line,i in lines:line.set_linewidth(2.8 if i in selected else 1.1);line.set_alpha(1 if not selected or i in selected else .15)
        canvas.draw_idle();update_status()
    def click(event):
        if event.button!=1 or event.inaxes is None or toolbar.mode:return
        candidates=[]
        for line,i in lines:
            if line.axes!=event.inaxes or not line.get_visible():continue
            hit,detail=line.contains(event)
            if hit:
                points=line.get_transform().transform(line.get_xydata());idx=detail.get('ind',[])
                if len(idx):candidates.append((float(np.nanmin(np.linalg.norm(points[idx]-[event.x,event.y],axis=1))),i))
        if not candidates:return
        i=min(candidates)[1]
        if event.key and ('ctrl' in event.key or 'control' in event.key):
            if i in selected:selected.remove(i)
            else:selected.add(i)
        else:selected.clear();selected.add(i)
        redraw()
    def assign(group):
        for i in selected:groups[i]=group
        redraw()
    def save():
        try:
            moved=save_groups(base,trials,groups,output);saved.clear();saved.update(groups)
            for mode in ['usable','reserve','all']:export_plot(trials,groups,mode,output,dpi)
            redraw();status.set(f'保存成功：移动 {moved} 组。图和清单：{output}')
        except Exception as exc:messagebox.showerror('保存失败',str(exc),parent=root)
    def export():
        try:export_plot(trials,groups,view.get(),output,dpi);status.set(f'已导出当前视图：{output}（未移动文件夹）')
        except Exception as exc:messagebox.showerror('导出失败',str(exc),parent=root)
    def close():
        if groups!=saved and not messagebox.askyesno('未保存分类','放弃未保存的分类并退出？',parent=root):return
        root.destroy()
    for key,box in boxes.items():box.bind('<<ListboxSelect>>',lambda event,k=key:list_select(k))
    ttk.Button(panel,text='选中 → backup',command=lambda:assign('reserve')).pack(fill='x',pady=4)
    ttk.Button(panel,text='选中 → 可用（恢复）',command=lambda:assign('usable')).pack(fill='x',pady=4)
    ttk.Button(panel,text='清空选择',command=lambda:(selected.clear(),redraw())).pack(fill='x',pady=4)
    ttk.Button(panel,text='保存分类并更新叠加图',command=save).pack(fill='x',pady=8)
    ttk.Button(panel,text='仅导出当前视图',command=export).pack(fill='x',pady=4)
    ttk.Label(panel,text='点击曲线或列表选择，Ctrl 多选。\n分类按钮仅改变预览；保存才移动。\n原始数值不平滑、不归一化。\n曲线差异不等同于检测错误。',wraplength=270).pack(pady=8)
    canvas.mpl_connect('button_press_event',click);root.protocol('WM_DELETE_WINDOW',close)
    redraw();root.mainloop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--static',action='store_true',help='Export plots only; never move trials')
    parser.add_argument('--dpi',type=int,default=160)
    args=parser.parse_args();base=args.data_root.resolve()
    if not base.is_dir():parser.error(f'Dataset not found: {base}')
    try:trials=read_trials(base)
    except (ValueError,OSError) as exc:parser.error(str(exc))
    output=args.output_dir or base/'manual_overlay_plots'
    if args.static:
        groups={i:t['group'] for i,t in enumerate(trials)}
        for view in ['usable','reserve','all']:export_plot(trials,groups,view,output,args.dpi)
        print(f'Exported {len(trials)} trials (including backup); no moves. {output}')
    else:open_selector(base,trials,output,args.dpi)


if __name__=='__main__':main()
