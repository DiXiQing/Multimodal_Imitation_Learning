"""Manual review of 1101 timestamp trials: three linked overlays, retained / backup. No training or smoothing."""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import hashlib
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import numpy as np
from matplotlib.figure import Figure

DATA_ROOT = r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\1101_D\30mm'


def read_trials(base, distance_mode=None):
    import re
    base=base.resolve();trials=[]
    group_file=base/'overlays_two_recordings/trial_recording_groups.csv'
    recording_map={}
    if group_file.is_file():
        with group_file.open(encoding='utf-8-sig',newline='') as f:
            recording_map={r['trial']:r['recording'] for r in csv.DictReader(f)}
    for home,group in [(base,'usable'),(base/'backup','reserve')]:
        if not home.is_dir():continue
        for folder in sorted(home.iterdir()):
            if not folder.is_dir() or not re.fullmatch(r'(?:\d{8}_\d{6}|grasp_\d+)',folder.name):continue
            path=folder/'camera.csv'
            if not path.is_file() or not (folder/'segment.json').is_file():continue
            settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'))
            scale=(settings.get('pixel_calibration') or {}).get('mm_per_px')
            if scale is None or not np.isfinite(float(scale)) or float(scale)<=0:
                raise ValueError(f'{folder.name}: missing valid mm calibration; px fallback forbidden')
            with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            if not rows:raise ValueError(f'{folder.name}: empty CSV')
            required=['index_mcp_object_distance_mm','finger_width_mm','frame','t_camera','valid','index_mcp_distance_valid']
            if any(k not in rows[0] for k in required):raise ValueError(f'{folder.name}: missing required mm/detection/timestamp columns')
            def values(key):
                result=[]
                for r in rows:
                    try:result.append(float(r[key]))
                    except (ValueError,TypeError,KeyError):result.append(np.nan)
                return np.asarray(result)
            d,a,frame,t=map(values,('index_mcp_object_distance_mm','finger_width_mm','frame','t_camera'))
            if not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError(f'{folder.name}: invalid camera timestamps')
            if not np.isfinite(frame).all() or np.any(np.diff(frame)<=0):raise ValueError(f'{folder.name}: invalid source frame IDs')
            valid=np.array([r['valid']=='1' and r['index_mcp_distance_valid']=='1' for r in rows])&np.isfinite(d)&np.isfinite(a)
            if not valid[0] or d[0]<=0:raise ValueError(f'{folder.name}: first frame cannot define initial distance')
            # Read finger_width_mm directly; manually corrected targets need not equal px * scale.
            for mm,px in [('index_mcp_object_distance_mm','index_mcp_object_distance')]:
                if px in rows[0]:
                    raw=values(px);measured=values(mm);mask=np.isfinite(raw)&np.isfinite(measured)
                    if not np.allclose(measured[mask],raw[mask]*float(scale),rtol=1e-5,atol=1e-4):raise ValueError(f'{folder.name}: inconsistent {mm} calibration')
            if all(k in rows[0] for k in ('index_mcp_x','index_mcp_y','object_x','object_y')):
                expected=np.hypot(values('index_mcp_x')-values('object_x'),values('index_mcp_y')-values('object_y'))*float(scale)
                mask=np.isfinite(expected)&np.isfinite(d)
                if not np.allclose(expected[mask],d[mask],rtol=1e-5,atol=1e-4):raise ValueError(f'{folder.name}: input is not index MCP Euclidean distance')
            initial=float(d[0]);d=d.copy();a=a.copy();d[~valid]=np.nan;a[~valid]=np.nan
            # Insert display-only gaps; original samples and measurements stay intact.
            gaps=np.flatnonzero((np.diff(frame)!=1)|(np.diff(t)>.1))+1
            def display(v):return np.insert(v,gaps,np.nan)
            trials.append(dict(path=folder.resolve(),label=folder.name,group=group,
                time=display(t-t[0]),distance=display(d),normalized_distance=display(d/initial),aperture=display(a),
                initial_distance_mm=initial,recording=recording_map.get(folder.name,'Other recording'),
                distance_key='index_mcp_object_distance_mm',rows=len(rows),invalid=int((~valid).sum())))
    if not trials:raise ValueError(f'No trials found in dataset root / direct backup: {base}')
    if len({t['label'] for t in trials})!=len(trials):raise ValueError('Duplicate trial name in dataset and backup; resolve before moving')
    return trials


def plot_lines(fig,trials,groups,view='usable'):
    from matplotlib.lines import Line2D
    axes=fig.subplots(1,3)
    names=sorted({t['recording'] for t in trials})
    palette=['#287db5','#e58b26','#7c6bad','#888888']
    colors={name:palette[i%len(palette)] for i,name in enumerate(names)}
    lines=[]
    for i,trial in enumerate(trials):
        for ax,x,y in [(axes[0],trial['normalized_distance'],trial['aperture']),
                       (axes[1],trial['time'],trial['distance']),
                       (axes[2],trial['time'],trial['aperture'])]:
            line,=ax.plot(x,y,color=colors[trial['recording']],lw=1.0,alpha=.45,picker=6)
            line.set_visible(view=='all' or groups[i]==view);lines.append((line,i))
    axes[0].set(xlabel=r'Normalized distance $l_t/l_0$',ylabel='Finger aperture (mm)')
    axes[1].set(xlabel='Time (s)',ylabel='Index MCP–object distance (mm)')
    axes[2].set(xlabel='Time (s)',ylabel='Finger aperture (mm)')
    for ax in axes:ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([0],[0],color=colors[name],lw=2,label=name) for name in names],loc='upper center',ncol=min(len(names),3),fontsize=9)
    fig.tight_layout(rect=(0,0,1,.91))
    return axes,lines


def export_plot(trials,groups,view,output,dpi):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    output.mkdir(parents=True,exist_ok=True)
    fig = Figure(figsize=(16,5.2),dpi=dpi);FigureCanvasAgg(fig)
    axes,_ = plot_lines(fig,trials,groups,view)
    count = sum(view=='all' or groups[i]==view for i in range(len(trials)))
    axes[0].set_title(f'Top view | {count} trials | {view} | raw measurements')
    fig.savefig(output/f'overlay_{view}.png',dpi=dpi)
    fig.savefig(output/f'overlay_{view}.pdf')
    fig.savefig(output/f'overlay_{view}.svg')
    with (output/'trial_mapping.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['trial','folder','group','path','frames','invalid_frames','distance_column','recording','initial_distance_mm'])
        writer.writeheader()
        writer.writerows(dict(trial=i+1,folder=t['label'],group=groups[i],path=str(t['path']),
            frames=t['rows'],invalid_frames=t['invalid'],distance_column=t['distance_key'],recording=t['recording'],initial_distance_mm=t['initial_distance_mm']) for i,t in enumerate(trials))


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
    root.geometry(f'{min(1800,root.winfo_screenwidth()-60)}x{min(950,root.winfo_screenheight()-100)}+20+20')
    root.minsize(1000,680)
    root.columnconfigure(0,weight=1)
    root.rowconfigure(2,weight=1,minsize=300)
    status=tk.StringVar(value='')
    path_label=ttk.Label(root,text=str(base),wraplength=1000)
    path_label.grid(row=0,column=0,sticky='ew',padx=8)
    status_label=ttk.Label(root,textvariable=status,wraplength=1000)
    status_label.grid(row=1,column=0,sticky='ew',padx=8,pady=4)
    plot=ttk.Frame(root);plot.grid(row=2,column=0,sticky='nsew')
    # Reserve the bottom controls independently of the canvas requested width.
    panel=ttk.Frame(root,padding=(8,4));panel.grid(row=3,column=0,sticky='ew')
    panel.columnconfigure(0,weight=1)
    panel.columnconfigure(1,weight=1)
    views=ttk.Frame(panel);views.grid(row=0,column=0,columnspan=3,sticky='w',pady=(0,3))
    view=tk.StringVar(value='usable')
    for text,value in [('显示可用','usable'),('显示 backup','reserve'),('显示全部','all')]:
        ttk.Radiobutton(views,text=text,variable=view,value=value,command=lambda:redraw()).pack(side='left',padx=(0,18))
    boxes={};ids={};headings={}
    for column,(key,title) in enumerate([('usable','可用'),('reserve','backup')]):
        frame=ttk.LabelFrame(panel,text=title,padding=4)
        frame.grid(row=1,column=column,sticky='nsew',padx=(0,8));headings[key]=frame
        box=tk.Listbox(frame,selectmode='extended',exportselection=False,width=20,height=4)
        scrollbar=ttk.Scrollbar(frame,command=box.yview);box.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right',fill='y');box.pack(fill='both',expand=True);boxes[key]=box
    actions=ttk.Frame(panel);actions.grid(row=1,column=2,sticky='nsew')
    actions.columnconfigure(0,weight=1);actions.columnconfigure(1,weight=1)
    fig=Figure(figsize=(15,5.2),dpi=100);axes,lines=plot_lines(fig,trials,groups)
    canvas=FigureCanvasTkAgg(fig,master=plot)
    toolbar=NavigationToolbar2Tk(canvas,plot,pack_toolbar=False)
    toolbar.pack(side='bottom',fill='x');toolbar.update()
    canvas.get_tk_widget().pack(side='top',fill='both',expand=True)
    def fit_header(event):
        if event.widget==root:
            path_label.configure(wraplength=max(300,event.width-20))
            status_label.configure(wraplength=max(300,event.width-20))
    root.bind('<Configure>',fit_header)
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
            line.set_alpha(1 if i in selected else (.45 if not selected else .10))
        for ax in axes:ax.relim(visible_only=True);ax.autoscale_view()
        axes[0].set_title(f'{sum(view.get()=="all" or g==view.get() for g in groups.values())} trials | raw measurements')
        canvas.draw_idle();update_status()
    def list_select(key):
        chosen={ids[key][position] for position in boxes[key].curselection()}
        if not chosen or chosen==selected:return
        selected.clear();selected.update(chosen)
        for other,box in boxes.items():
            if other!=key:box.selection_clear(0,'end')
        for line,i in lines:line.set_linewidth(2.8 if i in selected else 1.1);line.set_alpha(1 if i in selected else (.45 if not selected else .10))
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
            status.set('正在保存和校验文件，请稍候…');root.update_idletasks()
            moved=save_groups(base,trials,groups,output);saved.clear();saved.update(groups)
        except Exception as exc:
            messagebox.showerror('分类保存失败',str(exc),parent=root);return
        try:
            for mode in ['usable','reserve','all']:export_plot(trials,groups,mode,output,dpi)
            redraw();status.set(f'保存成功：移动 {moved} 组。图和清单：{output}')
        except Exception as exc:
            redraw();messagebox.showerror('分类已保存，图表导出失败',f'已完成 {moved} 组移动。可点击仅导出重试。\n{exc}',parent=root)
    def export():
        try:export_plot(trials,groups,view.get(),output,dpi);status.set(f'已导出当前视图：{output}（未移动文件夹）')
        except Exception as exc:messagebox.showerror('导出失败',str(exc),parent=root)
    def close():
        if groups!=saved and not messagebox.askyesno('未保存分类','放弃未保存的分类并退出？',parent=root):return
        root.destroy()
    for key,box in boxes.items():box.bind('<<ListboxSelect>>',lambda event,k=key:list_select(k))
    ttk.Button(actions,text='选中 → backup（预览）',command=lambda:assign('reserve')).grid(row=0,column=0,sticky='ew',padx=3,pady=3)
    ttk.Button(actions,text='选中 → 可用（恢复）',command=lambda:assign('usable')).grid(row=0,column=1,sticky='ew',padx=3,pady=3)
    ttk.Button(actions,text='清空选择',command=lambda:(selected.clear(),redraw())).grid(row=1,column=0,sticky='ew',padx=3,pady=3)
    ttk.Button(actions,text='保存分类并移动文件夹',command=save).grid(row=1,column=1,sticky='ew',padx=3,pady=3)
    ttk.Button(actions,text='仅导出当前视图',command=export).grid(row=2,column=0,sticky='ew',padx=3,pady=3)
    def open_video():
        import os
        if len(selected)!=1:messagebox.showinfo('选择一组','请先只选择一组实验。',parent=root);return
        p=trials[next(iter(selected))]['path']/'grasp_video.mp4'
        if p.is_file():os.startfile(str(p))
        else:messagebox.showerror('视频不存在',str(p),parent=root)
    ttk.Button(actions,text='打开选中实验视频',command=open_video).grid(row=2,column=1,sticky='ew',padx=3,pady=3)
    ttk.Label(panel,text='点击曲线或列表选择，Ctrl 多选；分类按钮仅预览，保存才移动。左图距离按初始值归一化，时间与开合未归一化。',wraplength=960).grid(row=2,column=0,columnspan=3,sticky='w',pady=(4,0))
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
    output=args.output_dir or base/'manual_review_plots'
    if args.static:
        groups={i:t['group'] for i,t in enumerate(trials)}
        for view in ['usable','reserve','all']:export_plot(trials,groups,view,output,args.dpi)
        print(f'Exported {len(trials)} trials (including backup); no moves. {output}')
    else:open_selector(base,trials,output,args.dpi)


if __name__=='__main__':main()
