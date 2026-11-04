from pathlib import Path
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def move_trials(trials, groups, base):
    base = base.resolve()
    plans = []
    for i, trial in enumerate(trials):
        source = trial['path'].parent.resolve()
        home = trial['home'].resolve()
        destination = (home / source.name if groups[i] == 'usable' else home / '备用' / source.name).resolve()
        if source == destination:
            continue
        if not source.is_relative_to(base) or not destination.is_relative_to(base):
            raise ValueError('Folder outside the selected dataset')
        if destination.exists():
            raise ValueError(f'Destination already exists: {destination}')
        plans.append((i, source, destination))
    if len({destination for _, _, destination in plans}) != len(plans):
        raise ValueError('Duplicate destination folders')
    finished = []
    try:
        for i, source, destination in plans:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            finished.append((i, source, destination))
    except Exception:
        for i, source, destination in reversed(finished):
            shutil.move(str(destination), str(source))
        raise
    for i, source, destination in finished:
        trials[i]['path'] = destination / trials[i]['path'].name
    return len(finished)


def open_selector(base, filename, excluded_trials, output, dpi):
    from plot_folder_overlays import read_trials, SIGNALS, plot_group, mean_curve
    trials, _ = read_trials(base, filename, True, True, excluded_trials)
    trials = [t for t in trials if not any(p.casefold() in {'待检查', 'filter_comparison'}
              for p in t['path'].relative_to(base).parts[:-1])]
    if not trials:
        raise ValueError(f'没有可用的有符号加速度实验，请选择包含 forward_acceleration 或 forward_raw 列的新录制目录：{base}')
    groups = {}
    for i, trial in enumerate(trials):
        parent = trial['path'].parent.parent
        reserve = parent.name.casefold() in {'backup', '备用'}
        trial['home'] = parent.parent if reserve else parent
        groups[i] = 'reserve' if reserve else 'usable'
    saved_groups = groups.copy()
    selected = set()
    window = tk.Tk()
    window.title('实验曲线筛选：可用 / 备用')
    window.geometry('1400x900')
    window.minsize(1000, 650)
    top = ttk.Frame(window, padding=8)
    top.pack(fill='x')
    status = tk.StringVar(value='点击曲线或列表选择；Ctrl 点击多选；点击按钮切换分类；保存后移动文件夹')
    ttk.Label(top, textvariable=status).pack(side='left', fill='x', expand=True)
    body = ttk.Frame(window)
    body.pack(fill='both', expand=True)
    plot_frame = ttk.Frame(body)
    plot_frame.pack(side='left', fill='both', expand=True)
    panel = ttk.Frame(body, padding=8, width=250)
    panel.pack(side='right', fill='y')
    view = tk.StringVar(value='usable')
    ttk.Label(panel, text='叠加图显示').pack(anchor='w')
    for text, value in [('可用', 'usable'), ('备用', 'reserve'), ('全部', 'all')]:
        ttk.Radiobutton(panel, text=text, variable=view, value=value, command=lambda: redraw()).pack(anchor='w')
    ttk.Button(panel, text='导出当前平均线 CSV', command=lambda: export_mean()).pack(fill='x', pady=4)
    lists, headings, list_ids = {}, {}, {}
    for key, title in [('usable', '可用'), ('reserve', '备用')]:
        frame = ttk.LabelFrame(panel, text=title, padding=6)
        frame.pack(fill='both', expand=True, pady=6)
        headings[key] = frame
        box = tk.Listbox(frame, selectmode='extended', exportselection=False, width=30, height=13)
        scroll = ttk.Scrollbar(frame, command=box.yview)
        box.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        box.pack(fill='both', expand=True)
        lists[key] = box
    fig = Figure(figsize=(11, 8), dpi=100)
    axes = fig.subplots(3, 1)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    colors = __import__('matplotlib').colormaps['turbo'](np.linspace(.05, .95, len(trials)))
    lines = []
    for ax, (col, label) in zip(axes, SIGNALS):
        for i, trial in enumerate(trials):
            line, = ax.plot(trial['x'], trial['data'][col], color=colors[i], lw=1.2, picker=7)
            lines.append((line, i))
        ax.set(xlim=(0, 100), ylabel=label)
        if col == 'forward_acceleration':
            ax.axhline(0, color='black', lw=.8, alpha=.4)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Recorded trial progress (%)')
    fig.tight_layout()
    def update_selection_status():
        selected_groups = {groups[i] for i in selected}
        text = '移入备用' if selected_groups == {'usable'} else '移入可用' if selected_groups == {'reserve'} else '移入另一分类'
        move_button.configure(text=text, state='normal' if selected else 'disabled')
        status.set(f'已选 {len(selected)} 组 | Ctrl 点击多选，按钮切换分类 | {"有未保存分类" if groups != saved_groups else "分类已保存"}')

    def redraw():
        for key, box in lists.items():
            box.delete(0, 'end')
            ids = [i for i in range(len(trials)) if groups[i] == key]
            list_ids[key] = ids
            headings[key].configure(text=f'{"可用" if key == "usable" else "备用"} ({len(ids)})')
            for position, i in enumerate(ids):
                box.insert('end', trials[i]['label'])
                box.itemconfig(position, foreground=__import__('matplotlib').colors.to_hex(colors[i]))
                if i in selected:
                    box.selection_set(position)
        for line, i in lines:
            line.set_visible(view.get() == 'all' or groups[i] == view.get())
            line.set_linewidth(3 if i in selected else 1.2)
            line.set_alpha(1 if not selected or i in selected else .18)
        for ax in axes:
            ax.relim(visible_only=True)
            ax.autoscale_view(scalex=False)
        count = sum(view.get() == 'all' or group == view.get() for group in groups.values())
        axes[0].set_title(f'{count} experiments')
        canvas.draw_idle()
        update_selection_status()

    def export_mean():
        visible = [t for i, t in enumerate(trials) if view.get() == 'all' or groups[i] == view.get()]
        if not visible:
            status.set('当前分类没有实验，无法生成平均线')
            return
        try:
            output.mkdir(parents=True, exist_ok=True)
            path = output / f'mean_curve_{view.get()}.csv'
            mean_curve(visible).to_csv(path, index=False, encoding='utf-8-sig')
            status.set(f'已导出 {len(visible)} 组实验的平均线：{path}')
        except OSError as exc:
            messagebox.showerror('导出失败', str(exc), parent=window)

    def plot_press(event):
        if event.button != 1 or event.inaxes is None:
            return
        candidates = []
        for line, i in lines:
            if line.axes != event.inaxes or not line.get_visible():
                continue
            hit, details = line.contains(event)
            if hit:
                points = line.get_transform().transform(line.get_xydata())
                inds = details.get('ind', np.arange(len(points)))
                distance = np.min(np.linalg.norm(points[inds] - [event.x, event.y], axis=1))
                candidates.append((distance, i))
        if not candidates:
            return
        i = min(candidates)[1]
        control = event.key is not None and ('ctrl' in event.key or 'control' in event.key)
        if control:
            if i in selected:
                selected.remove(i)
            else:
                selected.add(i)
        elif i not in selected:
            selected.clear()
            selected.add(i)
        redraw()

    def list_select(key):
        selected.clear()
        selected.update(list_ids[key][n] for n in lists[key].curselection())
        for other_key, box in lists.items():
            if other_key != key:
                box.selection_clear(0, 'end')
        for line, i in lines:
            line.set_linewidth(3 if i in selected else 1.2)
            line.set_alpha(1 if not selected or i in selected else .18)
        canvas.draw_idle()
        update_selection_status()

    def move_selected():
        if not selected:
            return
        for i in selected:
            groups[i] = 'reserve' if groups[i] == 'usable' else 'usable'
        destinations = {groups[i] for i in selected}
        if view.get() != 'all' and len(destinations) == 1:
            view.set(next(iter(destinations)))
        redraw()

    def save():
        try:
            count = move_trials(trials, groups, base)
            saved_groups.clear()
            saved_groups.update(groups)
            usable = [t for i, t in enumerate(trials) if groups[i] == 'usable']
            if usable:
                plot_group(usable, output, 'Smooth', dpi)
            redraw()
            status.set(f'已保存：可用 {len(usable)} 组，备用 {len(trials)-len(usable)} 组；移动 {count} 个文件夹')
        except Exception as exc:
            messagebox.showerror('保存失败', str(exc), parent=window)

    def close():
        if groups != saved_groups and not messagebox.askyesno('未保存分类', '关闭并放弃尚未保存的分类？', parent=window):
            return
        window.destroy()

    ttk.Button(top, text='保存分类', command=save).pack(side='right')
    ttk.Button(top, text='清空选择', command=lambda: (selected.clear(), redraw())).pack(side='right', padx=8)
    move_button = ttk.Button(top, text='移入另一分类', command=move_selected, state='disabled')
    move_button.pack(side='right')
    for key, box in lists.items():
        box.bind('<<ListboxSelect>>', lambda event, key=key: list_select(key))
    canvas.mpl_connect('button_press_event', plot_press)
    window.protocol('WM_DELETE_WINDOW', close)
    redraw()
    window.mainloop()
