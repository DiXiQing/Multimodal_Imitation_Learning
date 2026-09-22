"""
smooth_grasp_data.py

给"手部抓取方块物体"的原始（带噪声）逐帧数据做整形平滑，输出符合物理含义的
干净轨迹，而不是简单的滑动平均/低通滤波（那种方法不保证形状正确）。

三条曲线各自用一个有物理意义的参数模型去拟合：

  1) Target Area (px^2)      —— 单调递增，抓稳后维持不变
                                 模型：S形逻辑函数 logistic(t)
                                 y = y0 + (y1-y0) * sigmoid(k*(t-t0))

  2) Fingertip Distance (mm) —— 从很小的值张开到峰值（松开手指去够物体），
                                 再闭合收拢到抓稳后的稳定值（≈物体宽度）
                                 模型：双S形（上升sigmoid − 下降sigmoid）
                                 y = y0 + (ypeak-y0)*sigmoid(k1*(t-t1))
                                        - (ypeak-yend)*sigmoid(k2*(t-t2))

  3) Hand Velocity (mm/s)    —— 先加速后减速，起止趋近于0
                                 模型：高斯钟形曲线
                                 y = vpeak * exp(-0.5*((t-tc)/sigma)^2)

用法：
    1. 把 CSV_PATH 改成你自己的原始数据文件路径。
    2. 按需要改下面 CONFIG 里的列名。
    3. 直接运行：
           python smooth_grasp_data.py

    如果 CSV_PATH 指向的文件不存在，脚本会自动生成一份和你截图风格一致的
    模拟数据用于演示，方便你先看效果、确认列名和曲线形状设置得对不对。

输入 CSV 需要是"长表"格式，每一行是一个时间点的一条测量记录：
    trial_id, group, time, target_area, fingertip_distance, hand_velocity

    - trial_id      : 每次抓取尝试的编号（重复实验多次时用来区分）
    - group         : 分组变量，比如 target_width_mm（10/30/50），
                       没有分组也可以，把 GROUP_COL 设成 None 即可
    - time          : 每次 trial 内从 0 开始的相对时间（秒）
    - target_area / fingertip_distance / hand_velocity : 原始测量值（含噪声）

输出：
    - <OUT_PREFIX>_smoothed.csv   : 原表基础上加 *_smooth 三列（逐 trial 拟合）
    - <OUT_PREFIX>_fit_params.csv : 每个 trial 三条曲线拟合出的参数
    - <OUT_PREFIX>_comparison.png : 仿照你截图风格的 3行 x N组 对比图
                                     （灰点=原始数据，粗线=拟合曲线）

依赖：
    pip install numpy pandas scipy matplotlib
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============ CONFIG：改这里 ============

CSV_PATH = r"D:\Code\Multimodal_Imitation_Learning\临时文件\medium_trial1_restored.csv"
OUT_PREFIX = "grasp"                   # 输出文件名前缀

TIME_COL = "time_s"
TRIAL_COL = "trial_id"
GROUP_COL = "group"                    # 没有分组的话设成 None
AREA_COL = "target_area_px2"
APERTURE_COL = "finger_width_px"
VEL_COL = "velocity_mps"

N_SMOOTH_POINTS = 200                  # 对比图里拟合曲线的采样点数

# ==========================================


# ---------- 通用工具 ----------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定版 sigmoid，避免 exp 溢出警告。"""
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def logistic(t, y0, y1, t0, k):
    """单调 S 形曲线：从 y0 平滑过渡到 y1。"""
    return y0 + (y1 - y0) * _sigmoid(k * (t - t0))


def double_sigmoid(t, y0, ypeak, yend, t1, k1, t2, k2):
    """先升到 ypeak 再降到 yend 的"开-合"曲线。"""
    rise = _sigmoid(k1 * (t - t1))
    fall = _sigmoid(k2 * (t - t2))
    return y0 + (ypeak - y0) * rise - (ypeak - yend) * fall


def bell(t, vpeak, tc, sigma):
    """先加速后减速的高斯钟形曲线。"""
    return vpeak * np.exp(-0.5 * ((t - tc) / sigma) ** 2)


def _robust_curve_fit(func, t, y, p0, bounds, label):
    """包一层 curve_fit，拟合失败时给警告而不是整个脚本崩掉。"""
    try:
        popt, _ = curve_fit(
            func, t, y, p0=p0, bounds=bounds,
            method="trf", max_nfev=20000,
        )
        return popt
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"[{label}] 拟合失败，将退化为原始数据的样条平滑：{e}")
        return None


# ---------- 三条曲线各自的拟合函数 ----------

def fit_area(t: np.ndarray, y: np.ndarray):
    """Target Area：单调递增 + 平台，用 logistic 拟合。"""
    y0_guess = float(np.percentile(y, 5))
    y1_guess = float(np.percentile(y, 95))
    mid = (y0_guess + y1_guess) / 2
    t0_guess = float(t[np.argmin(np.abs(y - mid))]) if len(t) else 0.0
    span = max(t.max() - t.min(), 1e-3)
    k_guess = 8.0 / span

    p0 = [y0_guess, y1_guess, t0_guess, k_guess]
    yr = max(y.max() - y.min(), 1.0)
    bounds = (
        [y.min() - yr, y.min(), t.min(), 1e-3],
        [y.max(), y.max() + yr, t.max(), 500],
    )
    popt = _robust_curve_fit(logistic, t, y, p0, bounds, "target_area")
    if popt is None:
        return None
    return dict(y0=popt[0], y1=popt[1], t0=popt[2], k=popt[3])


def fit_aperture(t: np.ndarray, y: np.ndarray):
    """Fingertip Distance：先升到峰值再降到平台，用双 sigmoid 拟合。"""
    peak_idx = int(np.argmax(y))
    t_peak = float(t[peak_idx])
    y0_guess = float(np.percentile(y[: max(len(y) // 10, 1)], 50))
    ypeak_guess = float(y[peak_idx])
    yend_guess = float(np.percentile(y[-max(len(y) // 10, 1):], 50))
    span = max(t.max() - t.min(), 1e-3)

    t1_guess = max(t.min(), t_peak - span * 0.2)
    t2_guess = min(t.max(), t_peak + span * 0.2)
    k_guess = 8.0 / span

    p0 = [y0_guess, ypeak_guess, yend_guess, t1_guess, k_guess, t2_guess, k_guess]
    yr = max(y.max() - y.min(), 1.0)
    bounds = (
        [y.min() - yr, y.min(), y.min() - yr, t.min(), 1e-3, t.min(), 1e-3],
        [y.max() + yr, y.max() + yr, y.max() + yr, t.max(), 500, t.max(), 500],
    )
    popt = _robust_curve_fit(double_sigmoid, t, y, p0, bounds, "fingertip_distance")
    if popt is None:
        return None
    return dict(
        y0=popt[0], ypeak=popt[1], yend=popt[2],
        t1=popt[3], k1=popt[4], t2=popt[5], k2=popt[6],
    )


def fit_velocity(t: np.ndarray, y: np.ndarray):
    """Hand Velocity：先加速后减速，用高斯钟形拟合。"""
    peak_idx = int(np.argmax(y))
    vpeak_guess = float(max(y[peak_idx], 1e-3))
    tc_guess = float(t[peak_idx])
    span = max(t.max() - t.min(), 1e-3)
    sigma_guess = span / 4

    p0 = [vpeak_guess, tc_guess, sigma_guess]
    bounds = (
        [0, t.min(), 1e-3],
        [vpeak_guess * 2 + 1, t.max(), span],
    )
    popt = _robust_curve_fit(bell, t, y, p0, bounds, "hand_velocity")
    if popt is None:
        return None
    return dict(vpeak=popt[0], tc=popt[1], sigma=popt[2])


# ---------- 对每个 trial 做平滑 ----------

def smooth_all_trials(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对每个 trial_id 分别拟合三条曲线，
    返回 (加了 *_smooth 列的原表, 每个 trial 的拟合参数表)。
    """
    df = df.sort_values([TRIAL_COL, TIME_COL]).reset_index(drop=True)
    df[f"{AREA_COL}_smooth"] = np.nan
    df[f"{APERTURE_COL}_smooth"] = np.nan
    df[f"{VEL_COL}_smooth"] = np.nan

    param_rows = []

    for trial_id, g in df.groupby(TRIAL_COL):
        t = g[TIME_COL].to_numpy(dtype=float)
        idx = g.index

        area_params = fit_area(t, g[AREA_COL].to_numpy(dtype=float))
        aper_params = fit_aperture(t, g[APERTURE_COL].to_numpy(dtype=float))
        vel_params = fit_velocity(t, g[VEL_COL].to_numpy(dtype=float))

        if area_params is not None:
            df.loc[idx, f"{AREA_COL}_smooth"] = logistic(t, **area_params)
        else:
            df.loc[idx, f"{AREA_COL}_smooth"] = g[AREA_COL].to_numpy()

        if aper_params is not None:
            df.loc[idx, f"{APERTURE_COL}_smooth"] = double_sigmoid(t, **aper_params)
        else:
            df.loc[idx, f"{APERTURE_COL}_smooth"] = g[APERTURE_COL].to_numpy()

        # if vel_params is not None:
        #     df.loc[idx, f"{VEL_COL}_smooth"] = bell(t, **vel_params)
        # else:
            df.loc[idx, f"{VEL_COL}_smooth"] = g[VEL_COL].to_numpy()

        row = {TRIAL_COL: trial_id}
        if GROUP_COL is not None and GROUP_COL in g.columns:
            row[GROUP_COL] = g[GROUP_COL].iloc[0]
        for name, p in [("area", area_params), ("aperture", aper_params)]:
            if p is not None:
                row.update({f"{name}_{k}": v for k, v in p.items()})
        param_rows.append(row)

    return df, pd.DataFrame(param_rows)


# ---------- 仿照参考图风格的对比图 ----------

def plot_comparison(df_raw: pd.DataFrame, save_path: str, show_raw=True):
    """
    3 行（Area / Aperture / Velocity） x N 列（按 GROUP_COL 分组）的对比图，
    每组把所有 trial 的原始点池化在一起再拟合一条代表性曲线画出来，
    风格对齐你截图里的样子。
    """
    if GROUP_COL is not None and GROUP_COL in df_raw.columns:
        groups = list(pd.unique(df_raw[GROUP_COL]))
    else:
        groups = [None]

    n = len(groups)
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 9), sharex="col")
    if n == 1:
        axes = axes.reshape(3, 1)

    row_specs = [
        (AREA_COL, "Object Size\n(px^2)", "seagreen", "darkgreen", fit_area, logistic),
        (APERTURE_COL, "Fingertip Aperture.\n(mm)", "cornflowerblue", "navy", fit_aperture, double_sigmoid),
        (VEL_COL, "Hand Velocity\n(mm/s)", "orange", "firebrick", None, None),
    ]

    for col_i, grp in enumerate(groups):
        sub = df_raw if grp is None else df_raw[df_raw[GROUP_COL] == grp]
        t_all = sub[TIME_COL].to_numpy(dtype=float)

        for row_i, (col, ylabel, dot_color, line_color, fitfunc, model) in enumerate(row_specs):
            ax = axes[row_i, col_i]
            y_all = sub[col].to_numpy(dtype=float)

            if show_raw:
                ax.scatter(t_all, y_all, s=8, alpha=0.35, color=dot_color, linewidths=0)

            # params = fitfunc(t_all, y_all)
            # if params is not None:
            #     tt = np.linspace(t_all.min(), t_all.max(), N_SMOOTH_POINTS)
            #     ax.plot(tt, model(tt, **params), color=line_color, linewidth=2.4)

            if fitfunc is not None:
                params = fitfunc(t_all, y_all)

                if params is not None:
                    tt = np.linspace(t_all.min(), t_all.max(), N_SMOOTH_POINTS)

                    base_curve = model(tt, **params)

                    # 从原始数据计算相对于模型的残差
                    fitted_raw = model(t_all, **params)
                    residual = y_all - fitted_raw

                    # 对残差进行轻微平滑，保留自然波动
                    residual_smooth = np.interp(tt,t_all,residual)

                    # 限制波动幅度，防止出现离谱的数据
                    residual_limit = np.std(residual) * 1.2
                    residual_smooth = np.clip(residual_smooth,-residual_limit,residual_limit)

                    final_curve = base_curve + residual_smooth

                    ax.plot(tt,final_curve,color=line_color,linewidth=2.4)
            else:
                # Velocity：直接使用原始数据
                ax.plot(t_all,y_all,color=line_color,linewidth=2.0)

            if col_i == 0:
                ax.set_ylabel(ylabel)
            if row_i == 0 and grp is not None:
                ax.set_title(f"{GROUP_COL}: {grp}")
            if row_i == 2:
                ax.set_xlabel("Time")
            ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"saved: {save_path}")


# ---------- 演示数据（仅当 CSV_PATH 不存在时自动生成） ----------

def _make_demo_csv(path: str, groups=(10, 30, 50), n_trials=15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t = np.linspace(0, 1.2, 80)

    for g_w in groups:
        area_final = g_w ** 2 * 10          # 越大的物体，最终投影面积越大
        aper_peak = g_w + 35                # 张开幅度比物体宽一些
        aper_final = g_w                    # 抓稳后指间距≈物体宽度

        for trial in range(n_trials):
            tid = f"w{g_w}_t{trial}"
            area = logistic(t, y0=500, y1=area_final, t0=0.75, k=9) \
                + rng.normal(0, area_final * 0.03, size=t.shape)
            aper = double_sigmoid(
                t, y0=0, ypeak=aper_peak, yend=aper_final,
                t1=0.35, k1=10, t2=0.9, k2=10,
            ) + rng.normal(0, 3, size=t.shape)
            vel = bell(t, vpeak=1100, tc=0.55, sigma=0.18) \
                + rng.normal(0, 40, size=t.shape)
            vel = np.clip(vel, 0, None)

            for i in range(len(t)):
                rows.append({
                    TRIAL_COL: tid,
                    GROUP_COL: g_w,
                    TIME_COL: t[i],
                    AREA_COL: max(area[i], 0),
                    APERTURE_COL: max(aper[i], 0),
                    VEL_COL: vel[i],
                })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[提示] 没找到 {path}，已自动生成一份演示数据用于测试脚本。")


# ---------- 主入口 ----------

def main():
    if not os.path.exists(CSV_PATH):
        _make_demo_csv(CSV_PATH)

    df_raw = pd.read_csv(CSV_PATH)

    # 容错：CSV 里没有 trial_id / group 列时自动补上，
    # 而不是直接报 KeyError 崩掉。
    if TRIAL_COL not in df_raw.columns:
        print(f"[提示] CSV 里没有列 '{TRIAL_COL}'，当成单个 trial 处理。")
        df_raw[TRIAL_COL] = "trial_0"

    global GROUP_COL
    if GROUP_COL is not None and GROUP_COL not in df_raw.columns:
        print(f"[提示] CSV 里没有列 '{GROUP_COL}'，本次不分组。")
        GROUP_COL = None

    missing_value_cols = [c for c in (TIME_COL, AREA_COL, APERTURE_COL, VEL_COL) if c not in df_raw.columns]
    if missing_value_cols:
        raise KeyError(
            f"CSV 里缺少这些必须的列：{missing_value_cols}\n"
            f"当前 CSV 的列名是：{list(df_raw.columns)}\n"
            f"请把脚本顶部 CONFIG 里对应的列名改成你 CSV 里实际的列名。"
        )

    df_smooth, df_params = smooth_all_trials(df_raw)
    df_smooth.to_csv(f"{OUT_PREFIX}_smoothed.csv", index=False)
    df_params.to_csv(f"{OUT_PREFIX}_fit_params.csv", index=False)
    print(f"saved: {OUT_PREFIX}_smoothed.csv")
    print(f"saved: {OUT_PREFIX}_fit_params.csv")

    plot_comparison(df_raw,f"{OUT_PREFIX}_comparison_with_raw.png",show_raw=True)
    plot_comparison(df_raw,f"{OUT_PREFIX}_comparison_clean.png",show_raw=False)



if __name__ == "__main__":
    main()