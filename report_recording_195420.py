from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,sosfilt_zi

s=Path('Data/BackupData/D/TRIAL_Size_3.5cm_20261014_195420')
o=s/'grasp_segments'
c=pd.read_csv(s/'camera.csv');im=pd.read_csv(s/'imu.csv')
r=pd.read_csv(o/'crop_ranges.csv');summary=pd.read_csv(o/'crop_summary.csv')
ordered=im.sort_values('t_imu',kind='stable').drop_duplicates('t_imu',keep='last')
ti=ordered.t_imu.to_numpy();dt=np.diff(ti)
g=np.flatnonzero(dt>.06)
pd.DataFrame({'t_before':ti[g],'t_after':ti[g+1],'gap_seconds':dt[g],'near_camera_frame':np.interp(ti[g],c.t_camera,c.frame)}).to_csv(o/'imu_gaps.csv',index=False)
checks=[]
for row in summary.itertuples():
 p=o/row.trial
 raw=pd.read_csv(p/'grasp_data.csv');sm=pd.read_csv(p/'grasp_data_smoothed.csv')
 imu=pd.read_csv(p/'imu.csv').sort_values('t_imu',kind='stable').drop_duplicates('t_imu',keep='last')
 t=imu.t_imu.to_numpy();tc=raw.t_camera.to_numpy()
 assert t[0]<=tc[0] and t[-1]>=tc[-1]
 assert np.isfinite(raw.select_dtypes('number')).all().all()
 assert np.isfinite(sm.select_dtypes('number')).all().all()
 grid=t[0]+np.arange(int(np.ceil((tc[-1]-t[0])*50))+1)/50
 assert grid[-1]<=t[-1]
 u=np.interp(grid,t,imu.forward_raw);sos=butter(2,3,fs=50,output='sos')
 v=sosfilt(sos,u,zi=sosfilt_zi(sos)*u[0])[0]
 error=float(np.max(np.abs(sm.forward_acceleration-np.interp(tc,grid,v))))
 assert error<2e-5
 for col in ['finger_width','object_area']:
  assert np.allclose(sm[col],raw[col].rolling(3,min_periods=1).median().ewm(span=5,adjust=False).mean())
 assert np.array_equal(raw.forward_raw,sm.forward_raw)
 assert int(row.onset-row.start)==25 and int(row.end-row.stable)==15
 checks.append(dict(trial=row.trial,filter_roundtrip_max_error=error,imu_coverage=True,finite_values=True,pre_frames=25,post_frames=15))
pd.DataFrame(checks).to_csv(o/'filter_verification.csv',index=False)
t=lambda k:datetime.fromtimestamp(c.t_camera.iloc[k],timezone(timedelta(hours=9))).isoformat()
report=f'''# 裁剪检查结果

输出 {len(summary)} 段：3.5 cm {sum(summary.size_cm==3.5)} 段，4 cm {sum(summary.size_cm==4)} 段，4.5 cm {sum(summary.size_cm==4.5)} 段。

尺寸依据用户更正的顺序 3.5 → 4 → 4.5 cm，与原始画面中的换物事件对应；画面不能独立标定绝对厘米数。
第一次换物：帧 3150（{t(3150)}）移走旧物，帧 3300（{t(3300)}）放入新物。
第二次换物：帧 11700（{t(11700)}）旧物已移走，帧 11850（{t(11850)}）放入新物。size_transitions.jpg 保存换物画面。
片段名按包含前置静止区的实际首帧 t_camera 转 UTC+9，截取到秒，无额外后缀。原录制目录名不作修改。

## 分段

沿用 crop_signed_recording.py 和 Smooth.py，检测用 5 帧居中中值 FW，峰显著度 60 px、间距 50 帧，取消固定 150 px 峰下限，避免漏掉后段较低的完整抓取峰。
峰前 65 帧 FW 第 10 百分位为基线，基线加局部峰幅 10% 为起点阈值；每段包含起点前 25 帧。
稳定判据为连续 16 帧 FW 极差 ≤20 px，FW 中位数处于局部峰幅的 15%–80%；OA 极差/中位数 ≤22%，相对动作前 OA 中位数增长 ≥1.35 倍。结尾为稳定起点后 15 帧。
相较原流程，延长稳定验证窗口并加入 OA 稳定约束；大物体闭合后的 FW 比例较大，故提高闭合比例上限。所有候选检查原始起点、峰值、稳定点及末帧画面。
{len(r)} 个候选中排除 {sum(r.status=='excluded')} 个：3 个手部调整，3 个稳定尾段已含松手返回，5 个存在 FW=0 跟踪无效值，5 个无法满足稳定窗口。详见 crop_ranges.csv。稳定时间不足不等于没有发生物理抓取。

## IMU 覆盖及丢包检查

原始 IMU {len(im)} 行，重复时间戳 {im.t_imu.duplicated().sum()} 行，倒序间隔 {(im.t_imu.diff()<0).sum()} 个，本地 imu_index 跳号 {(im.imu_index.diff().dropna()!=1).sum()} 个。
IMU 覆盖整个相机时间区间，排序去重后最大间隙 {dt.max():.6f} 秒，超过 60 ms 的间隙 {len(g)} 个，超过 250 ms 的间隙 {sum(dt>.25)} 个。
导出片段最大 IMU 间隙 {summary.imu_max_gap_seconds.max():.6f} 秒；最大相机帧间隔 {summary.camera_max_gap_seconds.max():.6f} 秒。每段均有前后真实 IMU 样本覆盖；imu_gaps.csv 列出全录制超过 60 ms 的间隙。
录制代码以主机接收时间加批内 i/50 产生时间戳，imu_index 是本地计数。因此时间抖动、重复、倒序不能直接换算硬件丢包率；本地行序号无缺失，但不能宣称硬件零丢包。
同步时稳定排序、重复时间保留最后一行。导出的 imu.csv 保留所选时间范围内的原始行及原始顺序，并保留滤波网格边界支持样本。

## 平滑和验证

FW/OA：Smooth.smooth_trial 的 3 帧中值 + EMA span=5。ax/ay/az 和 forward_raw：50 Hz 时间网格，二阶 3 Hz 因果 Butterworth，随后插值至相机帧。
forward_raw 保持同步后的有符号原值，forward_acceleration 为有符号滤波结果；mag 仅为原流程附带字段，不替代有符号加速度。未进行均值对齐、曲线平移或幅值调整。
全部 {len(summary)} 段已完整解码视频验证：视频帧数=原始 frame 文件数=camera.csv、grasp_data.csv、grasp_data_smoothed.csv 行数，frame/t_camera 逐行一致。
复制图片逐一 SHA256 相同；源 camera.csv、imu.csv、recorded_video.mp4 导出前后 SHA256 相同。没有生成 JSON。
全部片段重新从导出的 imu.csv 重建有符号滤波结果，最大 CSV 往返误差 {max(x['filter_roundtrip_max_error'] for x in checks):.9g} m/s²；FW/OA 平滑也已逐段重算核对，见 filter_verification.csv。
视频使用 30 fps，每个原始帧只写入一次；CSV 保存真实采样时间，视频时长与真实时间可能有少量差异。
combined_overlay.png 为合并叠加图，overlay_3.5cm.png、overlay_4cm.png、overlay_4.5cm.png 为分尺寸图；横轴仅归一化片段时间至 0–100%，纵轴保留原数值。
'''
(o/'processing_report.md').write_text(report,encoding='utf-8')
print('Verified',len(checks),'segments; max filter error',max(x['filter_roundtrip_max_error'] for x in checks))
