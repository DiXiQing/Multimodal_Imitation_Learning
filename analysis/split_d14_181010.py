from pathlib import Path
root=Path(__file__).resolve().parents[1]
template=(root/'analysis/split_150004_grasps.py').read_text(encoding='utf-8')
template=template.replace("'Data/BackupData/D/TRIAL_Size_3.5cm_20261013_150004_279664'", "'Data/临时数据/D14/TRIAL_Size_3.5cm_20261013_181010_655846'")
template=template.replace('ranges = [(235,305), (388,459), (569,635), (782,847), (1028,1093), (1205,1274)]',
                         "ranges = [(r['start'],r['end']) for r in json.loads((source/'candidate_ranges.json').read_text())]")
template=template.replace('import cv2','import cv2\nfrom scipy.signal import butter,sosfilt,sosfilt_zi')
template=template.replace("smooth['mag'] = np.linalg.norm(smooth[['ax','ay','az']],axis=1)", """smooth['mag'] = np.linalg.norm(smooth[['ax','ay','az']],axis=1)
    ema_comparison = smooth.copy()
    uniform = ti[0] + np.arange(int(np.floor((ti[-1]-ti[0])*50))+1)/50
    u = np.column_stack([np.interp(uniform,ti,axes[:,j]) for j in range(3)])
    sos = butter(2,3,fs=50,output='sos')
    filtered = np.column_stack([sosfilt(sos,u[:,j],zi=sosfilt_zi(sos)*u[0,j])[0] for j in range(3)])
    smooth[['ax','ay','az']] = np.column_stack([np.interp(tc,uniform,filtered[:,j]) for j in range(3)])
    smooth['mag'] = np.linalg.norm(smooth[['ax','ay','az']],axis=1)""")
template=template.replace('destination.mkdir()', """destination.mkdir()
    comparison = destination/'filter_comparison'
    comparison.mkdir()
    ema_comparison.to_csv(comparison/'grasp_data_smoothed_ema.csv',index=False,float_format='%.9g')
    (destination/'acceleration_processing.json').write_text(json.dumps(dict(
        method='2nd-order causal Butterworth on gravity-removed axes before camera sync',
        cutoff_hz=3,imu_uniform_rate_hz=50,post_sync_ema=False,
        initialization='steady state at first segment sample',
        velocity='existing integrated velocity, not recomputed from low-pass axes'),indent=2),encoding='utf-8')""")
template=template.replace('150004_segment','d14_181010_segment').replace('6 grasps / 3.5 cm','21 grasps / 3.5 cm / acceleration 3 Hz')
template=template.replace('six_grasps_smooth_overlay.png','21_grasps_smooth_overlay.png').replace('all 6 videos','all 21 videos')
template=template.replace('color=plt.cm.tab10(number-1)', 'color=plt.cm.tab20((number-1)%20),lw=1.1')
template=template.replace("frameon=False,title=", "frameon=False,ncol=3,title=")
exec(compile(template,str(root/'analysis/split_150004_grasps.py'),'exec'))
