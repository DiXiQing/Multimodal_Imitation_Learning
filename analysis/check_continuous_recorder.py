from pathlib import Path
import sys,tempfile,json
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import Record_mediapipe_continuous as m
from imu_processing import gravity_in_sensor
import struct

def packet(angles=(0,0,0),dynamic=(0,0,0)):
    raw=gravity_in_sensor(angles)+np.array(dynamic)
    vals=[*np.rint(raw/16*32768).astype(int),0,0,0,*np.rint(np.array(angles)/180*32768).astype(int)]
    return b'\x55\x61'+struct.pack('<9h',*vals)

with tempfile.TemporaryDirectory(prefix='continuous_test_',dir=root/'analysis') as tmp:
    m.OUTPUT_ROOT=Path(tmp)/'中文数据'
    r=m.Recorder();r.connected=True;r.calibrating=False;r.calibration_ready=True
    r.calibration={'calibration_quality':'Stable calibration samples'}
    raw=np.full((48,64,3),80,dtype=np.uint8);display=raw.copy();display[:10]=240
    r.start_trial('4.5',64,48);first=r.trial
    r.notification(None,packet((30,0,0)))
    with r.lock:r.trial.save_frame(raw,display,m.time.time(),123.,7000.)
    assert r.trial.imu_count==1
    r.start_trial('4.5',64,48);second=r.trial
    assert first.closed and second is not first and r.connected
    assert second.frame_count==second.imu_count==0
    np.testing.assert_array_equal(r.velocity,np.zeros(3))
    r.notification(None,packet((0,45,0),(.1,0,0)))
    with r.lock:r.trial.save_frame(raw,display,m.time.time(),140.,8000.)
    r.finish_trial()
    assert second.closed and r.trial is None and r.connected
    r.notification(None,packet())
    r.start_trial('3.5',64,48);third=r.trial
    r.request_calibration()
    assert third.closed and r.trial is None and r.connected and r.calibrating
    r.notification(None,packet())
    assert len(r.raw)==1
    for trial in [first,second]:
        c=pd.read_csv(trial.path/'camera.csv');i=pd.read_csv(trial.path/'imu.csv')
        assert len(c)==len(i)==1 and c.frame.iloc[0]==i.imu_index.iloc[0]==0
        assert list(i.columns)==m.IMU_COLUMNS
        np.testing.assert_allclose(i.mag,np.linalg.norm(i[['ax','ay','az']],axis=1))
        assert (trial.path/'recorded_video.mp4').stat().st_size>0
        saved=m.cv2.imdecode(np.fromfile(trial.frames/'frame_000000.jpg',dtype=np.uint8),m.cv2.IMREAD_COLOR)
        assert saved.mean()<100, 'Saved image must be the original, not the bright overlay'
print('PASS: separate trials/CSV/video, counters and integration reset, connection retained, pause and recalibration without sample mixing, original frame saving')
