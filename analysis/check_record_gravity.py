"""Exercise the real recording callback without opening camera/BLE hardware."""
import ast
import struct
import threading
import time
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from imu_processing import parse_imu_packet,gravity_in_sensor,linear_acceleration
import IMUTest
tree=ast.parse((root/'Record_mediapipe.py').read_text(encoding='utf-8-sig'))
names={'SharedState','parse_witmotion_packet','extract_imu_packets'}
body=[n for n in tree.body if getattr(n,'name',None) in names]
ns=dict(np=np,struct=struct,threading=threading,time=time,GRAVITY=9.81,
        REALTIME_LPF_ALPHA=.15,parse_imu_packet=parse_imu_packet,
        gravity_in_sensor=gravity_in_sensor,linear_acceleration=linear_acceleration)
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'record-core','exec'),ns)
fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='recording_imu_handler')
wrapper=ast.parse('def build():\n imu_counter=0\n').body[0]
wrapper.body += [fn,ast.parse('return recording_imu_handler').body[0]]
ns.update(state=ns['SharedState'](),imu_records=[],imu_records_lock=threading.Lock())
exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),'record-callback','exec'),ns)
callback=ns['build']()
def packet(angles, dynamic=(0,0,0)):
    acc=gravity_in_sensor(angles)+np.array(dynamic)+np.array([.01,-.02,.015])
    values=[*np.rint(acc/16*32768).astype(int),0,0,0,*np.rint(np.array(angles)/180*32768).astype(int)]
    return b'\x55\x61'+struct.pack('<9h',*values)
initial=packet((0,0,0))
callback(None,initial*3)
state=ns['state'];state.acc_bias=np.mean(state.calib_linear_buffer,axis=0).tolist();state.calibrating=False
test=IMUTest.State();test.begin();test.process(parse_imu_packet(initial),0);test.finish()
for angle,dynamic in [((30,0,0),(0,0,0)),((0,45,0),(0,0,0)),((40,-35,120),(.12,-.07,.03))]:
    encoded=packet(angle,dynamic)
    callback(None,encoded)
    test.process(parse_imu_packet(encoded),time.time())
    np.testing.assert_allclose(state.latest_acc,test.linear,atol=1e-12)
    np.testing.assert_allclose(state.latest_mag,test.linear_mag,atol=1e-12)
    if dynamic==(0,0,0):assert state.latest_mag<.02
    row=ns['imu_records'][-1]
    assert len(row)==22
    np.testing.assert_allclose(row[5],np.linalg.norm(row[2:5]))
header=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id=='imu_writer' and n.func.attr=='writerow')
assert len(ast.literal_eval(header.args[0]))==22
assert len(ns['imu_records'])==3
print('PASS: recorder callback equals IMUTest, tilt compensation, dynamic preservation and CSV layout')
