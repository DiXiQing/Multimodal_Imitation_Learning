"""Continuous trial recording; Record_mediapipe.py is left unchanged.

Next trial: save/start without reconnecting. Recalibrate: save, calibrate,
then start a new trial. Finish trial: save and pause with preview still open.
"""
import asyncio
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time

import cv2
import numpy as np
from bleak import BleakClient, BleakScanner
from PIL import Image, ImageTk

import Record_mediapipe as config
from imu_processing import gravity_in_sensor, linear_acceleration, calibration_quality

# Defaults follow the current recorder; size can also be changed in the window.
OBJECT_SIZE_CM = config.TRIAL_NAME.split('Size_')[1].split('cm_')[0]
OUTPUT_ROOT = Path(__file__).resolve().parent / config.BACKUP_DIR
IMU_COLUMNS = ['imu_index','t_imu','ax','ay','az','mag','vx','vy','vz','velocity',
               'ax_raw_g','ay_raw_g','az_raw_g','gx_dps','gy_dps','gz_dps',
               'roll_deg','pitch_deg','yaw_deg','gravity_x_g','gravity_y_g','gravity_z_g']


class Trial:
    def __init__(self, base, size, width, height, calibration):
        self.name = f'TRIAL_Size_{float(size):g}cm_{datetime.now():%Y%m%d_%H%M%S_%f}'
        self.path = base / self.name
        self.frames = self.path / 'frames'
        self.frames.mkdir(parents=True, exist_ok=False)
        self.camera_file = (self.path/'camera.csv').open('w',newline='',encoding='utf-8')
        self.imu_file = (self.path/'imu.csv').open('w',newline='',encoding='utf-8')
        self.camera = csv.writer(self.camera_file);self.imu = csv.writer(self.imu_file)
        self.camera.writerow(['frame','t_camera','finger_width','object_area'])
        self.imu.writerow(IMU_COLUMNS)
        self.frame_count = self.imu_count = 0
        self.started = time.time()
        self.closed = False
        # OpenCV paths on Windows may not support Chinese filenames.
        fd = tempfile.NamedTemporaryFile(prefix='grasp_video_',suffix='.mp4',delete=False)
        self.video_temp = Path(fd.name);fd.close()
        self.video = cv2.VideoWriter(str(self.video_temp),cv2.VideoWriter_fourcc(*'mp4v'),config.TARGET_FPS,(width,height))
        if not self.video.isOpened():
            self.camera_file.close();self.imu_file.close()
            self.video_temp.unlink(missing_ok=True)
            raise RuntimeError('Cannot create trial video')
        (self.path/'imu_processing.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')

    def save_frame(self, raw, display, timestamp, aperture, area):
        if self.closed:raise RuntimeError('Trial already closed')
        ok,encoded=cv2.imencode('.jpg',raw)
        if not ok:raise RuntimeError('Cannot encode camera frame')
        encoded.tofile(self.frames/f'frame_{self.frame_count:06d}.jpg')
        self.camera.writerow([self.frame_count,f'{timestamp:.9f}',aperture,area])
        self.video.write(display)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.camera_file.flush();self.imu_file.flush()

    def save_imu(self, timestamp, net, packet, velocity):
        if self.closed or timestamp < self.started:return
        self.imu.writerow([self.imu_count,timestamp,*net,float(np.linalg.norm(net)),
                           *velocity,float(np.linalg.norm(velocity)),*packet['acc'],
                           *packet['gyro'],*packet['angles'],*gravity_in_sensor(packet['angles'])])
        self.imu_count += 1

    def close(self):
        if self.closed:return
        self.closed=True
        self.video.release();self.camera_file.close();self.imu_file.close()
        shutil.move(str(self.video_temp),str(self.path/'recorded_video.mp4'))
        (self.path/'trial_summary.json').write_text(json.dumps(dict(
            camera_frames=self.frame_count,imu_samples=self.imu_count,
            started=self.started,ended=time.time(),object_color=config.OBJECT_COLOR),indent=2),encoding='utf-8')
        print(f'[Saved] {self.path} | frames={self.frame_count}, IMU={self.imu_count}',flush=True)


class Recorder:
    def __init__(self):
        self.lock=threading.RLock()
        self.stop=threading.Event();self.recalibrate=threading.Event()
        self.connected=False;self.calibrating=True;self.calibration_ready=False
        self.status='Connecting to IMU...';self.trial=None
        self.bias=np.zeros(3);self.latest=np.zeros(3)
        self.raw=[];self.residual=[];self.gyro=[];self.times=[]
        self.calibration={};self.auto_start=False
        self.reset_integration()

    def reset_integration(self):
        self.filtered=np.zeros(3);self.previous=None;self.previous_time=None
        self.velocity=np.zeros(3)

    def begin_calibration(self):
        with self.lock:
            self.calibrating=True;self.calibration_ready=False
            self.raw=[];self.residual=[];self.gyro=[];self.times=[]
            self.calibration_started=time.monotonic()
            self.status='Keep IMU still: checking calibration (at least 2 s)'

    def finish_trial(self):
        with self.lock:
            if self.trial is not None:
                trial=self.trial;self.trial=None;trial.close()
            self.reset_integration()

    def start_trial(self, size, width, height):
        size=float(size)
        if not np.isfinite(size) or size<=0:raise ValueError('Object size must be positive')
        with self.lock:
            if not self.connected or self.calibrating or not self.calibration_ready:
                raise RuntimeError('Wait for IMU calibration')
            self.finish_trial()
            self.reset_integration()
            self.trial=Trial(OUTPUT_ROOT,size,width,height,self.calibration)
            self.status='Recording '+self.trial.name

    def request_calibration(self):
        with self.lock:
            if not self.connected:raise RuntimeError('IMU not connected')
            self.finish_trial()
            self.begin_calibration()
            self.recalibrate.set()

    def notification(self,sender,data):
        if self.stop.is_set():return
        received=time.time()
        for i,packet in enumerate(config.extract_imu_packets(bytes(data))):
            timestamp=received+i/50.
            with self.lock:
                if self.calibrating:
                    self.raw.append(packet['acc']);self.residual.append(packet['acc']-gravity_in_sensor(packet['angles']))
                    self.gyro.append(packet['gyro']);self.times.append(timestamp)
                    while len(self.times)>1 and timestamp-self.times[0]>2.1:
                        for buf in [self.raw,self.residual,self.gyro,self.times]:buf.pop(0)
                    continue
                net=linear_acceleration(packet,self.bias,config.GRAVITY)
                self.latest=net
                if self.trial is None:continue
                filtered=config.REALTIME_LPF_ALPHA*net+(1-config.REALTIME_LPF_ALPHA)*self.filtered
                if self.previous_time is not None:
                    dt=timestamp-self.previous_time
                    if 0<dt<=.2:self.velocity+=.5*(self.previous+filtered)*dt
                self.filtered=filtered;self.previous=filtered.copy();self.previous_time=timestamp
                self.trial.save_imu(timestamp,net,packet,self.velocity)

    async def measure(self):
        for attempt in range(1,config.IMU_MAX_CONNECT_ATTEMPTS+1):
            if self.stop.is_set():return
            try:
                with self.lock:self.status=f'Connecting to IMU ({attempt}/3)...'
                device=await BleakScanner.find_device_by_address(config.IMU_ADDRESS,timeout=config.IMU_SCAN_TIMEOUT_SECONDS)
                if device is None:raise ConnectionError('IMU not found; close other IMU apps')
                async with BleakClient(device,timeout=config.IMU_CONNECT_TIMEOUT_SECONDS,
                        winrt={'use_cached_services':False} if config.os.name=='nt' else {}) as client:
                    char=client.services.get_characteristic(config.IMU_NOTIFY_UUID)
                    if char is None or not {'notify','indicate'}.intersection(char.properties):
                        raise ConnectionError('IMU notification service unavailable')
                    with self.lock:self.connected=True
                    self.begin_calibration()
                    await client.start_notify(char,self.notification)
                    try:
                        while not self.stop.is_set():
                            if not client.is_connected:raise ConnectionError('IMU disconnected: recording stopped')
                            if self.recalibrate.is_set():
                                self.recalibrate.clear()
                                self.begin_calibration()
                            with self.lock:
                                waiting=self.calibrating
                                if waiting:
                                    elapsed=time.monotonic()-self.calibration_started
                                    good,reason=calibration_quality(self.raw,self.residual,self.gyro,self.times)
                                    if elapsed>=config.CALIB_SECONDS and good:
                                        self.bias=np.mean(self.residual,axis=0)
                                        self.calibration=dict(acceleration_method='attitude_gravity_removal_then_static_sensor_bias',
                                            calibration_seconds=config.CALIB_SECONDS,calibration_wait_seconds=elapsed,
                                            calibration_samples=len(self.raw),calibration_quality=reason,sensor_bias_g=self.bias.tolist(),
                                            gravity_m_s2=config.GRAVITY,attitude_convention='ENU Z-Y-X',ax_ay_az_units='m/s^2',
                                            ax_ay_az_frame='sensor/body',raw_acceleration_units='g')
                                        self.reset_integration();self.calibrating=False;self.calibration_ready=True
                                        self.status='Calibration complete';self.auto_start=True
                                        print('[IMU] Calibration complete; bias (g):',self.bias,flush=True)
                                    elif elapsed>config.CALIB_MAX_WAIT_SECONDS:
                                        raise RuntimeError('Calibration rejected: '+reason)
                                    elif elapsed>=config.CALIB_SECONDS:self.status=reason
                            await asyncio.sleep(.05)
                    finally:
                        if client.is_connected:await client.stop_notify(char)
                return
            except Exception as exc:
                with self.lock:
                    had_trial=self.trial is not None
                    self.connected=False;self.calibration_ready=False;self.calibrating=True
                    self.finish_trial();self.status=f'IMU error: {exc}'
                print('[IMU]',type(exc).__name__,str(exc),flush=True)
                if had_trial or attempt==config.IMU_MAX_CONNECT_ATTEMPTS:
                    self.stop.set();return
                for _ in range(int(config.IMU_RETRY_DELAY_SECONDS/.05)):
                    if self.stop.is_set():return
                    await asyncio.sleep(.05)

    def worker(self):
        try:asyncio.run(self.measure())
        finally:
            with self.lock:self.connected=False;self.finish_trial()


def main():
    import tkinter as tk
    from tkinter import messagebox
    window=tk.Tk();window.title('连续抓取录制 — 保持 IMU 连接')
    recorder=Recorder()
    cap=cv2.VideoCapture(config.CAMERA_INDEX,config.CAP_BACKEND)
    if not cap.isOpened():window.destroy();raise RuntimeError('Cannot open camera')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,config.FRAME_W);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,config.FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,config.TARGET_FPS)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hands=config.mp_hands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,
                               min_detection_confidence=.6,min_tracking_confidence=.6)
    status=tk.StringVar(value=recorder.status);info=tk.StringVar(value='Waiting for IMU')
    tk.Label(window,textvariable=status,font=('Microsoft YaHei',11)).pack(padx=10,pady=6)
    tk.Label(window,textvariable=info).pack()
    preview=tk.Label(window);preview.pack()
    controls=tk.Frame(window);controls.pack(padx=10,pady=10)
    tk.Label(controls,text='物体尺寸(cm)：').grid(row=0,column=0)
    size=tk.StringVar(value=OBJECT_SIZE_CM);tk.Entry(controls,textvariable=size,width=6).grid(row=0,column=1,padx=4)
    closing=False
    thread=threading.Thread(target=recorder.worker,daemon=True)

    def start(event=None):
        try:recorder.start_trial(size.get(),width,height)
        except Exception as exc:messagebox.showerror('无法开始录制',str(exc))

    def pause(event=None):
        recorder.finish_trial()
        with recorder.lock:recorder.status='Paused: preview and IMU connection remain active'

    def recalibrate(event=None):
        try:recorder.request_calibration()
        except Exception as exc:messagebox.showerror('无法校准',str(exc))

    def await_shutdown():
        if thread.is_alive():window.after(50,await_shutdown)
        else:window.destroy()

    def close(event=None):
        nonlocal closing
        if closing:return
        closing=True;recorder.stop.set();recorder.finish_trial()
        status.set('Saving trial and disconnecting IMU...')
        cap.release();hands.close();window.after(50,await_shutdown)

    start_button=tk.Button(controls,text='下一次实验 / 开始 (N)',command=start)
    start_button.grid(row=0,column=2,padx=5)
    finish_button=tk.Button(controls,text='结束本次 (S)',command=pause)
    finish_button.grid(row=0,column=3,padx=5)
    calib_button=tk.Button(controls,text='重新校准 (R)',command=recalibrate)
    calib_button.grid(row=0,column=4,padx=5)
    tk.Button(controls,text='保存并退出 (Q)',command=close).grid(row=0,column=5,padx=5)
    for key,fn in [('n',start),('s',pause),('r',recalibrate),('q',close)]:
        window.bind('<'+key+'>',fn);window.bind('<'+key.upper()+'>',fn)
    window.protocol('WM_DELETE_WINDOW',close)

    def update():
        if closing:return
        began=time.perf_counter()
        if recorder.stop.is_set():close();return
        try:
            with recorder.lock:
                if recorder.auto_start:
                    recorder.auto_start=False
                    recorder.start_trial(size.get(),width,height)
            ret,raw=cap.read()
            if not ret:raise RuntimeError('Camera read failed')
            timestamp=time.time();display=raw.copy()
            fw,thumb,index,landmarks=config.detect_finger_width(raw,hands)
            area,box=config.detect_cube(raw)
            if landmarks:config.mp_draw.draw_landmarks(display,landmarks,config.mp_hands.HAND_CONNECTIONS)
            if box:
                x,y,w,h=box;cv2.rectangle(display,(x,y),(x+w,y+h),config.OBJECT_BOX_COLORS[config.OBJECT_COLOR],2)
            with recorder.lock:
                trial=recorder.trial
                if trial:
                    cv2.putText(display,'REC',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
                    trial.save_frame(raw,display,timestamp,fw,area)
                status.set(recorder.status)
                info.set(f'FW={fw:.1f}px | OA={area:.0f}px² | Acc={np.linalg.norm(recorder.latest):.3f}m/s² | Frames={trial.frame_count if trial else 0}')
                ready=recorder.connected and recorder.calibration_ready and not recorder.calibrating
                start_button.config(state='normal' if ready else 'disabled')
                finish_button.config(state='normal' if trial else 'disabled')
                calib_button.config(state='normal' if ready else 'disabled')
            photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display,cv2.COLOR_BGR2RGB)))
            preview.config(image=photo);preview.image=photo
        except Exception as exc:
            print('[Recorder]',exc,flush=True);messagebox.showerror('录制错误',str(exc));close();return
        delay=max(1,int(1000*(1/config.TARGET_FPS-(time.perf_counter()-began))))
        window.after(delay,update)

    thread.start();update()
    try:window.mainloop()
    finally:
        recorder.stop.set();recorder.finish_trial()
        if not closing:cap.release();hands.close()
        thread.join()


if __name__=='__main__':main()
