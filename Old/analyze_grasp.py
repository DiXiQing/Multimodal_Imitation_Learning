"""
analyze_grasp_mp.py
用 MediaPipe 测指距 + 颜色测方块面积,分析抓取视频

MediaPipe负责手(拇指尖4、食指尖8),不受方块干扰
颜色检测负责红方块面积,并排除肤色区域避免把手当红色

依赖: pip install mediapipe opencv-python matplotlib

用法: 改 VIDEO_PATH,运行
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# ─────── 配置 ───────
VIDEO_PATH = r"C:\MineApp\Code\Multimodal_Imitation_Learning\VID_20260820_224047.mp4"

BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 255, 80])
MIN_CUBE_AREA = 3000
# ────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
K = np.ones((5,5), np.uint8)

THUMB_TIP = 4
INDEX_TIP = 8


def detect_cube(frame, hand_mask=None):
    """颜色检测红方块,返回(面积, 框)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    black = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)
    # 排除手的区域(手是亮的,一般不会被当黑色,但保险起见保留)
    if hand_mask is not None:
        black = cv2.bitwise_and(black, cv2.bitwise_not(hand_mask))
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, K)
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, K)
    cnts,_ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_CUBE_AREA:
        return 0, None
    return area, cv2.boundingRect(c)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] 打不开视频: {VIDEO_PATH}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = VIDEO_PATH.rsplit(".",1)[0] + "_mp.mp4"
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.3,
                           min_tracking_confidence=0.3)

    areas, dists, frames = [], [], []
    idx = 0
    hand_detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 方块
        area, box = detect_cube(frame)

        # 手(MediaPipe)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        dist = 0
        vis = frame.copy()

        if box is not None:
            x,y,bw,bh = box
            cv2.rectangle(vis, (x,y),(x+bw,y+bh),(0,0,255),2)
            cv2.putText(vis, f"area={int(area)}",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        if result.multi_hand_landmarks:
            hand_detected_count += 1
            lm = result.multi_hand_landmarks[0]
            thumb = lm.landmark[THUMB_TIP]
            index = lm.landmark[INDEX_TIP]
            tp = np.array([thumb.x*w, thumb.y*h])
            ip = np.array([index.x*w, index.y*h])
            dist = float(np.linalg.norm(tp-ip))

            mp_draw.draw_landmarks(vis, lm, mp_hands.HAND_CONNECTIONS)
            cv2.circle(vis, tuple(tp.astype(int)),10,(255,0,0),-1)
            cv2.circle(vis, tuple(ip.astype(int)),10,(0,0,255),-1)
            cv2.line(vis, tuple(tp.astype(int)), tuple(ip.astype(int)),(0,255,255),2)
            cv2.putText(vis, f"dist={int(dist)}",(10,70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)

        cv2.putText(vis, f"frame {idx}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        vw.write(vis)

        if area > 0 and dist > 0:
            areas.append(area); dists.append(dist); frames.append(idx)
        idx += 1

    cap.release()
    vw.release()
    hands.close()

    print(f"[Saved] 可视化视频: {out_video}")
    print(f"总帧数: {idx}")
    print(f"MediaPipe检测到手的帧: {hand_detected_count} ({hand_detected_count/idx*100:.0f}%)")
    print(f"两者都检测到的有效帧: {len(frames)}")

    if len(frames) < 2:
        print("\n[结论] MediaPipe在这个视频上检测率太低,不适用")
        print("       建议回到方向一(颜色法打补丁)")
        return

    areas = np.array(areas); dists = np.array(dists)

    fig, ax = plt.subplots(1,2, figsize=(14,5))
    ax[0].plot(frames, areas, 'r-o', ms=3)
    ax[0].set_xlabel('Frame'); ax[0].set_ylabel('Cube Area (px^2)', color='r')
    a0b = ax[0].twinx()
    a0b.plot(frames, dists, 'b-o', ms=3)
    a0b.set_ylabel('Finger Dist (px)', color='b')
    ax[0].set_title('Area & Finger Distance over Time')

    ax[1].scatter(areas, dists, c=frames, cmap='viridis', s=30)
    ax[1].set_xlabel('Object Area (px^2)')
    ax[1].set_ylabel('Finger Distance (px)')
    ax[1].set_title('Object Area vs Finger Distance')
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_plot = VIDEO_PATH.rsplit(".",1)[0] + "_mp_curve.png"
    plt.savefig(out_plot, dpi=150)
    print(f"[Saved] 曲线: {out_plot}")

    out_csv = VIDEO_PATH.rsplit(".",1)[0] + "_mp_data.csv"
    with open(out_csv,"w") as f:
        f.write("frame,cube_area,finger_dist\n")
        for fr,a,d in zip(frames,areas,dists):
            f.write(f"{fr},{a:.0f},{d:.1f}\n")
    print(f"[Saved] 数据: {out_csv}")
    plt.show()


if __name__ == "__main__":
    main()