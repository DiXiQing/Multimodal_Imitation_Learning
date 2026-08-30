"""
finger_gap_simple.py — 只做一件事: 找两个红点, 算距离

用法: python finger_gap_simple.py
按 'q' 退出
"""

import cv2
import numpy as np

CAMERA_INDEX = 0
CAP_BACKEND = cv2.CAP_DSHOW

# 蓝色固定阈值,蓝色不像红色那样跨0/180,一段就够
LOWER_BLUE = np.array([100, 100, 60])
UPPER_BLUE = np.array([130, 255, 255])

MIN_AREA = 20


def find_two_red_dots(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [(stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, num)
             if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]
    blobs.sort(key=lambda b: b[0], reverse=True)

    if len(blobs) < 2:
        return None, mask

    p1 = tuple(map(int, blobs[0][1]))
    p2 = tuple(map(int, blobs[1][1]))
    dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    return (dist, p1, p2), mask


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, CAP_BACKEND)
    if not cap.isOpened():
        print("[Error] 打不开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, mask = find_two_red_dots(frame)
        vis = frame.copy()

        if result is None:
            cv2.putText(vis, "BLUE DOTS NOT FOUND", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            dist, p1, p2 = result
            cv2.circle(vis, p1, 8, (255, 0, 0), -1)
            cv2.circle(vis, p2, 8, (0, 0, 255), -1)
            cv2.line(vis, p1, p2, (0, 255, 255), 2)
            cv2.putText(vis, f"Gap: {dist:.1f} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

        cv2.imshow("Finger Gap Simple", vis)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()