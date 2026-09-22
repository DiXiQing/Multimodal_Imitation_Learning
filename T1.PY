import cv2
import numpy as np

CAMERA_INDEX = 1
CAP_BACKEND = cv2.CAP_DSHOW

# 蓝色 HSV 范围，之后可以根据实际摄像头调整
BLUE_LOWER = np.array([90, 80, 50])
BLUE_UPPER = np.array([130, 255, 255])

MIN_AREA = 100


def detect_blue_points(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

    # 只取最大的两个蓝色区域
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    points = []

    for contour in contours:
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))

    return points, mask


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, CAP_BACKEND)

    if not cap.isOpened():
        print("[Error] 打不开摄像头")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        points, mask = detect_blue_points(frame)

        # 找到两个蓝色标记
        if len(points) == 2:
            p1, p2 = points

            # 两点欧氏距离
            finger_width = np.linalg.norm(np.array(p1) - np.array(p2))

            cv2.circle(frame, p1, 10, (0, 255, 0), -1)
            cv2.circle(frame, p2, 10, (0, 255, 0), -1)
            cv2.line(frame, p1, p2, (0, 255, 255), 3)

            cv2.putText(frame, f"Finger Width: {finger_width:.1f} px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        else:
            cv2.putText(frame, f"Blue markers: {len(points)}/2", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Finger Width", frame)
        cv2.imshow("Blue Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()