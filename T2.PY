import cv2
import numpy as np
import mediapipe as mp

# ─────── 配置 ───────
IMAGE_PATH = r"d:\Code\D\Poster2.jpg"  # ⚠️ 请替换

# 黑色方块的HSV范围 (如果方块反光有点发灰，可以把最后的80改成120)
BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 255, 80])

MIN_CUBE_AREA = 3000   # 最小面积
AR_MIN, AR_MAX = 0.6, 1.6  # 长宽比范围：接近1才像正方形/方块
MIN_SOLIDITY = 0.85        # 实心度：方块是实心的，绑带和手套通常有很多缝隙

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
THUMB_TIP = 4
INDEX_TIP = 8
# ────────────────────

def detect_cube(image):
    """找像“方块”的黑色轮廓，排除长条状的设备"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, None

    best_area = 0
    best_box = None

    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_CUBE_AREA:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        
        # 1. 过滤长条状物体（设备绑带、手环通常是长条的）
        if aspect_ratio < AR_MIN or aspect_ratio > AR_MAX:
            continue
            
        # 2. 过滤实心度低的物体（带缝隙的绑带）
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:
            continue

        # 选最大的那个符合条件的方块
        if area > best_area:
            best_area = area
            best_box = (x, y, w, h)

    return best_area, best_box


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[Error] 无法读取图片: {IMAGE_PATH}")
        return

    h, w, _ = img.shape

    # ─────── 1. 找方块并画红框 ───────
    area, box = detect_cube(img)
    if box is not None:
        x, y, bw, bh = box
        cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 0, 255), 3)
        cv2.putText(img, f"Area: {int(area)} px^2", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(f"检测到方块，面积为: {int(area)} 像素")
    else:
        print("未检测到方块，请调整光照或MIN_CUBE_AREA参数")

    # ─────── 2. 测手部距离 ───────
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            thumb = lm.landmark[THUMB_TIP]
            index = lm.landmark[INDEX_TIP]
            
            tp = (int(thumb.x * w), int(thumb.y * h))
            ip = (int(index.x * w), int(index.y * h))

            dist = float(np.linalg.norm(np.array(tp) - np.array(ip)))

            cv2.circle(img, tp, 10, (255, 0, 0), -1)
            cv2.circle(img, ip, 10, (0, 0, 255), -1)
            cv2.line(img, tp, ip, (0, 255, 255), 2)
            
            cv2.putText(img, f"Finger Dist: {int(dist)} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)
        else:
            print("未检测到手部")

    # ─────── 3. 显示保存 ───────
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = IMAGE_PATH.rsplit(".", 1)[0] + "_fixed.png"
    cv2.imwrite(output_path, img)
    print(f"[Saved] 结果已保存到: {output_path}")

if __name__ == "__main__":
    main()