from pathlib import Path
import sys
import csv
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import Record_mediapipe as config


def main():
    trial = PROJECT_ROOT / 'Data/Exprement Data/D13/2/TRIAL_Size_2cm_20261014_165842'
    output = PROJECT_ROOT / 'Report/vision_example'
    output.mkdir(parents=True, exist_ok=True)
    with config.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
            model_complexity=1, min_detection_confidence=.6,
            min_tracking_confidence=.6) as hands:
        for source in sorted((trial / 'frames').glob('frame_*.jpg')):
            number = int(source.stem.split('_')[-1])
            if number > 533:
                break
            frame = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f'Cannot read {source}')
            distance, thumb, index, landmarks = config.detect_finger_width(frame, hands)
    if landmarks is None:
        raise RuntimeError('No hand detected in target frame')
    display = frame.copy()
    config.mp_draw.draw_landmarks(display, landmarks, config.mp_hands.HAND_CONNECTIONS)
    cv2.circle(display, thumb, 8, (255, 0, 0), -1)
    cv2.circle(display, index, 8, (0, 0, 255), -1)
    cv2.line(display, thumb, index, (0, 255, 255), 2)
    label = f'Fingertip aperture : {distance:.2f}px'
    cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                .6, (0, 255, 255), 2)
    area, box = config.detect_cube(frame)
    if box is None:
        raise RuntimeError('No object detected in target frame')
    x, y, width, height = box
    object_color = config.OBJECT_BOX_COLORS[config.OBJECT_COLOR]
    cv2.rectangle(display, (x, y), (x + width, y + height), object_color, 2)
    cv2.putText(display, f'Object area : {area:.2f}px^2', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, .6, object_color, 2)
    target = output / 'frame_000533_object_annotated.png'
    cv2.imencode('.png', display)[1].tofile(target)
    with (trial / 'camera.csv').open(newline='', encoding='utf-8-sig') as handle:
        recorded = next((row for row in csv.DictReader(handle) if int(row['frame']) == 533), None)
    print({'output': str(target), 'thumb_tip_px': thumb, 'index_tip_px': index,
           'distance_px': distance, 'object_area_px2': area, 'object_box': box,
           'recorded_csv': recorded})


if __name__ == '__main__':
    main()
