import cv2

VIDEO_PATH = r"C:\MineApp\Code\Multimodal_Imitation_Learning\VID_20260818_234613.mp4"

cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FPS)

print(f"FPS: {cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FPS)}")