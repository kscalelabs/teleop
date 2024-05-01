import numpy as np
import time
import cv2
from src.teleop import Teleop

np.set_printoptions(precision=2, suppress=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
if not cap.isOpened():
    print("Error: Unable to open camera")
    exit()

tv = Teleop((480, 640), stereo=True)

while True:
    start = time.time()
    ret, frame = cap.read()
    if ret:
        width = frame.shape[1]
        left_frame = frame[:, :width//2]
        right_frame = frame[:, width//2:]
        frame = np.vstack((left_frame, right_frame))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tv.modify_shared_image(rgb_frame)
    end = time.time()
    
cap.release()