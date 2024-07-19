# flake8: noqa
# mypy: ignore-errors
import signal
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from data_collection.constants import CAM_HEIGHT, CAM_WIDTH


class ImageRecorder:
    def signal_handler(self, sig, frame):
        print("You pressed Ctrl+C!")
        for cap in self.caps:
            if cap.isOpened():
                cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    def __init__(self, camera_ids, pseudonyms, is_debug=False):
        self.is_debug = is_debug
        self.camera_ids = camera_ids
        self.camera_names = pseudonyms
        self.caps = {}
        self.threads = {}
        self.running = True
        self.locks = {}
        self.latest_frames = {}

        for camera_id in self.camera_ids:
            self.latest_frames[camera_id] = None
            self.locks[camera_id] = threading.Lock()

            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise IOError(f"Cannot open camera {camera_id}")
            self.caps[camera_id] = cap
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            time.sleep(2)
            print("Resize")

            # Set camera properties for maximum speed
            # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # cap.set(cv2.CAP_PROP_FPS, 120)  # Adjust based on your camera's capabilities
            self.update()
            if self.is_debug:
                setattr(self, f"timestamps_{camera_id}", deque(maxlen=50))

        signal.signal(signal.SIGINT, self.signal_handler)

    def update_camera(self, camera_id):
        cap = self.caps[camera_id]
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.locks[camera_id]:
                    self.latest_frames[camera_id] = frame
                    if self.is_debug:
                        getattr(self, f"timestamps_{camera_id}").append(time.time())

    def update(self):
        for camera_id in self.camera_ids:
            ret, frame = self.caps[camera_id].read()
            if ret:
                self.latest_frames[camera_id] = np.array(cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))).astype(np.uint8)
                if self.is_debug:
                    getattr(self, f"timestamps_{camera_id}").append(time.time())

    def get_images(self):
        image_dict = dict()
        for camera_id, camera_name in zip(self.camera_ids, self.camera_names):
            with self.locks[camera_id]:
                image_dict[camera_name] = self.latest_frames[camera_id]
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for camera_id in self.camera_ids:
            if self.is_debug:
                timestamps = getattr(self, f"timestamps_{camera_id}")
                if timestamps:
                    image_freq = 1 / dt_helper(timestamps)
                    print(f"Camera {camera_id} image_freq={image_freq:.2f} Hz")
                else:
                    print(f"Camera {camera_id} No timestamps available")
        print()

    def __del__(self):
        self.running = False
        for thread in self.threads.values():
            thread.join(timeout=1.0)  # Wait for threads to finish, but with a timeout
        for cap in self.caps.values():
            if cap.isOpened():
                cap.release()


# Example usage:
if __name__ == "__main__":
    recorder = ImageRecorder([0, 1, 2, 3], is_debug=True)  # Use cameras 0, 1, 2, and 3

    try:
        while True:
            frames = recorder.get_images()

            for camera_id, frame in frames.items():
                if frame is not None:
                    cv2.imshow(f"Camera {camera_id}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        recorder.print_diagnostics()
        cv2.destroyAllWindows()
