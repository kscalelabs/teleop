"""Utility classes and functions for data collection."""

import signal
import sys
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from data_collection.constants import CAM_HEIGHT, CAM_WIDTH, DT


class ImageRecorder:
    def signal_handler(self, sig: Any, frame: Any) -> None:
        print("You pressed Ctrl+C!")
        self.close_cameras()
        cv2.destroyAllWindows()
        sys.exit(0)

    def close_cameras(self) -> None:
        for cap in self.caps:
            try:
                if cap.isOpened():
                    cap.release()
            except AttributeError as e:
                print(e)
        cv2.destroyAllWindows()

    def __init__(self, camera_ids: list, pseudonyms: list, save_mp4: bool = False, save_path: str = "") -> None:
        self.is_debug = False
        self.camera_ids = camera_ids
        self.camera_names = pseudonyms
        self.caps = {}
        self.threads: dict = {}
        self.running = True
        self.locks = {}
        self.latest_frames: dict = {}
        self.out: dict = {}
        self.save_mp4 = save_mp4
        self.save_path = save_path

        for camera_id, camera_name in zip(self.camera_ids, self.camera_names):
            self.latest_frames[camera_id] = None
            self.locks[camera_id] = threading.Lock()

            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise IOError(f"Cannot open camera {camera_id}")
            self.caps[camera_id] = cap
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            if self.save_mp4:
                self.make_writer(camera_id, camera_name)
            time.sleep(2)
            self.update()
            if self.is_debug:
                setattr(self, f"timestamps_{camera_id}", deque(maxlen=50))

        signal.signal(signal.SIGINT, self.signal_handler)

    def update_camera(self, camera_id: str) -> None:
        cap = self.caps[camera_id]
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.locks[camera_id]:
                    self.latest_frames[camera_id] = frame
                    if self.is_debug:
                        getattr(self, f"timestamps_{camera_id}").append(time.time())

    def update(self) -> None:
        for camera_id in self.camera_ids:
            ret, frame = self.caps[camera_id].read()
            if ret:
                post = np.array(cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))).astype(np.uint8)
                if self.save_mp4:
                    self.out[camera_id].write(post)
                self.latest_frames[camera_id] = post
                if self.is_debug:
                    getattr(self, f"timestamps_{camera_id}").append(time.time())

    def set_save_path(self, save_path: str) -> None:
        self.save_path = save_path

    def make_writer(self, camera_id: str, camera_name: str) -> None:
        if self.save_mp4:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out[camera_id] = cv2.VideoWriter(
                f"{self.save_path}_{camera_name}.mp4", fourcc, int(1 / DT), (CAM_WIDTH, CAM_HEIGHT)
            )

    def close(self) -> None:
        if self.save_mp4:
            for vid in self.out.values():
                print(f"Releasing {vid}")
                vid.release()

    def get_images(self) -> dict:
        image_dict = dict()
        for camera_id, camera_name in zip(self.camera_ids, self.camera_names):
            with self.locks[camera_id]:
                image_dict[camera_name] = self.latest_frames[camera_id]
        return image_dict

    def print_diagnostics(self) -> None:
        def dt_helper(l: list) -> float:
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

    def __del__(self) -> None:
        self.running = False
        for thread in self.threads.values():
            thread.join(timeout=1.0)  # Wait for threads to finish, but with a timeout
        for cap in self.caps.values():
            if cap.isOpened():
                cap.release()


# Example usage:
if __name__ == "__main__":
    recorder = ImageRecorder([0, 1, 2, 3], ["cam1", "cam2", "cam3", "cam4"])  # Use cameras 0, 1, 2, and 3

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
