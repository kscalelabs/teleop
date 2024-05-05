import asyncio
from multiprocessing import Process, Array, Value, shared_memory
from typing import Tuple

import numpy as np
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands

class Teleop:
    def __init__(self,
        img_shape: Tuple[int, int] = (480, 640),
        fps: int = 60,
        stereo: bool = True,
    ):
        self.img_shape = img_shape
        self.fps = fps
        self.stereo = stereo

        self.img_shape = (2*img_shape[0], img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.shm_name = self.shm.name
        self.shared_image = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.shm.buf)
        self.shared_image[:] = np.zeros(self.img_shape, dtype=np.uint8)

        self.app = Vuer(host='0.0.0.0', cert="", key="", queries=dict(grid=False))
        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        self.app.spawn(start=False)(self.main)

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        self.process = Process(target=self.run)
        self.process.start()

    def run(self):
        self.app.run()

    async def main(self, session):
        session.upsert @ Hands(fps=self.fps, stream=True, key="hands")
        while True:
            display_image = self.shared_image

            if not self.stereo:
                session.upsert(
                ImageBackground(
                    display_image[:self.img_height],
                    format="jpeg",
                    quality=80,
                    key="left-image",
                    interpolate=True,
                    aspect=1.778,
                    distanceToCamera=2,
                    position=[0, -0.5, -2],
                    rotation=[0, 0, 0],
                ),
                to="bgChildren",
                )
            else:
                session.upsert(
                [ImageBackground(
                    display_image[:self.img_height],
                    format="jpeg",
                    quality=40,
                    key="left-image",
                    interpolate=True,
                    aspect=1.778,
                    distanceToCamera=2,
                    layers=1
                ),
                ImageBackground(
                    display_image[self.img_height:],
                    format="jpeg",
                    quality=40,
                    key="right-image",
                    interpolate=True,
                    aspect=1.778,
                    distanceToCamera=2,
                    layers=2
                )],
                to="bgChildren",
                )
            await asyncio.sleep(1/self.fps)

    def modify_shared_image(self, img, random=False):
        assert img.shape == self.img_shape, f"Image shape must be {self.img_shape}, got {img.shape}"
        existing_shm = shared_memory.SharedMemory(name=self.shm_name)
        shared_image = np.ndarray(self.img_shape, dtype=np.uint8, buffer=existing_shm.buf)
        shared_image[:] = img[:] if not random else np.random.randint(0, 256, self.img_shape, dtype=np.uint8)
        existing_shm.close()

    async def on_cam_move(self, event, session):
        try:
            with self.head_matrix_shared.get_lock():  # Use the lock to ensure thread-safe updates
                self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            with self.aspect_shared.get_lock():
                self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass

    async def on_hand_move(self, event, session):
        try:
            with self.left_hand_shared.get_lock():  # Use the lock to ensure thread-safe updates
                self.left_hand_shared[:] = event.value["leftHand"]
            with self.right_hand_shared.get_lock():
                self.right_hand_shared[:] = event.value["rightHand"]
            with self.left_landmarks_shared.get_lock():
                self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            with self.right_landmarks_shared.get_lock():
                self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        except: 
            pass

    @property
    def left_hand(self):
        with self.left_hand_shared.get_lock():
            return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
    
    @property
    def right_hand(self):
        with self.right_hand_shared.get_lock():
            return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
    
    @property
    def left_landmarks(self):
        with self.left_landmarks_shared.get_lock():
            return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        with self.right_landmarks_shared.get_lock():
            return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        with self.head_matrix_shared.get_lock():
            return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        with self.aspect_shared.get_lock():
            return float(self.aspect_shared.value)
