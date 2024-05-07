import asyncio
import time

import cv2
import numpy as np
from numpy.typing import NDArray
from vuer import Vuer, VuerSession
from vuer.schemas import ImageBackground


app = Vuer()

IMAGE_WIDTH: int = 1280
IMAGE_HEIGHT: int = 480
aspect_ratio: float = IMAGE_WIDTH / IMAGE_HEIGHT
CAMERA_FPS: int = 60
MAX_FPS: int = 60
VUER_IMG_QUALITY: int = 20
BGR_TO_RGB: NDArray = np.array([2, 1, 0], dtype=np.uint8)

print("Starting camera")
img_lock = asyncio.Lock()
img: NDArray[np.uint8] = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
cam.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
ret, frame = cam.read()


async def update_image() -> None:
    global cam
    if not cam.isOpened():
        raise ValueError("Camera is not available")
    start = time.time()
    ret, frame = cam.read()
    if ret:
        async with img_lock:
            global img
            img = frame[:, :, BGR_TO_RGB]
    else:
        print("Failed to read frame")
    print(f"Time to update image: {time.time() - start}")


@app.spawn(start=True)
async def main(session: VuerSession):
    await asyncio.sleep(0.1)
    while True:
        await asyncio.gather(update_image(), asyncio.sleep(1 / MAX_FPS))
        async with img_lock:
            global img
            session.upsert(
                ImageBackground(
                    img,
                    format="jpg",  # TODO: test ['b64png', 'b64jpeg']
                    quality=VUER_IMG_QUALITY,
                    key="video",
                    interpolate=True,
                    fixed=True,
                    aspect=aspect_ratio,
                    distanceToCamera=5,
                    position=[0, 0, -5],
                    # rotation=[0, 0, 0],
                ),
                to="bgChildren",
            )
