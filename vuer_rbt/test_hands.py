import math
from typing import Tuple

from scipy.spatial.transform import Rotation as R
import numpy as np
from vuer import Vuer, VuerSession
from vuer.schemas import Hands, Movable, Gripper
from vuer.events import Event
from asyncio import sleep

app = Vuer()


# "index-finger-tip" and has idx of 09
# "thumb-tip" and has idx of 04
# "middle-finger-tip" and has idx of 14
# fully open pinch is around 0.10 distance
# fully closed pinch is around 0.01 distance
def detect_pinch(
    event: Event, hand: str, finger: int, min_distance: float = 0.01
) -> Tuple[bool, float]:
    finger_tip_position = np.array(event.value[f"{hand}Landmarks"][finger])
    thumb_tip_position = np.array(event.value[f"{hand}Landmarks"][4])
    distance = np.linalg.norm(finger_tip_position - thumb_tip_position)
    if distance < min_distance:
        return True, distance
    return False, distance


@app.add_handler("HAND_MOVE")
async def hand_handler(event, session):
    # middle finger pinch turns on tracking
    left_active, _ = detect_pinch(event, "left", 14)
    right_active, _ = detect_pinch(event, "right", 14)
    # index finger pinch determines gripper position
    _, left_dist = detect_pinch(event, "left", 9)
    _, right_dist = detect_pinch(event, "right", 9)


    if left_active:
        print("Pinch detected in left hand")
        RT = event.value["leftHand"]
        x, y, z = RT[12], RT[13], RT[14]
        rx, ry, rz = R.from_matrix(
            [[RT[0], RT[4], RT[7]], [RT[1], RT[5], RT[8]], [RT[2], RT[6], RT[9]]]
        ).as_euler("xyz")
        session.upsert @ Movable(Gripper(position=[x, y, z], rotation=[rx, ry, rz]), key="left")
    if right_active:
        print("Pinch detected in right hand")
        RT = event.value["rightHand"]
        x, y, z = RT[12], RT[13], RT[14]
        rx, ry, rz = R.from_matrix(
            [[RT[0], RT[4], RT[7]], [RT[1], RT[5], RT[8]], [RT[2], RT[6], RT[9]]]
        ).as_euler("xyz")
        session.upsert @ Movable(Gripper(position=[x, y, z], rotation=[rx, ry, rz]), key="right")

@app.add_handler("OBJECT_MOVE")
async def move_handler(event, session):
    print(f"Movement Event: key-{event.key}", event.value)


@app.spawn(start=True)
async def main(session: VuerSession):
    session.upsert @ Hands(fps=60, stream=True, key="hands")
    session.upsert @ Movable(Gripper(), key="left")
    session.upsert @ Movable(Gripper(), key="right")
    while True:
        await sleep(1)
