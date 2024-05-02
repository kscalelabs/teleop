from asyncio import sleep
import os
import math

import numpy as np
import roboticstoolbox as rtb
from vuer import Vuer, VuerSession
from vuer.schemas import Movable, Gripper, Urdf, Scene, AmbientLight, PointLight

from src.stompy import StompyFixed

app = Vuer(static_root=f"{os.path.dirname(__file__)}/urdf/stompy_tiny")

stompy_rtb = rtb.robot.Robot.URDF(f"{os.path.dirname(__file__)}/urdf/stompy_tiny/robot.urdf")
left_arm_chain = [
    "joint_left_arm_2_x8_1_dof_x8",
    "joint_left_arm_2_x8_2_dof_x8",
    "joint_left_arm_2_x6_1_dof_x6",
    "joint_left_arm_2_x6_2_dof_x6",
    "joint_left_arm_2_x4_1_dof_x4",
    "joint_left_arm_2_hand_1_x4_1_dof_x4",
]
right_arm_chain = [
    "joint_right_arm_1_x8_1_dof_x8",
    "joint_right_arm_1_x8_2_dof_x8",
    "joint_right_arm_1_x6_1_dof_x6",
    "joint_right_arm_1_x6_2_dof_x6",
    "joint_right_arm_1_x4_1_dof_x4",
    "joint_right_arm_1_hand_1_x4_2_dof_x4",
]

@app.add_handler("OBJECT_MOVE")
async def move_handler(event, session):
    if event.key == "left":
        Tep = np.array(event.value["matrix"])
        result = stompy_rtb.ik_LM(
            Tep,
            start="link_left_arm_2_x8_1_outer_1",
            end="link_left_arm_2_hand_1_x4_1_inner_1",
            tol=0.1,
            mask=[1, 1, 1, 0, 0, 0], # which DOF to solve for (this is just xyz)
            # q0=q0, # intial guess for faster solve
            # joint_limits=True,
        )
        q, success, num_iter, num_search, residual = result
        print(f"success: {bool(success)} after {num_iter} iterations")
        print(f'\t q: {q}')
        session.upsert @ Urdf(
            src="http://localhost:8012/static/robot.urdf",
            jointValues={k: v for k, v in zip(left_arm_chain, q)},
            position=[0, 0, 1],
            key="robot",
        )
    elif event.key == "right":
        Tep = np.array(event.value["matrix"])
        result = stompy_rtb.ik_LM(
            Tep,
            start="link_right_arm_1_x8_1_outer_1",
            end="link_right_arm_1_hand_1_x4_2_inner_1",
            tol=0.1,
            mask=[1, 1, 1, 0, 0, 0], # which DOF to solve for (this is just xyz)
            # q0=q0, # intial guess for faster solve
            # joint_limits=True,
        )
        q, success, num_iter, num_search, residual = result
        print(f"success: {bool(success)} after {num_iter} iterations")
        session.upsert @ Urdf(
            src="http://localhost:8012/static/robot.urdf",
            jointValues={k: v for k, v in zip(right_arm_chain, q)},
            position=[0, 0, 1],
            key="robot",
        )

@app.spawn(start=True)
async def main(app: VuerSession):
    app.set @ Scene(
        rawChildren=[
            AmbientLight(intensity=1),
            PointLight(intensity=1, position=[0, 0, 2]),
            PointLight(intensity=3, position=[0, 1, 2]),
        ],
        grid=True,
        up=[0, 0, 1],
    )
    app.upsert @ Movable(Gripper(), position=[-0.1, -0.3, 0.75], key="right")
    app.upsert @ Movable(Gripper(), position=[0.1, -0.3, 0.75], key="left")
    await sleep(0.1)
    app.upsert @ Urdf(
        src="http://localhost:8012/static/robot.urdf",
        jointValues=StompyFixed.default_standing(),
        position=[0, 0, 1],
        key="robot",
    )
    while True:
        await sleep(1)
