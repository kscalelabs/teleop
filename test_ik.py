from asyncio import sleep
import os

import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from vuer import Vuer, VuerSession
from vuer.schemas import Movable, Gripper, Urdf, Scene, AmbientLight, PointLight

from src.stompy import StompyFixed

app = Vuer(static_root=f"{os.path.dirname(__file__)}/urdf/stompy_tiny")
robot_pos = [0, 0, 1]
left_ee_start = [0.2479, -0.3074, -0.1344]
left_ee_start = [p + r for p, r in zip(left_ee_start, robot_pos)]
right_ee_start = [-0.2479, -0.3074, -0.1344]
right_ee_start = [p + r for p, r in zip(right_ee_start, robot_pos)]

stompy_rtb = rtb.robot.Robot.URDF(
    f"{os.path.dirname(__file__)}/urdf/stompy_tiny/robot.urdf"
)
robot_inv = sm.SE3.Trans([-rp for rp in robot_pos])
q_left = [-1.7, -1.6, -0.34, -1.6, -1.4, -1.7]
q_right = [1.7, 1.6, 0.34, 1.6, 1.4, -0.26]
tol = 1e-3
mask = [1, 1, 1, 0, 0, 0]

@app.add_handler("OBJECT_MOVE")
async def move_handler(event, session):
    print("------------------------")
    global q_left, q_right
    if event.key == "left":
        Tg = np.array(event.value["matrix"]).reshape(4, 4).T
        print(f"Tg raw\n{Tg}\n")
        Tg = sm.SE3(Tg) * robot_inv
        print(f"Tg robot\n{Tg}\n")
        print(f"q0\n{q_left}\n")
        result = stompy_rtb.ik_LM(
            Tg,
            end="link_left_arm_2_hand_1_x4_2_outer_1",
            tol=tol,
            mask=mask,
            q0=q_left,
            joint_limits=False,
        )
        q, success, num_iter, num_search, residual = result
        print(f"success: {bool(success)} after {num_iter} iterations {num_search} searches (residual: {residual} < {tol})")
        # if success:    
        #     q_left = q
        print(f"qt\n{q_left}\n")
        session.upsert @ Urdf(
            src="http://localhost:8012/static/robot.urdf",
            jointValues={
                "joint_left_arm_2_x8_1_dof_x8": q_left[0],
                "joint_left_arm_2_x8_2_dof_x8": q_left[1],
                "joint_left_arm_2_x6_1_dof_x6": q_left[2],
                "joint_left_arm_2_x6_2_dof_x6": q_left[3],
                "joint_left_arm_2_x4_1_dof_x4": q_left[4],
                "joint_left_arm_2_hand_1_x4_1_dof_x4": q_left[5],
            },
            position=robot_pos,
            key="robot",
        )
    elif event.key == "right":
        Tg = np.array(event.value["matrix"]).reshape(4, 4).T
        print(f"Tg raw\n{Tg}\n")
        Tg = sm.SE3(Tg) * robot_inv
        print(f"Tg robot\n{Tg}\n")
        print(f"q0\n{q_right}\n")
        result = stompy_rtb.ik_LM(
            Tg,
            end="link_right_arm_1_hand_1_x4_2_outer_1",
            tol=tol,
            mask=mask,
            q0=q_right,
            joint_limits=False,
        )
        q, success, num_iter, num_search, residual = result
        print(f"success: {bool(success)} after {num_iter} iterations {num_search} searches (residual: {residual} < {tol})")
        # if success:
        #     q_right = q
        print(f"qt\n{q_right}\n")
        session.upsert @ Urdf(
            src="http://localhost:8012/static/robot.urdf",
            jointValues={
                "joint_right_arm_1_x8_1_dof_x8": q_right[0],
                "joint_right_arm_1_x8_2_dof_x8": q_right[1],
                "joint_right_arm_1_x6_1_dof_x6": q_right[2],
                "joint_right_arm_1_x6_2_dof_x6": q_right[3],
                "joint_right_arm_1_x4_1_dof_x4": q_right[4],
                "joint_right_arm_1_hand_1_x4_1_dof_x4": q_right[5],
            },
            position=robot_pos,
            key="robot",
        )


@app.spawn(start=True)
async def main(session: VuerSession):
    session.set @ Scene(
        rawChildren=[
            AmbientLight(intensity=1),
            PointLight(intensity=1, position=[0, 0, 2]),
            PointLight(intensity=3, position=[0, 1, 2]),
        ],
        grid=True,
        up=[0, 0, 1],
    )
    session.upsert @ Movable(Gripper(), position=right_ee_start, key="right")
    session.upsert @ Movable(Gripper(), position=left_ee_start, key="left")
    await sleep(0.1)
    session.upsert @ Urdf(
        src="http://localhost:8012/static/robot.urdf",
        jointValues=StompyFixed.default_standing(),
        position=robot_pos,
        key="robot",
    )
    while True:
        await sleep(1)
