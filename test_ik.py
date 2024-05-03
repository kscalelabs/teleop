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
q0_left = [0, 0, 0, 0, 0, 0]

@app.add_handler("OBJECT_MOVE")
async def move_handler(event, session):
    if event.key == "left":
        print("------------------------")
        Tep = np.array(event.value["matrix"]).reshape(4, 4).T
        print(f'Tep: {Tep}')
        print(f'Target xyz position: {Tep[:3, 3]}')
        # Change reference frame to match ROS URDF from THREE.js
        # Aligning THREE.js (Y-up, Z-forward) to ROS URDF (Z-up, X-forward)
        # transform_matrix = np.array([
        #     [1, 0, 0, 0],  # Keep X as X
        #     [0, 0, 1, 0],  # Map Z (THREE.js) to Y (ROS URDF)
        #     [0, 1, 0, 0],  # Map Y (THREE.js) to Z (ROS URDF)
        #     [0, 0, 0, 1]   # Homogeneous coordinate remains unchanged
        # ])
        # Tep = np.dot(Tep, transform_matrix)
        Tep = np.array([
            [1, 0, 0, 0.2],
            [0, 1, 0, -0.2],
            [0, 0, 1, 0],
            [0, 0, 0, 1] 
        ])
        print(f'Corrected Tep: {Tep}')
        print(f'Corrected Target xyz position: {Tep[:3, 3]}')
        result = stompy_rtb.ik_LM(
            Tep,
            end="link_right_arm_1_hand_1_base_1",
            tol=1e-3,
            mask=[1, 1, 10, 0, 0, 0], # which DOF to solve for (this is just xyz)
            q0=q0_left, # intial guess for faster solve
            # joint_limits=True,
        )
        q, success, num_iter, num_search, residual = result
        print(f"success: {bool(success)} after {num_iter} iterations")
        print(f'\t q: {q}')
        Teg = stompy_rtb.fkine(q, end="link_right_arm_1_hand_1_base_1")
        Teg = np.array(Teg)
        print(f'Teg: {Teg}')
        print(f'Actual xyz position: {Teg[:3, 3]}')
        session.upsert @ Urdf(
            src="http://localhost:8012/static/robot.urdf",
            jointValues={k: v for k, v in zip(left_arm_chain, q)},
            position=[0, 0, 1],
            key="robot",
        )
    elif event.key == "right":
        print("------------------------")
        Tep = np.array(event.value["matrix"]).reshape(4, 4).T
        print(f'Tep: {Tep}')
        print(f'Target xyz position: {Tep[:3, 3]}')
        # Change reference frame to match ROS URDF from THREE.js
        # Aligning THREE.js (Y-up, Z-forward) to ROS URDF (Z-up, X-forward)
        # transform_matrix = np.array([
        #     [1, 0, 0, 0],  # Keep X as X
        #     [0, 0, 1, 0],  # Map Z (THREE.js) to Y (ROS URDF)
        #     [0, 1, 0, 0],  # Map Y (THREE.js) to Z (ROS URDF)
        #     [0, 0, 0, 1]   # Homogeneous coordinate remains unchanged
        # ])
        # Tep = np.dot(Tep, transform_matrix)
        Tep = np.array([
            [1, 0, 0, -0.2],
            [0, 1, 0, -0.2],
            [0, 0, 1, 0],
            [0, 0, 0, 1] 
        ])
        print(f'Corrected Tep: {Tep}')
        print(f'Corrected Target xyz position: {Tep[:3, 3]}')
        result = stompy_rtb.ik_LM(
            Tep,
            end="link_right_arm_1_hand_1_base_1",
            tol=1e-3,
            mask=[1, 1, 1, 0, 0, 0], # which DOF to solve for (this is just xyz)
            # q0 = [1.7, 1.6, 0.34, 1.6, 1.4, -0.26],
            q0=[-1.86360008,  0.76775798,  0.36221785,  0.58700685, -1.55511321, -1.90407512],
            # joint_limits=True,
        )
        q, success, num_iter, num_search, residual = result
        print(f"success: {bool(success)} after {num_iter} iterations")
        print(f'\t q: {q}')
        Teg = stompy_rtb.fkine(q, end="link_right_arm_1_hand_1_base_1")
        Teg = np.array(Teg)
        print(f'Teg: {Teg}')
        print(f'Actual xyz position: {Teg[:3, 3]}')
        session.upsert @ Urdf(
            src="http://localhost:8012/static/robot.urdf",
            jointValues={
                "joint_right_arm_1_x8_1_dof_x8" : q[0],
                "joint_right_arm_1_x8_2_dof_x8" : q[1],
                "joint_right_arm_1_x6_1_dof_x6" : q[2],
                "joint_right_arm_1_x6_2_dof_x6" : q[3],
                "joint_right_arm_1_x4_1_dof_x4" : q[4],
                "joint_right_arm_1_hand_1_x4_1_dof_x4" : q[5],
                # "joint_right_arm_1_x8_1_dof_x8" : q[5],
                # "joint_right_arm_1_x8_2_dof_x8" : q[4],
                # "joint_right_arm_1_x6_1_dof_x6" : q[3],
                # "joint_right_arm_1_x6_2_dof_x6" : q[2],
                # "joint_right_arm_1_x4_1_dof_x4" : q[1],
                # "joint_right_arm_1_hand_1_x4_1_dof_x4" : q[0],
            },
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
    # app.upsert @ Movable(Gripper(), position=[-0.1, -0.3, 0.0], key="right")
    # app.upsert @ Movable(Gripper(), position=[0.1, -0.3, 0.0], key="left")
    await sleep(0.1)
    app.upsert @ Urdf(
        src="http://localhost:8012/static/robot.urdf",
        jointValues=StompyFixed.default_standing(),
        position=[0, 0, 1],
        key="robot",
    )
    while True:
        await sleep(1)
