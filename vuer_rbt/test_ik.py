from asyncio import sleep
import os

import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from vuer import Vuer, VuerSession
from vuer.schemas import Movable, Gripper, Urdf, Scene, AmbientLight, PointLight

from src.stompy import StompyFixed, joint_dict_to_list, qright_to_dict, qleft_to_dict

app = Vuer(static_root=f"{os.path.dirname(__file__)}/urdf/stompy_tiny_glb")
robot_pos = [0, 0, 1]
left_ee_start = [0.2479, -0.3074, -0.1344]
left_ee_start = [p + r for p, r in zip(left_ee_start, robot_pos)]
right_ee_start = [-0.2479, -0.3074, -0.1344]
right_ee_start = [p + r for p, r in zip(right_ee_start, robot_pos)]

stompy_rtb = rtb.robot.Robot.URDF(
    f"{os.path.dirname(__file__)}/urdf/stompy_tiny_glb/robot.urdf"
)
robot_inv = sm.SE3.Trans([-rp for rp in robot_pos])
q_left = [-1.7, -1.6, -0.34, -1.6, -1.4, -1.7]
q_right = [1.7, 1.6, 0.34, 1.6, 1.4, -0.26]
joints = StompyFixed.default_standing()
joints.update(qright_to_dict(q_right))
joints.update(qleft_to_dict(q_left))
T_left = sm.SE3()
T_right = sm.SE3()
tol = 1e-2
mask = [1, 1, 1, 0, 0, 0]
vel = 1e-2


async def move_to(T_hand, ee_link, q0, q_map):
    global joints, q_left, q_right
    print(f"q0\n{q0}\n")
    T_ee = stompy_rtb.fkine(q0, end=ee_link)
    print(f"T_ee\n{T_ee}\n")
    _T = np.eye(4)
    _T[:3, :3] = T_ee.R
    _T[:3, 3] = T_ee.t + np.linalg.norm((T_hand.t - T_ee.t)) * vel
    print(f"_T\n{_T}\n")
    result = stompy_rtb.ik_LM(
        _T,
        end=ee_link,
        tol=tol,
        mask=mask,
        q0=q0,
        joint_limits=False,
        # pinv=True,
    )
    q, success, num_iter, num_search, _ = result
    print(f"--- [{bool(success)}] after {num_iter} iterations {num_search} searches")
    if success:
        joints.update(q_map(q))
        if ee_link == "link_left_arm_2_hand_1_x4_2_outer_1":
            q_left = q
        else:
            q_right = q
        # q_full = joint_dict_to_list([StompyFixed().default_standing(), qleft_to_dict(q)])
        # Tr_full = stompy_rtb.fkine(q_full)
        # print(f"Tr_full\n{Tr_full}\n")
        # # stompy_rtb.plot(q_full)


@app.add_handler("OBJECT_MOVE")
async def move_handler(event, session):
    print("------------------------")
    global T_left, T_right
    T_hand = np.array(event.value["matrix"]).reshape(4, 4).T
    print(f"T_hand raw\n{T_hand}\n")
    T_hand = sm.SE3(T_hand) * robot_inv
    print(f"T_hand w/ robot\n{T_hand}\n")
    if event.key == "left":
        T_left = T_hand
    elif event.key == "right":
        T_right = T_hand



@app.spawn(start=True)
async def main(session: VuerSession):
    global joints, q_left, q_right, T_left, T_right
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
        jointValues=joints,
        position=robot_pos,
        key="robot",
    )
    while True:
        await sleep(1 / 100.0)
        await move_to(T_left, "link_left_arm_2_hand_1_x4_2_outer_1", q_left, qleft_to_dict)
        await move_to(T_right, "link_right_arm_1_hand_1_x4_2_outer_1", q_right, qright_to_dict)
        # print(f"q\n{joints}\n")
        session.upsert @ Urdf(
            src="http://localhost:8012/static/robot.urdf",
            jointValues=joints,
            position=robot_pos,
            key="robot",
        )
