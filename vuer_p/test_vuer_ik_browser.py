import asyncio
import os
import time
import math
from datetime import datetime
from typing import List, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p
import pybullet_data
from vuer import Vuer, VuerSession
from vuer.schemas import Movable, Gripper, Urdf, Scene, AmbientLight, PointLight

# ----- Stompy

robot_urdf_dir = f"{os.path.dirname(__file__)}/../urdf/stompy_tiny"
robot_urdf_path = f"{robot_urdf_dir}/robot.urdf"
robot_start_pos = [0, 0, 1]
robot_start_euler = [-math.pi / 4, 0, 0]  # only needed for pybullet somehow
ee_start_pos_right = [-0.2, -0.2, -0.2]
ee_start_pos_right = [g + r for g, r in zip(ee_start_pos_right, robot_start_pos)]
ee_start_pos_left = [0.2, -0.2, -0.2]
ee_start_pos_left = [g + r for g, r in zip(ee_start_pos_left, robot_start_pos)]
stompy_default_pos = {
    "joint_head_1_x4_1_dof_x4": -0.03490658503988659,
    "joint_legs_1_x8_2_dof_x8": 0.5061454830783556,
    "joint_legs_1_left_leg_1_x8_1_dof_x8": -0.5061454830783556,
    "joint_legs_1_left_leg_1_x10_1_dof_x10": 0.9773843811168246,
    "joint_legs_1_x8_1_dof_x8": -0.5061454830783556,
    "joint_legs_1_right_leg_1_x8_1_dof_x8": -0.5061454830783556,
    "joint_legs_1_right_leg_1_x10_2_dof_x10": -0.9773843811168246,
    "joint_legs_1_left_leg_1_knee_revolute": -0.10471975511965978,
    "joint_legs_1_right_leg_1_knee_revolute": 0.10471975511965978,
    "joint_legs_1_left_leg_1_ankle_revolute": 0.0,
    "joint_legs_1_right_leg_1_ankle_revolute": 0.0,
    "joint_legs_1_left_leg_1_x4_1_dof_x4": 0.0,
    "joint_legs_1_right_leg_1_x4_1_dof_x4": 0.0,
    # left arm
    "joint_left_arm_2_x8_1_dof_x8": -1.7,
    "joint_left_arm_2_x8_2_dof_x8": -1.6,
    "joint_left_arm_2_x6_1_dof_x6": -0.34,
    "joint_left_arm_2_x6_2_dof_x6": -1.6,
    "joint_left_arm_2_x4_1_dof_x4": -1.4,
    "joint_left_arm_2_hand_1_x4_1_dof_x4": -1.7,
    # right arm
    "joint_right_arm_1_x8_1_dof_x8": 1.7,
    "joint_right_arm_1_x8_2_dof_x8": 1.6,
    "joint_right_arm_1_x6_1_dof_x6": 0.34,
    "joint_right_arm_1_x6_2_dof_x6": 1.6,
    "joint_right_arm_1_x4_1_dof_x4": 1.4,
    "joint_right_arm_1_hand_1_x4_1_dof_x4": -0.26,
}
ee_link_right = "link_right_arm_1_hand_1_x4_2_outer_1"
ee_link_left = "link_left_arm_2_hand_1_x4_2_outer_1"
ee_chain_left = [
    "joint_left_arm_2_x8_1_dof_x8",
    "joint_left_arm_2_x8_2_dof_x8",
    "joint_left_arm_2_x6_1_dof_x6",
    "joint_left_arm_2_x6_2_dof_x6",
    "joint_left_arm_2_x4_1_dof_x4",
    "joint_left_arm_2_hand_1_x4_1_dof_x4",
]
ee_chain_right = [
    "joint_right_arm_1_x8_1_dof_x8",
    "joint_right_arm_1_x8_2_dof_x8",
    "joint_right_arm_1_x6_1_dof_x6",
    "joint_right_arm_1_x6_2_dof_x6",
    "joint_right_arm_1_x4_1_dof_x4",
    "joint_right_arm_1_hand_1_x4_1_dof_x4",
]
# IK solver outputs 37 length vector with DOF joints in the following order
ik_joints_names = [
    "joint_head_1_x4_1_dof_x4",
    "joint_head_1_x4_2_dof_x4",
    "joint_right_arm_1_x8_1_dof_x8",
    "joint_right_arm_1_x8_2_dof_x8",
    "joint_right_arm_1_x6_1_dof_x6",
    "joint_right_arm_1_x6_2_dof_x6",
    "joint_right_arm_1_x4_1_dof_x4",
    "joint_right_arm_1_hand_1_x4_1_dof_x4",
    "joint_right_arm_1_hand_1_slider_1",
    "joint_right_arm_1_hand_1_slider_2",
    "joint_right_arm_1_hand_1_x4_2_dof_x4",
    "joint_left_arm_2_x8_1_dof_x8",
    "joint_left_arm_2_x8_2_dof_x8",
    "joint_left_arm_2_x6_1_dof_x6",
    "joint_left_arm_2_x6_2_dof_x6",
    "joint_left_arm_2_x4_1_dof_x4",
    "joint_left_arm_2_hand_1_x4_1_dof_x4",
    "joint_left_arm_2_hand_1_slider_1",
    "joint_left_arm_2_hand_1_slider_2",
    "joint_left_arm_2_hand_1_x4_2_dof_x4",
    "joint_torso_1_x8_1_dof_x8",
    "joint_legs_1_x8_1_dof_x8",
    "joint_legs_1_right_leg_1_x8_1_dof_x8",
    "joint_legs_1_right_leg_1_x10_2_dof_x10",
    "joint_legs_1_right_leg_1_knee_revolute",
    "joint_legs_1_right_leg_1_x10_1_dof_x10",
    "joint_legs_1_right_leg_1_ankle_revolute",
    "joint_legs_1_right_leg_1_x6_1_dof_x6",
    "joint_legs_1_right_leg_1_x4_1_dof_x4",
    "joint_legs_1_x8_2_dof_x8",
    "joint_legs_1_left_leg_1_x8_1_dof_x8",
    "joint_legs_1_left_leg_1_x10_1_dof_x10",
    "joint_legs_1_left_leg_1_knee_revolute",
    "joint_legs_1_left_leg_1_x10_2_dof_x10",
    "joint_legs_1_left_leg_1_ankle_revolute",
    "joint_legs_1_left_leg_1_x6_1_dof_x6",
    "joint_legs_1_left_leg_1_x4_1_dof_x4",
]

# ----- PyBullet
clid = p.connect(p.SHARED_MEMORY)
if clid < 0:
    p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
id = p.loadURDF(robot_urdf_path, [0, 0, 0], useFixedBase=True)
p.setGravity(0, 0, 0)
p.resetBasePositionAndOrientation(
    id, robot_start_pos, p.getQuaternionFromEuler(robot_start_euler)
)
numJoints = p.getNumJoints(id)
jNames = []
jChildNames = []
jUpperLimit = []
jLowerLimit = []
jRestPose = []
jDamping = []
j_dof_idx_map = {}
# https://github.com/bulletphysics/bullet3/blob/e9c461b0ace140d5c73972760781d94b7b5eee53/examples/SharedMemory/SharedMemoryPublic.h#L292
for i in range(numJoints):
    info = p.getJointInfo(id, i)
    joint_name = info[1].decode("utf-8")
    jNames.append(joint_name)
    jChildNames.append(info[12].decode("utf-8"))
    jLowerLimit.append(info[9])
    jUpperLimit.append(info[10])
    if joint_name in stompy_default_pos:
        jRestPose.append(stompy_default_pos[joint_name])
        jDamping.append(0.1)
    else:
        jRestPose.append(0)
        jDamping.append(10)
ee_idx_right = jChildNames.index(ee_link_right)
ee_idx_left = jChildNames.index(ee_link_left)
for i in range(numJoints):
    p.resetJointState(id, i, jRestPose[i])
ll = jLowerLimit
ul = jUpperLimit
jr = [ul[i] - ll[i] for i in range(numJoints)]
rp = jRestPose
jd = jDamping
joints = jRestPose.copy()
ee_goal_pos_left = ee_start_pos_left
ee_goal_orn_left = p.getQuaternionFromEuler([-math.pi / 2, 0, 0])
ee_goal_pos_right = ee_start_pos_right
ee_goal_orn_right = p.getQuaternionFromEuler([-math.pi / 2, 0, 0])


def joint_list_to_dict(joint_list: List[float]) -> Dict[str, float]:
    joint_dict = {}
    for joint in ik_joints_names:
        joint_dict[joint] = joint_list[jNames.index(joint)]
    return joint_dict


def make_full_joint_list(ik_joints: List[float]) -> List[float]:
    full_joints = jRestPose.copy()
    for i, j in enumerate(ik_joints):
        full_joints[jNames.index(ik_joints_names[i])] = j
    return full_joints


def update_joint_list(
    joint_list: List[float], joint_dict: Dict[str, float]
) -> List[float]:
    for i, val in enumerate(joint_list):
        joint_list[i] = joint_dict[jNames[i]]
    return joint_list


async def ik(lock, arm):
    # start_time = time.time()
    if arm == "left":
        ee_idx = ee_idx_left
        ee_chain = ee_chain_left
        pos = ee_goal_pos_left
        orn = ee_goal_orn_left
    else:
        ee_idx = ee_idx_right
        ee_chain = ee_chain_right
        pos = ee_goal_pos_right
        orn = ee_goal_orn_right
    # print(f"ik {arm} {pos} {orn}")
    jointPoses = p.calculateInverseKinematics(id, ee_idx, pos, orn, ll, ul, jr, rp)
    jointPoses = make_full_joint_list(jointPoses)
    async with lock:
        global joints
        for j in ee_chain:
            _index = jNames.index(j)
            joints[_index] = jointPoses[_index]
            p.resetJointState(id, _index, jointPoses[_index])
    # end_time = time.time()
    # print(f"ik {arm} took {end_time - start_time} seconds")


# ----- Vuer
app = Vuer(static_root=robot_urdf_dir + "_glb")


@app.add_handler("OBJECT_MOVE")
async def move_handler(event, session):
    print("------------------------")
    global ee_goal_pos_left, ee_goal_orn_left, ee_goal_pos_right, ee_goal_orn_right
    hand_T = np.array(event.value["matrix"]).reshape(4, 4).T
    # print(f"hand_T {event.key} (vuer frame) \n{hand_T}\n")
    # Convert from PyBullet to Vuer Frame
    # Flip Y axis
    hand_T[1, 3] = -hand_T[1, 3]
    # Flip X axis
    hand_T[0, 3] = -hand_T[0, 3]
    # print(f"hand_T {event.key} (pybullet frame) \n{hand_T}\n")
    # Orientation starts with rotation matrix
    hand_R = R.from_matrix(hand_T[:3, :3])
    hand_R *= R.from_euler("xyz", [-math.pi / 2, 0, 0])
    hand_orn = hand_R.as_quat()
    if event.key == "left":
        ee_goal_pos_left = hand_T[:3, 3]
        ee_goal_orn_left = hand_orn
    else:
        ee_goal_pos_right = hand_T[:3, 3]
        ee_goal_orn_right = hand_orn


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
    session.upsert @ Movable(Gripper(), position=ee_start_pos_right, key="right")
    session.upsert @ Movable(Gripper(), position=ee_start_pos_left, key="left")
    await asyncio.sleep(0.1)
    global joints
    lock = asyncio.Lock()
    async with lock:
        session.upsert @ Urdf(
            src="http://localhost:8012/static/robot.urdf",
            jointValues=joint_list_to_dict(joints),
            position=robot_start_pos,
            key="robot",
        )
    while True:
        # await asyncio.sleep(1)
        await asyncio.gather(
            ik(lock, "left"),
            ik(lock, "right"),
        )
        async with lock:
            session.upsert @ Urdf(
                src="http://localhost:8012/static/robot.urdf",
                jointValues=joint_list_to_dict(joints),
                position=robot_start_pos,
                key="robot",
            )
