import asyncio
from copy import deepcopy
import os
import math
from typing import List, Dict
import time
import numpy as np
from numpy.typing import NDArray
import pybullet as p
import pybullet_data
from vuer import Vuer, VuerSession
from vuer.schemas import Hands, PointLight, Urdf

# web urdf is used for vuer
URDF_WEB: str = (
    "https://raw.githubusercontent.com/kscalelabs/webstompy/pawel/new_stomp/urdf/stompy_new/upper_limb_assembly_5_dof_merged_simplified.urdf"
)
# local urdf is used for pybullet
URDF_LOCAL: str = f"urdf/stompy_new/Edited2.urdf"

# starting positions for robot trunk relative to world frames
START_POS_TRUNK_VUER: NDArray = np.array([0, 1., 0])
START_EUL_TRUNK_VUER: NDArray = np.array([-math.pi, -3.8, 0])
START_POS_TRUNK_PYBULLET: NDArray = np.array([0, 0, 1.])
START_EUL_TRUNK_PYBULLET: NDArray = np.array([-math.pi/2, 0, -1.])
VIEWER_TO_PYBULLET_SCALE = 1.0  # Adjust this value if needed

# starting positions for robot end effectors are defined relative to robot trunk frame
START_POS_EER_VUER: NDArray = np.array([-.2, .3, .2]) + START_POS_TRUNK_PYBULLET
START_POS_EEL_VUER: NDArray = np.array([0.3, -0.1, .3]) + START_POS_TRUNK_PYBULLET

# conversion between PyBullet and Vuer axes
VUER_TO_PB_AXES: NDArray = np.array([2, 0, 1], dtype=np.uint8)
VUER_TO_PB_AXES_SIGN: NDArray = np.array([1, 1, 1], dtype=np.int8)

# starting joint positions (Q means "joint angles")
START_Q: Dict[str, float] = {
    # torso
    "joint_torso_1_rmd_x8_90_mock_1_dof_x8": 0,

    # left arm (5dof)
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": 2.61,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": -1.38,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": 0,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": 2.83,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": 1.32,

    # left hand (2dof)
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_1": 0.0,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2": 0.0,
}   

# link names are based on the URDF
EEL_LINK: str = "end_effector_link" #need to make left and right version
EER_LINK: str = "end_effector_link" 

# kinematic chains for each arm and hand
EEL_CHAIN_ARM: List[str] = [
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4',
]
EEL_CHAIN_HAND: List[str] = [
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_1', 
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2'
]

EER_CHAIN_ARM: List[str] = [
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8",
]
EER_CHAIN_HAND: List[str] = [
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_1",
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_2",
]

# PyBullet IK will output a 37dof list in this exact order
IK_Q_LIST: List[str] = [
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4',
    # 'joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_1', 
    # 'joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2'
]

# PyBullet inverse kinematics (IK) params
DAMPING_CHAIN: float = 0.1
DAMPING_NON_CHAIN: float = 10.0

# PyBullet init
HEADLESS: bool = False
if HEADLESS:
    print("Starting PyBullet in headless mode.")
    clid = p.connect(p.DIRECT)
else:
    print("Starting PyBullet in GUI mode.")
    clid = p.connect(p.SHARED_MEMORY)
    if clid < 0:
        p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
pb_robot_id = p.loadURDF(URDF_LOCAL, [0, 0, 0], useFixedBase=True)
p.setGravity(0, 0, -9.81)

pb_num_joints: int = p.getNumJoints(pb_robot_id)

print(f"\t number of joints: {pb_num_joints}")
pb_joint_names: List[str] = [""] * pb_num_joints
pb_child_link_names: List[str] = [""] * pb_num_joints
pb_joint_upper_limit: List[float] = [0.0] * pb_num_joints
pb_joint_lower_limit: List[float] = [0.0] * pb_num_joints
pb_joint_ranges: List[float] = [0.0] * pb_num_joints
pb_start_q: List[float] = [0.0] * pb_num_joints
pb_damping: List[float] = [0.0] * pb_num_joints
pb_q_map: Dict[str, int] = {}
for i in range(pb_num_joints):
    info = p.getJointInfo(pb_robot_id, i)
    name = info[1].decode("utf-8")
    pb_joint_names[i] = name
    pb_child_link_names[i] = info[12].decode("utf-8")
    pb_joint_lower_limit[i] = info[8]
    pb_joint_upper_limit[i] = info[9]
    pb_joint_ranges[i] = abs(info[9] - info[8])
    if name in START_Q:
        pb_start_q[i] = START_Q[name]
    if name in EER_CHAIN_ARM or name in EEL_CHAIN_ARM:
        pb_damping[i] = DAMPING_CHAIN
    else:
        pb_damping[i] = DAMPING_NON_CHAIN
    if name in IK_Q_LIST:
        pb_q_map[name] = i

pb_eer_id = pb_child_link_names.index(EER_LINK)
pb_eel_id = pb_child_link_names.index(EEL_LINK)

for i in range(pb_num_joints):
    p.resetJointState(pb_robot_id, i, pb_start_q[i])

p.resetBasePositionAndOrientation(
    pb_robot_id,
    START_POS_TRUNK_PYBULLET,
    p.getQuaternionFromEuler(START_EUL_TRUNK_PYBULLET),
)

# Set the camera view
target_position = START_POS_TRUNK_PYBULLET
camera_distance = 2.0
camera_yaw = 50
camera_pitch = -35

p.resetDebugVisualizerCamera(
    cameraDistance=camera_distance,
    cameraYaw=camera_yaw,
    cameraPitch=camera_pitch,
    cameraTargetPosition=target_position
)

# Vuer rendering params
MAX_FPS: int = 60
VUER_LIGHT_POS: NDArray = np.array([0, 2, 2])
VUER_LIGHT_INTENSITY: float = 10.0

# Vuer hand tracking and pinch detection params
HAND_FPS: int = 30
INDEX_FINGER_TIP_ID: int = 9
THUMB_FINGER_TIP_ID: int = 4
MIDDLE_FINGER_TIP_ID: int = 14
PINCH_DIST_OPENED: float = 0.10  # 10cm
PINCH_DIST_CLOSED: float = 0.01  # 1cm

# pre-compute gripper "slider" ranges for faster callback
EE_S_MIN: float = -0.034
EE_S_MAX: float = 0.0
ee_s_range: float = EE_S_MAX - EE_S_MIN

# global variables get updated by various async functions
q_lock = asyncio.Lock()
q = deepcopy(START_Q)
goal_pos_eer: NDArray = START_POS_EER_VUER
goal_orn_eer: NDArray = p.getQuaternionFromEuler(START_EUL_TRUNK_VUER)
goal_pos_eel: NDArray = START_POS_EEL_VUER
goal_orn_eel: NDArray = p.getQuaternionFromEuler([0, 0, 0])

# Add a red point at the goal position
p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)

async def ik(arm: str, max_attempts=20, max_iterations=100) -> float:
    global goal_pos_eer, goal_orn_eer, goal_pos_eel, goal_orn_eel, q

    if arm == "right":
        ee_id = pb_eer_id
        ee_chain = EER_CHAIN_ARM
        target_pos = goal_pos_eer
    else:
        ee_id = pb_eel_id
        ee_chain = EEL_CHAIN_ARM
        target_pos = goal_pos_eel

    best_error = float('inf')
    best_solution = None

    for attempt in range(max_attempts):
        initial_guess = [np.random.uniform(pb_joint_lower_limit[pb_q_map[joint]], 
                                           pb_joint_upper_limit[pb_q_map[joint]]) 
                         for joint in ee_chain]

        solution = p.calculateInverseKinematics(
            pb_robot_id,
            ee_id,
            target_pos,
            lowerLimits=[pb_joint_lower_limit[pb_q_map[joint]] for joint in ee_chain],
            upperLimits=[pb_joint_upper_limit[pb_q_map[joint]] for joint in ee_chain],
            jointRanges=[pb_joint_ranges[pb_q_map[joint]] for joint in ee_chain],
            restPoses=initial_guess,
            maxNumIterations=max_iterations,
            residualThreshold=1e-5
        )

        # Apply the solution
        for i, joint in enumerate(ee_chain):
            p.resetJointState(pb_robot_id, pb_q_map[joint], solution[i])
        p.stepSimulation()

        # Check the error
        actual_pos, _ = p.getLinkState(pb_robot_id, ee_id)[:2]
        error = np.linalg.norm(np.array(target_pos) - np.array(actual_pos))

        if error < best_error:
            best_error = error
            best_solution = solution

        print(f"Attempt {attempt + 1}: Error = {error}")

        if error < 0.01:  # 1cm tolerance
            break

    if best_solution:
        async with q_lock:
            for i, joint in enumerate(ee_chain):
                q[joint] = best_solution[i]
                p.resetJointState(pb_robot_id, pb_q_map[joint], best_solution[i])

    print(f"Best solution found with error: {best_error}")
    print(f"Final position: {p.getLinkState(pb_robot_id, ee_id)[0]}")
    print(f"Goal position: {target_pos}")

    # Visualize the error
    actual_pos, _ = p.getLinkState(pb_robot_id, ee_id)[:2]
    p.addUserDebugLine(actual_pos, target_pos, [1, 0, 0], 2, 0)

    p.stepSimulation()

    return best_error

app = Vuer()

@app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
    # right hand
    rindex_pos: NDArray = np.array(event.value["rightLandmarks"][INDEX_FINGER_TIP_ID])
    rthumb_pos: NDArray = np.array(event.value["rightLandmarks"][THUMB_FINGER_TIP_ID])
    rpinch_dist: NDArray = np.linalg.norm(rindex_pos - rthumb_pos)
    # index finger to thumb pinch turns on tracking
    if rpinch_dist < PINCH_DIST_CLOSED:
        global goal_pos_eel
        # Convert from viewer coordinates to PyBullet coordinates
        pybullet_pos = np.array([
            -rthumb_pos[2],
            rthumb_pos[0],
            rthumb_pos[1]
        ]) * VIEWER_TO_PYBULLET_SCALE + START_POS_TRUNK_PYBULLET
        goal_pos_eel = pybullet_pos
        print(f"New goal_pos_eel: {goal_pos_eel}")
        
        # Update the visualized goal position
        p.removeAllUserDebugItems()
        p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)

    # left hand
    lindex_pos: NDArray = np.array(event.value["leftLandmarks"][INDEX_FINGER_TIP_ID])
    lthumb_pos: NDArray = np.array(event.value["leftLandmarks"][THUMB_FINGER_TIP_ID])
    lpinch_dist: NDArray = np.linalg.norm(lindex_pos - lthumb_pos)
    # You can add left hand control logic here if needed

# Convert PyBullet position and orientation to viewer coordinates
VIEWER_POS_TRUNK = np.array([START_POS_TRUNK_PYBULLET[1], START_POS_TRUNK_PYBULLET[2], -START_POS_TRUNK_PYBULLET[0]])
VIEWER_ROT_TRUNK = np.array([START_EUL_TRUNK_PYBULLET[1], START_EUL_TRUNK_PYBULLET[2], -START_EUL_TRUNK_PYBULLET[0]])

@app.spawn(start=True)
async def main(session: VuerSession):
    session.upsert @ PointLight(intensity=VUER_LIGHT_INTENSITY, position=VUER_LIGHT_POS)
    session.upsert @ Hands(fps=HAND_FPS, stream=True, key="hands")
    await asyncio.sleep(0.1)
    session.upsert @ Urdf(
        src=URDF_WEB,
        jointValues=START_Q,
        position=VIEWER_POS_TRUNK,
        rotation=VIEWER_ROT_TRUNK,
        key="robot",
    )
    global q
    counter = 0
    while True:
        counter += 1
        error = await ik("left")
        await asyncio.sleep(1 / MAX_FPS)
        
        async with q_lock:
            session.upsert @ Urdf(
                src=URDF_WEB,
                jointValues=q,
                position=VIEWER_POS_TRUNK,
                rotation=VIEWER_ROT_TRUNK,
                key="robot",
            )
        
        if counter % 100 == 0:
            print(f"Iteration {counter}, Current error: {error}")

        # Visualize the current position and the target
        actual_pos, _ = p.getLinkState(pb_robot_id, pb_eel_id)[:2]
        p.addUserDebugLine(actual_pos, goal_pos_eel, [1, 0, 0], 2, 0)
if __name__ == "__main__":
    app.run()