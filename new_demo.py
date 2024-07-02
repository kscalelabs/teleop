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
from vuer.schemas import  Hands, PointLight, Urdf

# web urdf is used for vuer
URDF_WEB: str = (
    "https://raw.githubusercontent.com/kscalelabs/webstompy/pawel/new_stomp/urdf/stompy_new/upper_limb_assembly_5_dof_merged_simplified.urdf"
)
# local urdf is used for pybullet
URDF_LOCAL: str = f"urdf/robot2/upper_limb_assembly_5_dof_merged_simplified.urdf"

# starting positions for robot trunk relative to world frames
START_POS_TRUNK_VUER: NDArray =  np.array([0, 1., 0]) #np.array([0, 1, 0])
START_EUL_TRUNK_VUER: NDArray = np.array([-math.pi, -3.8, 0])
START_POS_TRUNK_PYBULLET: NDArray = np.array([0, 0, 1.])
START_EUL_TRUNK_PYBULLET: NDArray = np.array([-math.pi/2, 0, -1.])

# starting positions for robot end effectors are defined relative to robot trunk frame
# which is right in the middle of the chest
START_POS_EER_VUER: NDArray = np.array([-.2, .3, .2]) # np.array([-0.2, -0.2, -0.2])
START_POS_EEL_VUER: NDArray = np.array([-.5, 0.2 , 0]) #np.array([0.2, -0.2, -0.2])
START_POS_EER_VUER += START_POS_TRUNK_PYBULLET
START_POS_EEL_VUER += START_POS_TRUNK_PYBULLET

# conversion between PyBullet and Vuer axes
VUER_TO_PB_AXES: NDArray = np.array([2, 0, 1], dtype=np.uint8)
VUER_TO_PB_AXES_SIGN: NDArray = np.array([1, 1, 1], dtype=np.int8)

# starting joint positions (Q means "joint angles")
START_Q: Dict[str, float] = {
    # torso
    "joint_torso_1_rmd_x8_90_mock_1_dof_x8": 0,

    # left arm (7dof)
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": 2.42,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": 4.42,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": 1.85,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": 1.74,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": -2.09,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_rmd_x4_24_mock_1_dof_x4": 0, # not working

    # left hand (2dof)
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_1": 0.0,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2": 0.0,

    # right arm (7dof)
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": 2.42,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": 1.61,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": -1.31,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": 2.88,
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": -1.88,
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_rmd_x4_24_mock_1_dof_x4": 0, # not working

    # right hand (2dof)
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_1": 0.0,
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_2": 0.0,
}   

# link names are based on the URDF
# EER means "end effector right"
# EEL means "end effector left"
EEL_LINK: str = "fused_component_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slide_1"
EER_LINK: str = "fused_component_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slide_1"

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
# THATS THE LIST
IK_Q_LIST: List[str] = [
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4',
    # slider
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_1', 
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2'
]

# PyBullet inverse kinematics (IK) params
# damping determines which joints are used for ik
# TODO: more custom damping will allow for legs/torso to help reach ee target
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

# Create a list to store movable joint indices and names
movable_joint_indices = []
joint_names = []
joint_index_to_name = {}
joint_name_to_index = {}

# Iterate through all joints
for i in range(pb_num_joints):
    joint_info = p.getJointInfo(pb_robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    
    # Check if the joint is not fixed (i.e., it's movable)
    if joint_type != p.JOINT_FIXED and joint_type != p.JOINT_PRISMATIC:
        movable_joint_indices.append(i)
        joint_names.append(joint_name)
        joint_index_to_name[i] = joint_name
        joint_name_to_index[joint_name] = i

p.resetBasePositionAndOrientation(
    pb_robot_id,
    START_POS_TRUNK_PYBULLET,
    p.getQuaternionFromEuler(START_EUL_TRUNK_PYBULLET),
)
# Set the camera view
target_position = START_POS_TRUNK_PYBULLET  # Use the robot's starting position as the target
camera_distance = 2.0  # Distance from the target (adjust as needed)
camera_yaw = 50  # Camera yaw angle in degrees
camera_pitch = -35  # Camera pitch angle in degrees

p.resetDebugVisualizerCamera(
    cameraDistance=camera_distance,
    cameraYaw=camera_yaw,
    cameraPitch=camera_pitch,
    cameraTargetPosition=target_position
)

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
print("\t ... done")

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
goal_orn_eel: NDArray = p.getQuaternionFromEuler(START_EUL_TRUNK_VUER)
# pfb30
goal_orn_eel = p.getQuaternionFromEuler([0, 0, 0])
# Define the point coordinates
point_coords = [goal_pos_eel]
# Define the color (RGB, values from 0 to 1)
point_color = [[1, 0, 0]]  # Red color
# Define the point size
point_size = 20  # Adjust this value to make the point larger or smaller
# Add the point to the simulation
p.addUserDebugPoints(point_coords, point_color, pointSize=point_size)

async def ik(arm: str) -> None:
    # start_time = time.time()
    if arm == "right":
        global goal_pos_eer, goal_orn_eer
        ee_id = pb_eer_id
        ee_chain = EER_CHAIN_ARM
        pos = goal_pos_eer
        orn = goal_orn_eer
    else:
        global goal_pos_eel, goal_orn_eel
        ee_id = pb_eel_id
        ee_chain = EEL_CHAIN_ARM
        pos = goal_pos_eel
        orn = goal_orn_eel
    # print(f"ik {arm} {pos} {orn}")
    pb_q = p.calculateInverseKinematics(
        pb_robot_id,
        ee_id,
        pos,
        orn,
        pb_joint_lower_limit,
        pb_joint_upper_limit,
        pb_joint_ranges,
        pb_start_q,
    )

    async with q_lock:
        global q
        new_changes = []
        for i, val in enumerate(pb_q):
            joint_name = IK_Q_LIST[i]
            if joint_name in ee_chain:
                q[joint_name] = val
                new_changes.append((joint_name[-20:], val))
                p.resetJointState(pb_robot_id, pb_q_map[joint_name], val)

                # take into account dynamics
                # p.setJointMotorControl2(bodyIndex=pb_robot_id,
                #                         jointIndex=pb_q_map[joint_name],
                #                         controlMode=p.POSITION_CONTROL,
                #                         targetPosition=val,
                #                         targetVelocity=0,
                #                         force=200,
                #                         positionGain=0.03,
                #                         velocityGain=1)
        # p.stepSimulation()
        # # If you want to set the joint positions:
        # for i, joint_index in enumerate(movable_joint_indices):
        #     new_changes.append((joint_names[i], pb_q[i]))
        #     p.resetJointState(pb_robot_id, joint_index, pb_q[i])

        print(new_changes)
        
        print(goal_pos_eel)
    # print(f"ik {arm} took {time.time() - start_time} seconds")

app = Vuer()

@app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
    # right hand
    rindex_pos: NDArray = np.array(event.value["rightLandmarks"][INDEX_FINGER_TIP_ID])
    rthumb_pos: NDArray = np.array(event.value["rightLandmarks"][THUMB_FINGER_TIP_ID])
    rpinch_dist: NDArray = np.linalg.norm(rindex_pos - rthumb_pos)
    # index finger to thumb pinch turns on tracking
    if rpinch_dist < PINCH_DIST_CLOSED:
        # print("Pinch detected in right hand")
        # NOTE: right hand controls left arm and left gripper
        global goal_pos_eel
        goal_pos_eel = np.multiply(rthumb_pos[VUER_TO_PB_AXES], VUER_TO_PB_AXES_SIGN)
        # pfb30
        # goal_pos_eel = np.array([-.5, 0.2, 1.])
        print(f"goal_pos_eel {goal_pos_eel}")
        # # pinching with middle finger controls gripper
        # rmiddl_pos: NDArray = np.array(event.value["rightLandmarks"][MIDDLE_FINGER_TIP_ID])
        # rgrip_dist: float = np.linalg.norm(rthumb_pos - rmiddl_pos) / PINCH_DIST_OPENED
        # # print(f"right gripper at {rgrip_dist}")
        # _s: float = EE_S_MIN + rgrip_dist * ee_s_range
        # async with q_lock:
        #     q["joint_left_arm_2_hand_1_slider_1"] = _s
        #     q["joint_left_arm_2_hand_1_slider_2"] = _s
    # left hand
    lindex_pos: NDArray = np.array(event.value["leftLandmarks"][INDEX_FINGER_TIP_ID])
    lthumb_pos: NDArray = np.array(event.value["leftLandmarks"][THUMB_FINGER_TIP_ID])
    lpinch_dist: NDArray = np.linalg.norm(lindex_pos - lthumb_pos)
    # index finger to thumb pinch turns on tracking
    if lpinch_dist < PINCH_DIST_CLOSED:
        # print("Pinch detected in left hand")
        # NOTE: left hand controls right arm and right gripper
        global goal_pos_eer
        goal_pos_eer = np.multiply(lthumb_pos[VUER_TO_PB_AXES], VUER_TO_PB_AXES_SIGN)

        print(f"goal_pos_eer {goal_pos_eer}")
        # # pinching with middle finger controls gripper
        # lmiddl_pos: NDArray = np.array(event.value["leftLandmarks"][MIDDLE_FINGER_TIP_ID])
        # lgrip_dist: float = np.linalg.norm(lthumb_pos - lmiddl_pos) / PINCH_DIST_OPENED
        # # _s: float = EE_S_MIN + lgrip_dist * ee_s_range
        # # # print(f"left gripper at {lgrip_dist}")
        # # async with q_lock:
        # #     q["joint_right_arm_1_hand_1_slider_1"] = _s
        # #     q["joint_right_arm_1_hand_1_slider_2"] = _s


@app.spawn(start=True)
async def main(session: VuerSession):
    session.upsert @ PointLight(intensity=VUER_LIGHT_INTENSITY, position=VUER_LIGHT_POS)
    session.upsert @ Hands(fps=HAND_FPS, stream=True, key="hands")
    await asyncio.sleep(0.1)
    session.upsert @ Urdf(
        src=URDF_WEB,
        jointValues=START_Q,
        position=START_POS_TRUNK_VUER,
        rotation=START_EUL_TRUNK_VUER,
        key="robot",
    )
    global q, img
    # TODO: ved add canbus controller
    #motor_controller = CanBusController()
    # standard position of left hand: [0.12057996 0.31537491 0.94372624]
    while True:
        await asyncio.gather(
            ik("left"),  # ~1ms
            # ik("right"),  # ~1ms
            asyncio.sleep(1 / MAX_FPS),  # ~16ms @ 60fps
            # update the joints
            # ved
            # motor_controller.send_joint_positions(q)
        )
        
        async with q_lock:
            session.upsert @ Urdf(
                src=URDF_WEB,
                jointValues=q,
                position=START_POS_TRUNK_VUER,
                rotation=START_EUL_TRUNK_VUER,
                key="robot",
            )
