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
import random

# web urdf is used for vuer
URDF_WEB: str = (
    "https://raw.githubusercontent.com/kscalelabs/webstompy/pawel/new_stomp/urdf/stompy_new/upper_limb_assembly_5_dof_merged_simplified.urdf"
)
# local urdf is used for pybullet
URDF_LOCAL: str = f"urdf/stompy_new/multiarm.urdf"



# starting positions for robot trunk relative to world frames
START_POS_TRUNK_VUER: NDArray = np.array([0, 1., 0])
START_EUL_TRUNK_VUER: NDArray = np.array([-math.pi, -3.8, 0])
START_POS_TRUNK_PYBULLET: NDArray = np.array([0, 0, 1.])
START_EUL_TRUNK_PYBULLET: NDArray = np.array([-math.pi/2, 0, -1.])
VIEWER_TO_PYBULLET_SCALE = 1.0  # Adjust this value if needed

# starting positions for robot end effectors are defined relative to robot trunk frame
START_POS_EER_VUER: NDArray = np.array([-.2, .3, .2]) + START_POS_TRUNK_PYBULLET
START_POS_EEL_VUER: NDArray = np.array([0.0, 0.2, -.3]) + START_POS_TRUNK_PYBULLET

# conversion between PyBullet and Vuer axes
VUER_TO_PB_AXES: NDArray = np.array([2, 0, 1], dtype=np.uint8)
VUER_TO_PB_AXES_SIGN: NDArray = np.array([1, 1, 1], dtype=np.int8)

# starting joint positions (Q means "joint angles")
START_Q: Dict[str, float] = {
    # torso
    "joint_torso_1_rmd_x8_90_mock_1_dof_x8": 0,

    # left arm (5dof)
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": 0,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": 1,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": 0,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": 1,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": 1.32,

    # left hand (2dof)
    #"joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_1": 0.0,
    #"joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2": 0.0,

    # right arm 
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8': 0.0,
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8': 4.2,
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4': 0.0,
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4': 2.83,
    'joint_full_arm_5_dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4':1.32,

    # right hand
    #'joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_1': 0.0,
    #'joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_2': 0.0,

}   

# link names are based on the URDF
EEL_LINK: str = "left_end_effector_link"
EER_LINK: str = "right_end_effector_link"

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
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4',
]
EER_CHAIN_HAND: List[str] = [
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_1",
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_2",
]
EE_Chain_Torso: List[str] = [
   'joint_torso_1_rmd_x8_90_mock_1_dof_x8',
]

# Update IK_Q_LIST to include right arm joints and torso joint
IK_Q_LIST: List[str] = [
    'joint_torso_1_rmd_x8_90_mock_1_dof_x8',  
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4',
    
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
pb_start_q = [0.0] * pb_num_joints
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
    if name in EER_CHAIN_ARM or name in EEL_CHAIN_ARM: #or name in EE_Chain_Torso:
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
goal_orn_eer: NDArray = p.getQuaternionFromEuler([0, 0, 0])
goal_pos_eel: NDArray = START_POS_EEL_VUER
goal_orn_eel: NDArray = p.getQuaternionFromEuler([0, 0, 0])

# Add a red point at the goal position
p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)
p.addUserDebugPoints([goal_pos_eer], [[1, 0, 0]], pointSize=20)

def setup_collision_filter():
    # Disable collisions between adjacent links
    for i in range(p.getNumJoints(pb_robot_id)):
        p.setCollisionFilterPair(pb_robot_id, pb_robot_id, i, i-1, 0)
    
    # Optionally, you can disable collisions between specific pairs of links
    # For example, to disable collision between link 0 and link 2:
    # p.setCollisionFilterPair(pb_robot_id, pb_robot_id, 0, 2, 0)
    
    # If you want to disable self-collision completely (not recommended for most cases):
    # p.setCollisionFilterGroupMask(pb_robot_id, -1, 0, 0)

    # If you want to set up more specific collision groups:
    # p.setCollisionFilterGroupMask(pb_robot_id, link_index, group, mask


async def ik(max_attempts=50, max_iterations=100) -> tuple[float, float]:
    global goal_pos_eer, goal_orn_eer, goal_pos_eel, goal_orn_eel, q

    best_error_left = float('inf')
    best_error_right = float('inf')
    best_solution_left = None
    best_solution_right = None

    # Define torso joint (currently fixed, but prepared for future use)
    torso_joint = 'joint_torso_1_rmd_x8_90_mock_1_dof_x8'

    # Combine all joints for IK
    all_joints = [torso_joint] + EEL_CHAIN_ARM + EER_CHAIN_ARM
    joint_indices = [pb_q_map[joint] for joint in all_joints if joint in pb_q_map]

    # Prepare IK parameters
    lower_limits = [pb_joint_lower_limit[i] for i in joint_indices]
    upper_limits = [pb_joint_upper_limit[i] for i in joint_indices]
    joint_ranges = [pb_joint_ranges[i] for i in joint_indices]
    rest_poses = [pb_start_q[i] for i in joint_indices]

    for attempt in range(max_attempts):
        # Solve IK for left arm
        solution_left = p.calculateInverseKinematics(
            pb_robot_id,
            pb_eel_id,
            goal_pos_eel,
            targetOrientation=goal_orn_eel,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            jointDamping=[0.1]*len(joint_indices),
            solver=p.IK_DLS,
            maxNumIterations=max_iterations,
            residualThreshold=1e-5
        )

        # Solve IK for right arm
        solution_right = p.calculateInverseKinematics(
            pb_robot_id,
            pb_eer_id,
            goal_pos_eer,
            targetOrientation=goal_orn_eer,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            jointDamping=[0.1]*len(joint_indices),
            solver=p.IK_DLS,
            maxNumIterations=max_iterations,
            residualThreshold=1e-5
        )

        # Apply solutions
        for i, idx in enumerate(joint_indices):
            p.resetJointState(pb_robot_id, idx, solution_left[i])
        
        p.stepSimulation()

        # Check error for left arm
        actual_pos_left, actual_orn_left = p.getLinkState(pb_robot_id, pb_eel_id)[:2]
        error_left = np.linalg.norm(np.array(goal_pos_eel) - np.array(actual_pos_left))

        # Apply right arm solution
        for i, idx in enumerate(joint_indices):
            p.resetJointState(pb_robot_id, idx, solution_right[i])
        
        p.stepSimulation()

        # Check error for right arm
        actual_pos_right, actual_orn_right = p.getLinkState(pb_robot_id, pb_eer_id)[:2]
        error_right = np.linalg.norm(np.array(goal_pos_eer) - np.array(actual_pos_right))

        if error_left < best_error_left:
            best_error_left = error_left
            best_solution_left = solution_left

        if error_right < best_error_right:
            best_error_right = error_right
            best_solution_right = solution_right

        if error_left < 0.01 and error_right < 0.01:
            break

    # Apply the best solutions
    if best_solution_left is not None and best_solution_right is not None:
        async with q_lock:
            for i, idx in enumerate(joint_indices):
                joint_name = pb_joint_names[idx]
                # Use left arm solution for left arm joints, right arm solution for right arm joints
                if joint_name in EEL_CHAIN_ARM:
                    value = best_solution_left[i]
                elif joint_name in EER_CHAIN_ARM:
                    value = best_solution_right[i]
                else:
                    # For shared joints (like torso), average the solutions
                    value = (best_solution_left[i] + best_solution_right[i]) / 2
                q[joint_name] = value
                p.resetJointState(pb_robot_id, idx, value)

    print(f"Best solution found with errors: Left = {best_error_left}, Right = {best_error_right}")
    print(f"Final left position: {p.getLinkState(pb_robot_id, pb_eel_id)[0]}")
    print(f"Final right position: {p.getLinkState(pb_robot_id, pb_eer_id)[0]}")
    print(f"Goal left position: {goal_pos_eel}")
    print(f"Goal right position: {goal_pos_eer}")

    # Visualize the error
    p.addUserDebugLine(p.getLinkState(pb_robot_id, pb_eel_id)[0], goal_pos_eel, [1, 0, 0], 2, 0)
    p.addUserDebugLine(p.getLinkState(pb_robot_id, pb_eer_id)[0], goal_pos_eer, [0, 0, 1], 2, 0)

    p.stepSimulation()

    return best_error_left, best_error_right
app = Vuer()
def update_viewer_goal(session: VuerSession, goal_pos: NDArray):
    # Adjust the conversion to account for the new rotation
    viewer_pos = np.array([
        goal_pos[1] - VIEWER_POS_TRUNK[0],
        (goal_pos[2] - VIEWER_POS_TRUNK[1]),
        -(goal_pos[0] - VIEWER_POS_TRUNK[2])
    ])

    
    # Create a small sphere URDF to represent the goal
    sphere_urdf = f"""
    <?xml version="1.0"?>
    <robot name="sphere">
      <link name="sphere_link">
        <visual>
          <geometry>
            <sphere radius="0.05"/>
          </geometry>
          <material>
            <color rgba="1 0 0 1"/>
          </material>
        </visual>
      </link>
    </robot>
    """
    
    session.upsert @ Urdf(
        urdf=sphere_urdf,
        position=viewer_pos,
        key="goal_marker"
    )

# Convert PyBullet position to viewer coordinates
VIEWER_POS_TRUNK = np.array([
    START_POS_TRUNK_PYBULLET[1],
    START_POS_TRUNK_PYBULLET[2],
    -START_POS_TRUNK_PYBULLET[0]
])

# Convert PyBullet orientation to viewer coordinates
VIEWER_ROT_TRUNK = np.array([
    START_EUL_TRUNK_PYBULLET[1] + math.pi,
    -START_EUL_TRUNK_PYBULLET[2],
    START_EUL_TRUNK_PYBULLET[0] + math.pi/2
])

@app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
    global goal_pos_eer, goal_pos_eel
    
    # right hand
    rindex_pos: NDArray = np.array(event.value["rightLandmarks"][INDEX_FINGER_TIP_ID])
    rthumb_pos: NDArray = np.array(event.value["rightLandmarks"][THUMB_FINGER_TIP_ID])
    rpinch_dist: NDArray = np.linalg.norm(rindex_pos - rthumb_pos)
    if rpinch_dist < PINCH_DIST_CLOSED:
        pybullet_pos = np.array([
            -(rthumb_pos[2] - VIEWER_POS_TRUNK[2]),
            rthumb_pos[0] + VIEWER_POS_TRUNK[0],
            -(rthumb_pos[1] - VIEWER_POS_TRUNK[1])
        ]) * VIEWER_TO_PYBULLET_SCALE
        goal_pos_eer = pybullet_pos
        print(f"New goal_pos_eer: {goal_pos_eer}")
        
        # Update the visualized goal position for right arm
        p.addUserDebugPoints([goal_pos_eer], [[0, 0, 1]], pointSize=20)

    # left hand
    lindex_pos: NDArray = np.array(event.value["leftLandmarks"][INDEX_FINGER_TIP_ID])
    lthumb_pos: NDArray = np.array(event.value["leftLandmarks"][THUMB_FINGER_TIP_ID])
    lpinch_dist: NDArray = np.linalg.norm(lindex_pos - lthumb_pos)
    if lpinch_dist < PINCH_DIST_CLOSED:
        pybullet_pos = np.array([
            -(lthumb_pos[2] - VIEWER_POS_TRUNK[2]),
            lthumb_pos[0] + VIEWER_POS_TRUNK[0],
            -(lthumb_pos[1] - VIEWER_POS_TRUNK[1])
        ]) * VIEWER_TO_PYBULLET_SCALE
        goal_pos_eel = pybullet_pos
        print(f"New goal_pos_eel: {goal_pos_eel}")
        
        # Update the visualized goal position for left arm
        p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)



@app.spawn(start=True)
async def main(session: VuerSession):
    global q
    
    setup_collision_filter()
    
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
    
    counter = 0
    while True:
        counter += 1
        error_left, error_right = await ik()
        await asyncio.sleep(1 / MAX_FPS)
        
        # Update joint values from PyBullet
        async with q_lock:
            for idx in range(p.getNumJoints(pb_robot_id)):
                joint_name = pb_joint_names[idx]
                joint_state = p.getJointState(pb_robot_id, idx)
                q[joint_name] = joint_state[0]
            session.upsert @ Urdf(
                src=URDF_WEB,
                jointValues=q,
                position=VIEWER_POS_TRUNK,
                rotation=VIEWER_ROT_TRUNK,
                key="robot",
            )
        
        if counter % 100 == 0:
            print(f"Iteration {counter}, Left arm error: {error_left}, Right arm error: {error_right}")

        # Visualize the current positions and the targets for both arms
        update_viewer_goal(session, goal_pos_eel)
        update_viewer_goal(session, goal_pos_eer)

if __name__ == "__main__":
    app.run()