import math
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pybullet as p
import pybullet_data
from numpy.typing import NDArray


# local urdf is used for pybullet
URDF_LOCAL: str = f"urdf/stompy_new/edited.urdf"

# starting positions for robot trunk relative to world frames
START_POS_TRUNK_PYBULLET: NDArray = np.array([0, 0, 1.])
START_EUL_TRUNK_PYBULLET: NDArray = np.array([-math.pi/2, 0, -1.])

# starting positions for robot end effectors are defined relative to robot trunk frame
# which is right in the middle of the chest
START_POS_EEL: NDArray = np.array([-0.4, 0.2, .5])
START_POS_EEL += START_POS_TRUNK_PYBULLET

# starting joint positions (Q means "joint angles")
START_Q: Dict[str, float] = {
    # torso
    "joint_torso_1_rmd_x8_90_mock_1_dof_x8": 0,

    # left arm (7dof)
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
EEL_LINK: str = "fused_component_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_outer_rmd_x4_24_1"#"end_effector_link" #"fused_component_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slide_1"

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
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2',
    #"end_effector_link"
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
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2',
    #"end_effector_link"
]

# PyBullet inverse kinematics (IK) params
# damping determines which joints are used for ik
# TODO: more custom damping will allow for legs/torso to help reach ee target
DAMPING_CHAIN: float = 0.1
DAMPING_NON_CHAIN: float = 10.

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
rest_pose: List[float] = [0.0] * pb_num_joints
pb_damping: List[float] = [0.0] * pb_num_joints
pb_q_map: Dict[str, int] = {}
for i in range(pb_num_joints):
    info = p.getJointInfo(pb_robot_id, i)
    name = info[1].decode("utf-8")

    pb_joint_names[i] = name
    print(info[12].decode("utf-8"))
    pb_child_link_names[i] = info[12].decode("utf-8")
    pb_joint_lower_limit[i] = info[8]
    pb_joint_upper_limit[i] = info[9]
    pb_joint_ranges[i] = abs(info[9] - info[8])

    if name in START_Q:
        pb_start_q[i] = START_Q[name]
    if name in IK_Q_LIST or name in IK_Q_LIST:
        pb_damping[i] = DAMPING_CHAIN
    else:
        pb_damping[i] = DAMPING_NON_CHAIN
    if name in IK_Q_LIST:
        pb_q_map[name] = i

pb_eel_id = pb_child_link_names.index(EEL_LINK)

for i in range(pb_num_joints):
    p.resetJointState(pb_robot_id, i, pb_start_q[i])
print("\t ... done")


q = deepcopy(START_Q)
goal_pos_eel: NDArray = START_POS_EEL
goal_orn_eel: NDArray = p.getQuaternionFromEuler([0, 0, 0])

# Add a red point at the goal position
point_coords = [goal_pos_eel]
point_color = [[1, 0, 0]]  # Red color
point_size = 20
p.addUserDebugPoints(point_coords, point_color, pointSize=point_size)

def ik() -> None:
    global goal_pos_eel, goal_orn_eel, q
    ee_id = pb_eel_id
    ee_chain = EEL_CHAIN_ARM
    pos = goal_pos_eel
    orn = goal_orn_eel

    best_error = float('inf')
    best_solution = None

    for attempt in range(10):  # Try 10 times
        # Perturb the initial guess slightly
        initial_guess = [p.getJointState(pb_robot_id, i)[0] + np.random.uniform(-0.1, 0.1) for i in range(p.getNumJoints(pb_robot_id))]

        pb_q = p.calculateInverseKinematics(
            pb_robot_id,
            ee_id,
            pos,
            orn,
            lowerLimits=pb_joint_lower_limit,
            upperLimits=pb_joint_upper_limit,
            jointRanges=pb_joint_ranges,
            restPoses=initial_guess,
            maxNumIterations=100,
            residualThreshold=1e-5
        )

        # Apply the IK solution temporarily
        for i, joint_name in enumerate(IK_Q_LIST):
            if joint_name in ee_chain:
                p.resetJointState(pb_robot_id, pb_q_map[joint_name], pb_q[i])

        # Check the error
        actual_pos, _ = p.getLinkState(pb_robot_id, ee_id)[:2]
        error = np.linalg.norm(np.array(pos) - np.array(actual_pos))

        if error < best_error:
            best_error = error
            best_solution = pb_q

        print(f"Attempt {attempt + 1}: Error = {error}")

    # Apply the best solution found
    new_changes = []
    for i, joint_name in enumerate(IK_Q_LIST):
        if joint_name in ee_chain:
            q[joint_name] = best_solution[i]
            new_changes.append((joint_name[-20:], best_solution[i]))
            p.resetJointState(pb_robot_id, pb_q_map[joint_name], best_solution[i])

    print(f"Best solution found with error: {best_error}")
    print(new_changes)
    print(f"Final position: {p.getLinkState(pb_robot_id, ee_id)[0]}")
    print(f"Goal position: {goal_pos_eel}")

    # Visualize the error
    actual_pos, _ = p.getLinkState(pb_robot_id, ee_id)[:2]
    p.addUserDebugLine(actual_pos, goal_pos_eel, [1, 0, 0], 2, 0)

    p.stepSimulation()



counter = 0
while True:
    counter += 1
    time.sleep(1)
    p.stepSimulation()
    ik()
    actual_pos, _ = p.getLinkState(pb_robot_id, pb_eel_id)[:2]
    p.addUserDebugLine(actual_pos, goal_pos_eel, [1, 0, 0], 2, 0)
    # if counter == 1000:
    #     goal_pos_eel = np.array([0., 0.2 , 0.2])
        #p.addUserDebugPoints([goal_pos_eel], point_color, pointSize=point_size)
