import math
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pybullet as p
import pybullet_data
from numpy.typing import NDArray

# Path to the URDF file
URDF_LOCAL: str = "urdf/stompy_new/Edited2.urdf"

# Starting position and orientation for the robot's trunk
START_POS_TRUNK_PYBULLET: NDArray = np.array([0, 0, 1.])
START_EUL_TRUNK_PYBULLET: NDArray = np.array([-math.pi/2, 0, -1.])

# Starting position for the end effector
# START_POS_EEL: NDArray = np.array([-0.4, 0.2, .5]) + START_POS_TRUNK_PYBULLET
START_POS_EEL: NDArray = np.array([0.4, -0.4, 1.5]) + START_POS_TRUNK_PYBULLET

# Initial joint positions
START_Q: Dict[str, float] = {
    "joint_torso_1_rmd_x8_90_mock_1_dof_x8": 0,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": 2.61,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": -1.38,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": 0,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": 2.83,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": 1.32,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_1": 0.0,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2": 0.0,
}   

# Name of the end effector link
EEL_LINK: str = "end_effector_link"

# Kinematic chain for the arm
#removed the sliders, they break things and i dont think make sense to have in our ik, theyre seperate controls. 
#AKA control them somewhere else
EEL_CHAIN_ARM: List[str] = [
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4',
]

# PyBullet initialization
print("Starting PyBullet in GUI mode.")
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
pb_robot_id = p.loadURDF(URDF_LOCAL, [0, 0, 0], useFixedBase=True)
p.setGravity(0, 0, -9.81)

# Get the number of joints in the robot
pb_num_joints: int = p.getNumJoints(pb_robot_id)

# Initialize lists and dictionaries for joint information
pb_joint_names: List[str] = [""] * pb_num_joints
pb_child_link_names: List[str] = [""] * pb_num_joints
pb_joint_upper_limit: List[float] = [0.0] * pb_num_joints
pb_joint_lower_limit: List[float] = [0.0] * pb_num_joints
pb_joint_ranges: List[float] = [0.0] * pb_num_joints
pb_start_q: List[float] = [0.0] * pb_num_joints
pb_q_map: Dict[str, int] = {}

# Populate joint information
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
    if name in EEL_CHAIN_ARM:
        pb_q_map[name] = i

# Get the index of the end effector link
pb_eel_id = pb_child_link_names.index(EEL_LINK)

# Set initial joint positions
for i in range(pb_num_joints):
    p.resetJointState(pb_robot_id, i, pb_start_q[i])

# Set the initial position and orientation of the robot
p.resetBasePositionAndOrientation(
    pb_robot_id,
    START_POS_TRUNK_PYBULLET,
    p.getQuaternionFromEuler(START_EUL_TRUNK_PYBULLET),
)

# Set up the camera view
p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=START_POS_TRUNK_PYBULLET
)

# Initialize goal position and current joint angles
q = deepcopy(START_Q)
goal_pos_eel: NDArray = START_POS_EEL

# Add a red point at the goal position
p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)

def ik(max_attempts=20, max_iterations=100):
    """
    Perform inverse kinematics to reach the goal position.
    
    Args:
    max_attempts (int): Maximum number of attempts to find a solution.
    max_iterations (int): Maximum iterations for each IK attempt.
    
    Returns:
    float: The best error achieved (distance from goal).
    """
    global goal_pos_eel, q
    ee_id = pb_eel_id
    ee_chain = EEL_CHAIN_ARM
    target_pos = goal_pos_eel

    best_error = float('inf')
    best_solution = None

    for attempt in range(max_attempts):
        # Generate a random initial guess within joint limits
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

    # Apply the best solution found
    if best_solution:
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

# Main loop

counter = 0

while True:
    counter += 1
    time.sleep(0.1)  # Reduced from 1 second to 0.1 seconds for faster updates
    error = ik()
    
    # Visualize the current position and the target
    actual_pos, _ = p.getLinkState(pb_robot_id, pb_eel_id)[:2]
    p.addUserDebugLine(actual_pos, goal_pos_eel, [1, 0, 0], 2, 0)
    
    if counter % 100 == 0:
        print(f"Iteration {counter}, Current error: {error}")

