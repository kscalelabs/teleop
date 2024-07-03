import math
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pybullet as p
import pybullet_data
from numpy.typing import NDArray

from dataclasses import dataclass

@dataclass
class IKTarget:
    link_name: str
    position: NDArray
    priority: float  # Higher number means higher priority

# Path to the URDF file
URDF_LOCAL: str = "urdf/stompy_new/Edited2.urdf"

# Starting position and orientation for the robot's trunk
START_POS_TRUNK_PYBULLET: NDArray = np.array([0, 0, 1.])
START_EUL_TRUNK_PYBULLET: NDArray = np.array([-math.pi/2, 0, -1.])

# Starting position for the end effector
START_POS_EEL: NDArray = np.array([-0.4, 0.2, .5]) + START_POS_TRUNK_PYBULLET

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

def ik(targets, max_attempts=40, max_iterations=100):
    global q
    ee_chain = EEL_CHAIN_ARM

    best_error = float('inf')
    best_solution = None
    best_positions = {}

    for attempt in range(max_attempts):
        # Generate a random initial guess within joint limits
        initial_guess = [np.random.uniform(pb_joint_lower_limit[pb_q_map[joint]], 
                                           pb_joint_upper_limit[pb_q_map[joint]]) 
                         for joint in ee_chain]

        # Start with the highest priority target (end effector)
        primary_target = targets[0]
        solution = p.calculateInverseKinematics(
            pb_robot_id,
            pb_child_link_names.index(primary_target.link_name),
            primary_target.position,
            lowerLimits=[pb_joint_lower_limit[pb_q_map[joint]] for joint in ee_chain],
            upperLimits=[pb_joint_upper_limit[pb_q_map[joint]] for joint in ee_chain],
            jointRanges=[pb_joint_ranges[pb_q_map[joint]] for joint in ee_chain],
            restPoses=initial_guess,
            maxNumIterations=max_iterations,
            residualThreshold=1e-5
        )

        # Check the error for the primary target without applying the solution
        temp_positions = {}
        for i, joint in enumerate(ee_chain):
            p.resetJointState(pb_robot_id, pb_q_map[joint], solution[i])
        p.stepSimulation()
        
        actual_pos, _ = p.getLinkState(pb_robot_id, pb_child_link_names.index(primary_target.link_name))[:2]
        error = np.linalg.norm(np.array(primary_target.position) - np.array(actual_pos))

        if error < best_error:
            best_error = error
            best_solution = solution
            best_positions = {target.link_name: p.getLinkState(pb_robot_id, pb_child_link_names.index(target.link_name))[0]
                              for target in targets}

        # Reset to original state after checking
        for i, joint in enumerate(ee_chain):
            p.resetJointState(pb_robot_id, pb_q_map[joint], q[joint])
        p.stepSimulation()

    # Apply the best solution found
    if best_solution:
        for i, joint in enumerate(ee_chain):
            q[joint] = best_solution[i]
            p.resetJointState(pb_robot_id, pb_q_map[joint], best_solution[i])

    p.stepSimulation()

    return best_error, best_positions

# Define targets
end_effector_target = IKTarget(EEL_LINK, START_POS_EEL, 1.0)
elbow_target = IKTarget("fused_component_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_inner_rmd_x4_24_1", 
                        np.array([-0.2, 0.3, 1.2]), 0.5)

targets = [end_effector_target, elbow_target]

# Main loop
# Main loop
counter = 0
visualization_items = []

while True:
    counter += 1
    time.sleep(.0005)
    
    error, best_positions = ik(targets)
    
    # Clear previous visualizations
    for item in visualization_items:
        p.removeUserDebugItem(item)
    visualization_items.clear()
    
    # Visualize only the best solution
    for target in targets:
        line_id = p.addUserDebugLine(
            best_positions[target.link_name], 
            target.position, 
            [1, 0, 0], 
            2, 
            0
        )
        visualization_items.append(line_id)
    
    if counter % 100 == 0:
        print(f"Iteration {counter}, Current error: {error}")

    # Here you could add code to update targets based on user input for teleop