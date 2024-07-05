import asyncio
from copy import deepcopy
import math
from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
import pybullet as p
import pybullet_data
from vuer import Vuer, VuerSession
from vuer.schemas import Hands, PointLight, Urdf

# URDF paths
URDF_WEB: str = "https://raw.githubusercontent.com/kscalelabs/webstompy/pawel/new_stomp/urdf/stompy_new/upper_limb_assembly_5_dof_merged_simplified.urdf"
URDF_LOCAL: str = "urdf/stompy_new/multiarm.urdf"

# Robot configuration
START_POS_TRUNK_PYBULLET: NDArray = np.array([0, 0, 1.])
START_EUL_TRUNK_PYBULLET: NDArray = np.array([-math.pi/2, 0, -1.])
START_POS_TRUNK_VUER: NDArray = np.array([0, 1., 0])
START_EUL_TRUNK_VUER: NDArray = np.array([-math.pi, -3.8, 0])

# Starting positions for robot end effectors
START_POS_EEL: NDArray = np.array([0.3, -0.1, .3]) + START_POS_TRUNK_PYBULLET
START_POS_EER: NDArray = np.array([-0.3, -0.1, .3]) + START_POS_TRUNK_PYBULLET

# Starting joint positions
START_Q: Dict[str, float] = {
    "joint_torso_1_rmd_x8_90_mock_1_dof_x8": 0,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": 0,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": 4.2,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": 0,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": 2.83,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": 1.32,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": 0,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": -4.2,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": 0,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": -2.83,
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": -1.32,
}

# End effector links
EEL_LINK: str = "left_end_effector_link"
EER_LINK: str = "right_end_effector_link"

# Kinematic chains for each arm
EEL_CHAIN_ARM: List[str] = [
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4',
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4',
]
EER_CHAIN_ARM: List[str] = [
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4',
    'joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4',
    'joint_full_arm_5_dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4',
]

# PyBullet setup
print("Starting PyBullet in GUI mode.")
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
pb_robot_id = p.loadURDF(URDF_LOCAL, [0, 0, 0], useFixedBase=True)
p.setGravity(0, 0, -9.81)

# Initialize joint information
pb_num_joints: int = p.getNumJoints(pb_robot_id)
pb_joint_names: List[str] = [""] * pb_num_joints
pb_child_link_names: List[str] = [""] * pb_num_joints
pb_joint_upper_limit: List[float] = [0.0] * pb_num_joints
pb_joint_lower_limit: List[float] = [0.0] * pb_num_joints
pb_joint_ranges: List[float] = [0.0] * pb_num_joints
pb_start_q: List[float] = [0.0] * pb_num_joints
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
    if name in EEL_CHAIN_ARM or name in EER_CHAIN_ARM:
        pb_q_map[name] = i

pb_eel_id = pb_child_link_names.index(EEL_LINK)
pb_eer_id = pb_child_link_names.index(EER_LINK)

# Set initial robot state
for i in range(pb_num_joints):
    p.resetJointState(pb_robot_id, i, pb_start_q[i])

p.resetBasePositionAndOrientation(
    pb_robot_id,
    START_POS_TRUNK_PYBULLET,
    p.getQuaternionFromEuler(START_EUL_TRUNK_PYBULLET),
)

# Set camera view
p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=START_POS_TRUNK_PYBULLET
)

# Vuer rendering params
MAX_FPS: int = 60
VUER_LIGHT_POS: NDArray = np.array([0, 2, 2])
VUER_LIGHT_INTENSITY: float = 10.0

# Vuer hand tracking params
HAND_FPS: int = 30
INDEX_FINGER_TIP_ID: int = 8
THUMB_FINGER_TIP_ID: int = 4
PINCH_DIST_CLOSED: float = 0.05

# Global variables
q_lock = asyncio.Lock()
q = deepcopy(START_Q)
goal_pos_eel: NDArray = START_POS_EEL
goal_pos_eer: NDArray = START_POS_EER

# Add goal position markers
p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)
p.addUserDebugPoints([goal_pos_eer], [[0, 0, 1]], pointSize=20)

def visualize_target_and_actual(target_pos, actual_pos, color):
    p.addUserDebugPoints([target_pos], [color], pointSize=6)
    p.addUserDebugPoints([actual_pos], [[1, 1, 1]], pointSize=4)  # White for actual position
    p.addUserDebugLine(target_pos, actual_pos, color, lineWidth=2)

async def ik(arm: str, max_attempts=20, max_iterations=100) -> float:
    global goal_pos_eel, goal_pos_eer, q
    
    # Get the current torso position and orientation
    torso_pos, torso_orn = p.getBasePositionAndOrientation(pb_robot_id)
    
    if arm == "right":
        ee_id = pb_eer_id
        ee_chain = EER_CHAIN_ARM
        target_pos = goal_pos_eer
        color = [0, 0, 1]  # Blue for right arm
    else:
        ee_id = pb_eel_id
        ee_chain = EEL_CHAIN_ARM
        target_pos = goal_pos_eel
        color = [1, 0, 0]  # Red for left arm

    joint_indices = [pb_q_map[joint] for joint in ee_chain]
    
    best_error = float('inf')
    best_solution = None

    # Prepare joint limit arrays for IK calculation
    lower_limits = [pb_joint_lower_limit[idx] for idx in joint_indices]
    upper_limits = [pb_joint_upper_limit[idx] for idx in joint_indices]
    joint_ranges = [upper - lower for upper, lower in zip(upper_limits, lower_limits)]

    # Transform target position to torso's local frame
    inv_torso_pos, inv_torso_orn = p.invertTransform(torso_pos, torso_orn)
    target_pos_local = p.multiplyTransforms(inv_torso_pos, inv_torso_orn, target_pos, [0, 0, 0, 1])[0]

    # Get all movable joints
    num_joints = p.getNumJoints(pb_robot_id)
    all_joints = range(num_joints)
    movable_joints = [j for j in all_joints if p.getJointInfo(pb_robot_id, j)[2] != p.JOINT_FIXED]
    
    # Prepare current positions for all movable joints
    current_positions = [p.getJointState(pb_robot_id, j)[0] for j in movable_joints]

    for attempt in range(max_attempts):
        solution = p.calculateInverseKinematics(
            pb_robot_id,
            ee_id,
            target_pos_local,
            currentPositions=current_positions,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=current_positions,
            maxNumIterations=max_iterations,
            residualThreshold=1e-5
        )

        # Apply the solution only to the arm joints
        for i, idx in enumerate(joint_indices):
            p.resetJointState(pb_robot_id, idx, solution[movable_joints.index(idx)])
        p.stepSimulation()

        actual_pos, _ = p.getLinkState(pb_robot_id, ee_id)[:2]
        error = np.linalg.norm(np.array(target_pos) - np.array(actual_pos))

        if error < best_error:
            best_error = error
            best_solution = [solution[movable_joints.index(idx)] for idx in joint_indices]

        if error < 0.01:  # 1cm tolerance
            break

    if best_solution:
        for i, joint in enumerate(ee_chain):
            q[joint] = best_solution[i]
            p.resetJointState(pb_robot_id, pb_q_map[joint], best_solution[i])

    final_pos, _ = p.getLinkState(pb_robot_id, ee_id)[:2]
    
    # Visualize target and actual positions
    visualize_target_and_actual(target_pos, final_pos, color)

    p.stepSimulation()

    return best_error

def visualize_target_and_actual(target_pos, actual_pos, color):
    p.addUserDebugPoints([target_pos], [color], pointSize=6)
    p.addUserDebugPoints([actual_pos], [[1, 1, 1]], pointSize=4)  # White for actual position
    p.addUserDebugLine(target_pos, actual_pos, color, lineWidth=2)

def verify_arm_config(arm: str):
    if arm == "right":
        ee_id = pb_eer_id
        ee_chain = EER_CHAIN_ARM
        ee_link = EER_LINK
    else:
        ee_id = pb_eel_id
        ee_chain = EEL_CHAIN_ARM
        ee_link = EEL_LINK

    print(f"\nVerifying {arm} arm configuration:")
    for joint in ee_chain:
        idx = pb_q_map[joint]
        print(f"Joint: {joint}")
        print(f"  Index: {idx}")
        print(f"  Lower limit: {pb_joint_lower_limit[idx]}")
        print(f"  Upper limit: {pb_joint_upper_limit[idx]}")
        print(f"  Current value: {p.getJointState(pb_robot_id, idx)[0]}")
    
    print(f"\n{arm.capitalize()} arm end effector:")
    print(f"  Link name: {ee_link}")
    print(f"  Link index: {ee_id}")
    ee_state = p.getLinkState(pb_robot_id, ee_id)
    print(f"  Current position: {ee_state[0]}")
    print(f"  Current orientation: {ee_state[1]}")

app = Vuer()

@app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
    global goal_pos_eer, goal_pos_eel
    
    # right hand
    rthumb_pos = np.array(event.value["rightLandmarks"][THUMB_FINGER_TIP_ID])
    rpinch_dist = np.linalg.norm(np.array(event.value["rightLandmarks"][INDEX_FINGER_TIP_ID]) - rthumb_pos)
    if rpinch_dist < PINCH_DIST_CLOSED:
        goal_pos_eer = np.array([rthumb_pos[2], -rthumb_pos[0], rthumb_pos[1]]) + START_POS_TRUNK_PYBULLET
        print(f"New goal_pos_eer: {goal_pos_eer}")
        p.addUserDebugPoints([goal_pos_eer], [[0, 0, 1]], pointSize=20)

    # left hand
    lthumb_pos = np.array(event.value["leftLandmarks"][THUMB_FINGER_TIP_ID])
    lpinch_dist = np.linalg.norm(np.array(event.value["leftLandmarks"][INDEX_FINGER_TIP_ID]) - lthumb_pos)
    if lpinch_dist < PINCH_DIST_CLOSED:
        goal_pos_eel = np.array([lthumb_pos[2], -lthumb_pos[0], lthumb_pos[1]]) + START_POS_TRUNK_PYBULLET
        print(f"New goal_pos_eel: {goal_pos_eel}")
        p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)

def update_viewer_goal(session: VuerSession, goal_pos: NDArray, key: str):
    viewer_pos = np.array([
        goal_pos[1] - START_POS_TRUNK_VUER[0],
        (goal_pos[2] - START_POS_TRUNK_VUER[1]),
        -(goal_pos[0] - START_POS_TRUNK_VUER[2])
    ])
    
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
        key=key
    )

@app.spawn(start=True)
async def main(session: VuerSession):
    global q
    
    # Verify arm configurations before starting
    verify_arm_config("left")
    verify_arm_config("right")
    
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
    
    counter = 0
    while True:
        counter += 1
        error_left = await ik("left")
        error_right = await ik("right")
        await asyncio.sleep(1 / MAX_FPS)
        
        # Update joint values from PyBullet
        async with q_lock:
            for idx in range(pb_num_joints):
                joint_name = pb_joint_names[idx]
                joint_state = p.getJointState(pb_robot_id, idx)
                q[joint_name] = joint_state[0]
            
            session.upsert @ Urdf(
                src=URDF_WEB,
                jointValues=q,
                position=START_POS_TRUNK_VUER,
                rotation=START_EUL_TRUNK_VUER,
                key="robot",
            )
        
        if counter % 1 == 0:
            print(f"Iteration {counter}, Left arm error: {error_left}, Right arm error: {error_right}")


        # Visualize the current positions and the targets for both arms
        left_actual_pos, _ = p.getLinkState(pb_robot_id, pb_eel_id)[:2]
        right_actual_pos, _ = p.getLinkState(pb_robot_id, pb_eer_id)[:2]
        p.addUserDebugLine(left_actual_pos, goal_pos_eel, [1, 0, 0], 2, 0)
        p.addUserDebugLine(right_actual_pos, goal_pos_eer, [0, 0, 1], 2, 0)
        update_viewer_goal(session, goal_pos_eel, "left_goal_marker")
        update_viewer_goal(session, goal_pos_eer, "right_goal_marker")

if __name__ == "__main__":
    app.run()