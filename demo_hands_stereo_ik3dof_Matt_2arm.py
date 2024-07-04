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
START_POS_EEL: NDArray = np.array([0.3, -0.1, .3]) + START_POS_TRUNK_PYBULLET
START_POS_EER: NDArray = np.array([-0.3, -0.1, .3]) + START_POS_TRUNK_PYBULLET

# Vuer-specific configurations
START_POS_TRUNK_VUER: NDArray = np.array([0, 1., 0])
START_EUL_TRUNK_VUER: NDArray = np.array([-math.pi, -3.8, 0])
VUER_TO_PB_AXES: NDArray = np.array([2, 0, 1], dtype=np.uint8)
VUER_TO_PB_AXES_SIGN: NDArray = np.array([1, 1, 1], dtype=np.int8)

# Update the START_Q dictionary
START_Q: Dict[str, float] = {
    #"joint_torso_1_rmd_x8_90_mock_1_dof_x8": 0,
    # Left arm
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": -2.61,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": 1.38,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": 0,
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": -2.83,
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": -1.32,
    # Right arm (mirrored)
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8": -2.61,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8": 1.38,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4": 0,
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4": -2.83,
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4": -1.32,
}

# End effector links
EEL_LINK: str = "left_end_effector_link"
EER_LINK: str = "right_end_effector_link"

# Kinematic chains
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

# Initialize global variables
q_lock = asyncio.Lock()
q = deepcopy(START_Q)
goal_pos_eel: NDArray = START_POS_EEL
goal_pos_eer: NDArray = START_POS_EER
goal_orn_eel: NDArray = p.getQuaternionFromEuler([0, 0, 0])
goal_orn_eer: NDArray = p.getQuaternionFromEuler([0, 0, 0])

# Add goal position markers
p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)
p.addUserDebugPoints([goal_pos_eer], [[0, 0, 1]], pointSize=20)

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



async def ik(arm: str, max_attempts=20, max_iterations=100) -> float:
    global goal_pos_eel, goal_pos_eer, q
    
    if arm == "right":
        ee_id = pb_eer_id
        ee_chain = EER_CHAIN_ARM
        target_pos = goal_pos_eer
        q_subset = {joint: q[joint] for joint in EER_CHAIN_ARM}
    else:
        ee_id = pb_eel_id
        ee_chain = EEL_CHAIN_ARM
        target_pos = goal_pos_eel
        q_subset = {joint: q[joint] for joint in EEL_CHAIN_ARM}

    joint_indices = [pb_q_map[joint] for joint in ee_chain]
    
    best_error = float('inf')
    best_solution = None

    # Store initial state
    initial_states = [p.getJointState(pb_robot_id, idx)[0] for idx in range(p.getNumJoints(pb_robot_id))]

    for attempt in range(max_attempts):
        # Reset to initial state before each attempt
        for idx, state in enumerate(initial_states):
            p.resetJointState(pb_robot_id, idx, state)

        solution = p.calculateInverseKinematics(
            pb_robot_id,
            ee_id,
            target_pos,
            maxNumIterations=max_iterations,
            residualThreshold=1e-5
        )

        # Apply the solution only to the current arm's joints
        for i, idx in enumerate(joint_indices):
            joint_value = solution[i]
            joint_value = max(pb_joint_lower_limit[idx], min(pb_joint_upper_limit[idx], joint_value))
            p.resetJointState(pb_robot_id, idx, joint_value)
        
        p.stepSimulation()

        actual_pos, _ = p.getLinkState(pb_robot_id, ee_id)[:2]
        error = np.linalg.norm(np.array(target_pos) - np.array(actual_pos))

        if error < best_error:
            best_error = error
            best_solution = [solution[i] for i in range(len(joint_indices))]

        if error < 0.01:  # 1cm tolerance
            break

    # Apply the best solution
    if best_solution:
        for i, joint in enumerate(ee_chain):
            q_subset[joint] = best_solution[i]
            p.resetJointState(pb_robot_id, pb_q_map[joint], best_solution[i])

    # Update global q only for the current arm
    q.update(q_subset)

    print(f"\n{arm.capitalize()} arm results:")
    print(f"  Best error: {best_error}")
    print(f"  Final position: {p.getLinkState(pb_robot_id, ee_id)[0]}")
    print(f"  Goal position: {target_pos}")
    
    print(f"  Joint values:")
    for joint, value in q_subset.items():
        print(f"    {joint}: {value}")

    # Visualize the error
    actual_pos, _ = p.getLinkState(pb_robot_id, ee_id)[:2]
    color = [0, 0, 1] if arm == "right" else [1, 0, 0]
    p.addUserDebugLine(actual_pos, target_pos, color, 2, 0)

    return best_error

def update_viewer_goal(session: VuerSession, goal_pos: NDArray, key: str):
    # Convert PyBullet coordinates to Vuer coordinates
    viewer_pos = np.array([
        goal_pos[1] - START_POS_TRUNK_VUER[0],
        (goal_pos[2] - START_POS_TRUNK_VUER[1]),
        -(goal_pos[0] - START_POS_TRUNK_VUER[2])
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
    
    # Update or create the goal marker in the Vuer scene
    session.upsert @ Urdf(
        urdf=sphere_urdf,
        position=viewer_pos,
        key=key
    )

app = Vuer()

@app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
    global goal_pos_eer, goal_pos_eel
    
    base_pos, _ = p.getBasePositionAndOrientation(pb_robot_id)
    
    # right hand
    rthumb_pos = np.array(event.value["rightLandmarks"][THUMB_FINGER_TIP_ID])
    rpinch_dist = np.linalg.norm(np.array(event.value["rightLandmarks"][INDEX_FINGER_TIP_ID]) - rthumb_pos)
    if rpinch_dist < PINCH_DIST_CLOSED:
        goal_pos_eer = np.array([rthumb_pos[2], -rthumb_pos[0], rthumb_pos[1]]) + base_pos
        p.addUserDebugPoints([goal_pos_eer], [[0, 0, 1]], pointSize=20)

    # left hand
    lthumb_pos = np.array(event.value["leftLandmarks"][THUMB_FINGER_TIP_ID])
    lpinch_dist = np.linalg.norm(np.array(event.value["leftLandmarks"][INDEX_FINGER_TIP_ID]) - lthumb_pos)
    if lpinch_dist < PINCH_DIST_CLOSED:
        goal_pos_eel = np.array([lthumb_pos[2], -lthumb_pos[0], lthumb_pos[1]]) + base_pos
        p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)

@app.spawn(start=True)
async def main(session: VuerSession):
    global q
    
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
        
        # Perform IK for both arms
        error_left = await ik("left")
        error_right = await ik("right")
        
        # Short delay to maintain desired frame rate
        await asyncio.sleep(1 / MAX_FPS)
        
        # Update joint values from PyBullet to Vuer
        async with q_lock:
            for idx in range(pb_num_joints):
                joint_name = pb_joint_names[idx]
                joint_state = p.getJointState(pb_robot_id, idx)
                q[joint_name] = joint_state[0]
            
            # Update the robot's joint values in the Vuer scene
            session.upsert @ Urdf(
                src=URDF_WEB,
                jointValues=q,
                position=START_POS_TRUNK_VUER,
                rotation=START_EUL_TRUNK_VUER,
                key="robot",
            )
        
        # Print status every 100 iterations
        if counter % 100 == 0:
            print(f"Iteration {counter}, Left arm error: {error_left}, Right arm error: {error_right}")

        # Visualize current positions and targets for both arms
        left_actual_pos, _ = p.getLinkState(pb_robot_id, pb_eel_id)[:2]
        right_actual_pos, _ = p.getLinkState(pb_robot_id, pb_eer_id)[:2]
        p.addUserDebugLine(left_actual_pos, goal_pos_eel, [1, 0, 0], 2, 0)
        p.addUserDebugLine(right_actual_pos, goal_pos_eer, [0, 0, 1], 2, 0)

        # Update goal markers in the Vuer scene
        update_viewer_goal(session, goal_pos_eel, "left_goal_marker")
        update_viewer_goal(session, goal_pos_eer, "right_goal_marker")

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

if __name__ == "__main__":
    app.run()