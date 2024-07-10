"""
POC for integrating PyBullet with Vuer for real-time robot control.

This script demonstrates the integration of PyBullet physics simulation with Vuer
for real-time robot control and visualization. It includes inverse kinematics (IK)
calculations, hand tracking, and optional firmware control.

Usage:
    python script_name.py [--firmware] [--gui] [--fps FPS] [--urdf PATH]

Options:
    --firmware  Enable firmware control (default: False)
    --gui       Use PyBullet GUI mode (default: False)
    --fps FPS   Set the maximum frames per second (default: 60)
    --urdf PATH Path to the URDF file (default: local path)
"""

import argparse
import asyncio
import math
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from numpy.typing import NDArray
from vuer import Vuer, VuerSession
from vuer.schemas import Hands, PointLight, Urdf

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DELTA = 10
URDF_WEB = "https://raw.githubusercontent.com/kscalelabs/teleop/master/urdf/stompy/upper_limb_assembly_5_dof_merged_simplified.urdf"
URDF_LOCAL = "urdf/stompy/upper_limb_assembly_5_dof_merged_simplified.urdf"

# Robot configuration
START_POS_TRUNK_PYBULLET: NDArray = np.array([0, 0, 1])
START_EUL_TRUNK_PYBULLET: NDArray = np.array([-math.pi / 2, 0, 2.15])
START_POS_TRUNK_VUER: NDArray = np.array([0, 1, 0])
START_EUL_TRUNK_VUER: NDArray = np.array([-math.pi, -0.68, 0])

# Starting positions for robot end effectors
START_POS_EEL: NDArray = np.array([-0.35, -0.25, 0.0]) + START_POS_TRUNK_PYBULLET
START_POS_EER: NDArray = np.array([-0.35, 0.25, 0.0]) + START_POS_TRUNK_PYBULLET

PB_TO_VUER_AXES: NDArray = np.array([2, 0, 1], dtype=np.uint8)
PB_TO_VUER_AXES_SIGN: NDArray = np.array([1, 1, 1], dtype=np.int8)

# Starting joint positions
START_Q: Dict[str, float] = OrderedDict(
    [
        # trunk
        ("joint_torso_1_rmd_x8_90_mock_1_dof_x8", 0),
        # left arm
        ("joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8", 0.503),
        ("joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8", -1.33),
        ("joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4", 0),
        ("joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4", 5.03),
        ("joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4", 1.76),
        # left gripper
        ("joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_1", 0.0),
        ("joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2", 0.0),
        # right arm
        ("joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8", 0.68),
        ("joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8", 1.24),
        ("joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4", 0),
        ("joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4", 3.45),
        ("joint_full_arm_5_dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4", 0),
        # right gripper
        ("joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_1", 0.0),
        ("joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_2", 0.0),
    ]
)

OFFSET = list(START_Q.values())

# End effector links
EEL_JOINT: str = "left_end_effector_joint"
EER_JOINT: str = "right_end_effector_joint"

# Kinematic chains for each arm
EEL_CHAIN_ARM = [
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8",
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8",
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4",
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4",
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4",
]
EER_CHAIN_ARM = [
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8",
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8",
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4",
    "joint_full_arm_5_dof_2_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4",
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4",
]
EEL_CHAIN_HAND = [
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_1",
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_slider_2",
]
EER_CHAIN_HAND = [
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_1",
    "joint_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slider_2",
]

# Hand tracking parameters
INDEX_FINGER_TIP_ID, THUMB_FINGER_TIP_ID, MIDDLE_FINGER_TIP_ID = 8, 4, 14
PINCH_DIST_CLOSED, PINCH_DIST_OPENED = 0.1, 0.1  # 10 cm
EE_S_MIN, EE_S_MAX = 0.0, 0.05

# Global variables
q_lock = asyncio.Lock()
q = deepcopy(START_Q)
goal_pos_eel, goal_pos_eer = START_POS_EEL, START_POS_EER


def setup_pybullet(use_gui: bool, urdf_path: str) -> Tuple[int, Dict]:
    """
    Set up PyBullet simulation environment.

    Args:
        use_gui (bool): Whether to use GUI mode.
        urdf_path (str): Path to the URDF file.

    Returns:
        Tuple containing robot ID and joint information dictionary.
    """
    p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
    p.setGravity(0, 0, -9.81)

    joint_info = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        joint_info[name] = {
            "index": i,
            "lower_limit": info[8],
            "upper_limit": info[9],
            "child_link_name": info[12].decode("utf-8"),
        }
        if name in START_Q:
            p.resetJointState(robot_id, i, START_Q[name])

    p.resetBasePositionAndOrientation(
        robot_id,
        START_POS_TRUNK_PYBULLET,
        p.getQuaternionFromEuler(START_EUL_TRUNK_PYBULLET),
    )
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=START_POS_TRUNK_PYBULLET
    )

    # Add goal position markers
    p.addUserDebugPoints([goal_pos_eel], [[1, 0, 0]], pointSize=20)
    p.addUserDebugPoints([goal_pos_eer], [[0, 0, 1]], pointSize=20)

    return robot_id, joint_info


async def inverse_kinematics(robot_id: int, joint_info: Dict, movable_joints: Dict, arm: str, max_attempts: int = 20) -> float:
    """
    Perform inverse kinematics calculation for the specified arm.

    Args:
        robot_id (int): PyBullet robot ID.
        joint_info (Dict): Joint information dictionary.
        arm (str): Arm to calculate IK for ('left' or 'right').
        max_attempts (int): Maximum number of IK calculation attempts.

    Returns:
        float: Error between target and actual position.
    """
    global goal_pos_eel, goal_pos_eer, q

    ee_id = joint_info[EEL_JOINT if arm == "left" else EER_JOINT]["index"]
    # TODO: add right arm support
    ee_chain = EEL_CHAIN_ARM + EEL_CHAIN_HAND if arm == "left" else EER_CHAIN_ARM + EER_CHAIN_HAND
    target_pos = goal_pos_eel if arm == "left" else goal_pos_eer
    joint_damping = [0.1 if i not in ee_chain else 100 for i in range(len(joint_info))]

    lower_limits = [joint_info[joint]["lower_limit"] for joint in ee_chain]
    upper_limits = [joint_info[joint]["upper_limit"] for joint in ee_chain]
    joint_ranges = [upper - lower for upper, lower in zip(upper_limits, lower_limits)]

    torso_pos, torso_orn = p.getBasePositionAndOrientation(robot_id)
    inv_torso_pos, inv_torso_orn = p.invertTransform(torso_pos, torso_orn)
    target_pos_local = p.multiplyTransforms(inv_torso_pos, inv_torso_orn, target_pos, [0, 0, 0, 1])[0]

    current_positions = [p.getJointState(robot_id, j)[0] for j in movable_joints]

    solution = p.calculateInverseKinematics(
        robot_id,
        ee_id,
        target_pos_local,
        currentPositions=current_positions,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=OFFSET,
        jointDamping=joint_damping,
    )

    actual_pos, _ = p.getLinkState(robot_id, ee_id)[:2]
    error = np.linalg.norm(np.array(target_pos) - np.array(actual_pos))

    async with q_lock:
        for i, val in enumerate(solution):
            joint_name = list(START_Q.keys())[i]
            if joint_name in ee_chain:
                q[joint_name] = val
                p.resetJointState(robot_id, joint_info[joint_name]["index"], val)

    return error


def hand_move_handler(event, robot_id: int, joint_info: Dict):
    """
    Handle hand movement events from Vuer.

    Args:
        event: Vuer hand movement event.
        robot_id (int): PyBullet robot ID.
        joint_info (Dict): Joint information dictionary.
    """
    global goal_pos_eer, goal_pos_eel, q

    # Right hand
    rthumb_pos = np.array(event.value["rightLandmarks"][THUMB_FINGER_TIP_ID])
    rpinch_dist = np.linalg.norm(np.array(event.value["rightLandmarks"][INDEX_FINGER_TIP_ID]) - rthumb_pos)
    if rpinch_dist < PINCH_DIST_CLOSED:
        goal_pos_eer = np.multiply(rthumb_pos[PB_TO_VUER_AXES], PB_TO_VUER_AXES_SIGN)

        # Gripper control
        rmiddl_pos = np.array(event.value["rightLandmarks"][MIDDLE_FINGER_TIP_ID])
        rgrip_dist = np.linalg.norm(rthumb_pos - rmiddl_pos) / PINCH_DIST_OPENED
        _s = EE_S_MIN + rgrip_dist * (EE_S_MAX - EE_S_MIN)

        for slider in EER_CHAIN_HAND:
            q[slider] = 0.05 - _s
            p.resetJointState(robot_id, joint_info[slider]["index"], 0.05 - _s)

    # Left hand
    lthumb_pos = np.array(event.value["leftLandmarks"][THUMB_FINGER_TIP_ID])
    lpinch_dist = np.linalg.norm(np.array(event.value["leftLandmarks"][INDEX_FINGER_TIP_ID]) - lthumb_pos)
    if lpinch_dist < PINCH_DIST_CLOSED:
        goal_pos_eel = np.multiply(lthumb_pos[PB_TO_VUER_AXES], PB_TO_VUER_AXES_SIGN)

        # Gripper control
        lmiddl_pos = np.array(event.value["leftLandmarks"][MIDDLE_FINGER_TIP_ID])
        lgrip_dist = np.linalg.norm(lthumb_pos - lmiddl_pos) / PINCH_DIST_OPENED
        _s = EE_S_MIN + lgrip_dist * (EE_S_MAX - EE_S_MIN)

        for slider in EEL_CHAIN_HAND:
            q[slider] = 0.05 - _s
            p.resetJointState(robot_id, joint_info[slider]["index"], 0.05 - _s)


async def main_loop(session: VuerSession, robot_id: int, joint_info: Dict, max_fps: int, use_firmware: bool):
    """
    Main application loop.

    Args:
        session (VuerSession): Vuer session object.
        robot_id (int): PyBullet robot ID.
        joint_info (Dict): Joint information dictionary.
        max_fps (int): Maximum frames per second.
        use_firmware (bool): Whether to use firmware control.
    """
    global q
    movable_joints = [j for j in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED]

    session.upsert @ PointLight(intensity=10.0, position=[0, 2, 2])
    session.upsert @ Hands(fps=30, stream=True, key="hands")
    await asyncio.sleep(0.1)
    session.upsert @ Urdf(
        src=URDF_WEB,
        jointValues=START_Q,
        position=START_POS_TRUNK_VUER,
        rotation=START_EUL_TRUNK_VUER,
        key="robot",
    )

    if use_firmware:
        from firmware.scripts.robot_controller import Robot
        # TODO update it to the actual setup depending on the robot
        robot = Robot("left_arm")
        robot.zero_out()
        new_positions = {"left_arm": [q[pos] for pos in EEL_CHAIN_ARM + EEL_CHAIN_HAND]}

    while True:
        await asyncio.gather(
            inverse_kinematics(robot_id, joint_info, movable_joints, "left"),
            inverse_kinematics(robot_id, joint_info, movable_joints, "right"),
            asyncio.sleep(1 / max_fps),
        )

        async with q_lock:
            session.upsert @ Urdf(
                src=URDF_WEB,
                jointValues=q,
                position=START_POS_TRUNK_VUER,
                rotation=START_EUL_TRUNK_VUER,
                key="robot",
            )

        if use_firmware:
            # TODO update q
            new_positions["left_arm"] = [q[pos] for pos in EEL_CHAIN_ARM + EEL_CHAIN_HAND]
            robot.set_position(new_positions)


def main():
    """Main entry point of the application."""
    parser = argparse.ArgumentParser(description="PyBullet and Vuer integration for robot control")
    parser.add_argument("--firmware", action="store_true", help="Enable firmware control")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI mode")
    parser.add_argument("--fps", type=int, default=60, help="Maximum frames per second")
    parser.add_argument("--urdf", type=str, default=URDF_LOCAL, help="Path to URDF file")
    args = parser.parse_args()

    robot_id, joint_info = setup_pybullet(args.gui, args.urdf)

    app = Vuer()

    @app.add_handler("HAND_MOVE")
    async def hand_move_wrapper(event, _):
        hand_move_handler(event, robot_id, joint_info)

    @app.spawn(start=True)
    async def app_main(session: VuerSession):
        await main_loop(session, robot_id, joint_info, args.fps, args.firmware)

    app.run()


if __name__ == "__main__":
    main()