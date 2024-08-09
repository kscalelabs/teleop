"""Demo application for PyBullet and Vuer integration for data collection.


TODO:
1. Test stub

"""
import argparse
import asyncio
import logging
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pybullet as p
import pybullet_data
from numpy.typing import NDArray
from vuer import Vuer, VuerSession
from vuer.schemas import Hands, PointLight, Urdf



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DELTA = 10
# URDF_WEB = "https://raw.githubusercontent.com/kscalelabs/teleop/c65a3ea28ace532c66dc9fa369707a45997d19ec/urdf/stompy_mini/full_assembly_simplified.urdf"
URDF_LOCAL = "urdf/stompy_mini/upper_half_assembly_simplified.urdf"
UPDATE_RATE = 1

# Robot configuration
START_POS_TRUNK_PYBULLET: NDArray = np.array([0, 0, 1])
START_EUL_TRUNK_PYBULLET: NDArray = np.array([math.pi, 0, 0])
START_POS_TRUNK_VUER: NDArray = np.array([0, 1, 0])
# START_EUL_TRUNK_VUER: NDArray = np.array([-math.pi, -0.68, 0])
START_EUL_TRUNK_VUER: NDArray = np.array([0,0, 0])

# Starting positions for robot end effectors
START_POS_EEL: NDArray = np.array([-0.35, -0.25, 0.0]) + START_POS_TRUNK_PYBULLET
START_POS_EER: NDArray = np.array([-0.35, 0.25, 0.0]) + START_POS_TRUNK_PYBULLET

PB_TO_VUER_AXES: NDArray = np.array([2, 0, 1], dtype=np.uint8)
PB_TO_VUER_AXES_SIGN: NDArray = np.array([1, 1, 1], dtype=np.int8)


# Starting joint positions in PyBullet (corresponds to 0 on real robot)
START_Q: Dict[str, float] = OrderedDict(
    [
        # left arm
        ("left shoulder pitch", -1.02),
        ("left shoulder yaw", 1.38),
        ("left shoulder roll", -3.24),
        ("left elbow pitch", 1.2),
        # ("left wrist roll", 0),

        # right arm
        ("right shoulder pitch", 3.12),
        ("right shoulder yaw", -1.98),
        ("right shoulder roll", -1.38),
        ("right elbow pitch", 1.32),
        # ("right wrist roll", 0),
    ]
)

# End effector links
EEL_JOINT: str = "left_end_effector_joint"
EER_JOINT: str = "right_end_effector_joint"

# Kinematic chains for each arm
EEL_CHAIN_ARM = [
    "left shoulder pitch",
    "left shoulder yaw",
    "left shoulder roll",
    "left elbow pitch",
    # "left wrist roll",
]
EER_CHAIN_ARM = [
    "right shoulder pitch",
    "right shoulder yaw",
    "right shoulder roll",
    "right elbow pitch",
    # "right wrist roll",
]

EEL_CHAIN_HAND = []
EER_CHAIN_HAND = []

OFFSET = list(START_Q.values())
OFFSET_LEFT = [START_Q[joint] for joint in EEL_CHAIN_ARM + EEL_CHAIN_HAND]

# Hand tracking parameters
INDEX_FINGER_TIP_ID, THUMB_FINGER_TIP_ID, MIDDLE_FINGER_TIP_ID = 8, 4, 14
PINCH_DIST_CLOSED, PINCH_DIST_OPENED = 0.1, 0.1  # 10 cm
EE_S_MIN, EE_S_MAX = 0.0, 0.05

# Global variables
q_lock = asyncio.Lock()
q = deepcopy(START_Q)
goal_pos_eel, goal_pos_eer = START_POS_EEL, START_POS_EER


class TeleopRobot:
    def __init__(self, use_firmware: bool = False, shared_dict: dict = {}) -> None:
        self.app = Vuer()
        self.robot_id = None
        self.joint_info: dict = {}
        self.actual_pos_eel, self.actual_pos_eer = START_POS_EEL, START_POS_EER
        self.goal_pos_eel, self.goal_pos_eer = START_POS_EEL, START_POS_EER
        self.q = deepcopy(START_Q)
        self.q_lock = asyncio.Lock()

        if use_firmware:
            from firmware.robot.robot import Robot
            self.robot = Robot(config_path="config.yaml", setup="left_arm_teleop")
            self.robot.zero_out()
        else:
            self.robot = None

        self.shared_data = shared_dict
        self.update_positions()
        self.update_shared_data()

    def update_shared_data(self) -> None:
        self.shared_data["positions"] = self.get_positions()
        self.shared_data["velocities"] = self.get_velocities()

    def test(self) -> None:
        if not self.robot:
            print("Firmware not enabled")
            return
        self.robot.test_motors(low=0, high=45)

    def setup_pybullet(self, use_gui: bool, urdf_path: str) -> None:
        """Set up PyBullet simulation environment."""
        p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
        p.setGravity(0, 0, -9.81)

        self.joint_info = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode("utf-8")
            self.joint_info[name] = {
                "index": i,
                "lower_limit": info[8],
                "upper_limit": info[9],
                "child_link_name": info[12].decode("utf-8"),
            }
            if name in START_Q:
                p.resetJointState(self.robot_id, i, START_Q[name])

        p.resetBasePositionAndOrientation(
            self.robot_id,
            START_POS_TRUNK_PYBULLET,
            p.getQuaternionFromEuler(START_EUL_TRUNK_PYBULLET),
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=START_POS_TRUNK_PYBULLET
        )

        # Add goal position markers
        p.addUserDebugPoints([self.goal_pos_eel], [[1, 0, 0]], pointSize=20)
        p.addUserDebugPoints([self.goal_pos_eer], [[0, 0, 1]], pointSize=20)

    async def inverse_kinematics(self, arm: str, max_attempts: int = 20) -> float | np.floating[Any]:
        """Perform inverse kinematics calculation for the specified arm."""
        ee_id = self.joint_info[EEL_JOINT if arm == "left" else EER_JOINT]["index"]
        ee_chain = EEL_CHAIN_ARM + EEL_CHAIN_HAND if arm == "left" else EER_CHAIN_ARM + EER_CHAIN_HAND
        target_pos = self.goal_pos_eel if arm == "left" else self.goal_pos_eer
        joint_damping = [0.1 if str(i) not in ee_chain else 100 for i in range(len(self.joint_info))]

        lower_limits = [self.joint_info[joint]["lower_limit"] for joint in ee_chain]
        upper_limits = [self.joint_info[joint]["upper_limit"] for joint in ee_chain]
        joint_ranges = [upper - lower for upper, lower in zip(upper_limits, lower_limits)]

        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)
        inv_torso_pos, inv_torso_orn = p.invertTransform(torso_pos, torso_orn)
        target_pos_local = p.multiplyTransforms(inv_torso_pos, inv_torso_orn, target_pos, [0, 0, 0, 1])[0]

        movable_joints = [
            j for j in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, j)[2] != p.JOINT_FIXED
        ]

        current_positions = [p.getJointState(self.robot_id, j)[0] for j in movable_joints]
        solution = p.calculateInverseKinematics(
            self.robot_id,
            ee_id,
            target_pos_local,
            currentPositions=current_positions,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=OFFSET,
            jointDamping=joint_damping,
        )

        actual_pos, _ = p.getLinkState(self.robot_id, ee_id)[:2]
        error = np.linalg.norm(np.array(target_pos) - np.array(actual_pos))

        if arm == "left":
            self.actual_pos_eel = actual_pos
        else:
            self.actual_pos_eer = actual_pos

        async with self.q_lock:
            for i, val in enumerate(solution):
                joint_name = list(START_Q.keys())[i]
                if joint_name in ee_chain:
                    self.q[joint_name] = val
                    p.resetJointState(self.robot_id, self.joint_info[joint_name]["index"], val)

        return error

    def hand_move_handler(self, event: Any) -> None:
        """Handle hand movement events from Vuer."""
        # Right hand
        rthumb_pos = np.array(event.value["rightLandmarks"][THUMB_FINGER_TIP_ID])
        rpinch_dist = np.linalg.norm(np.array(event.value["rightLandmarks"][INDEX_FINGER_TIP_ID]) - rthumb_pos)
        if rpinch_dist < PINCH_DIST_CLOSED:
            self.goal_pos_eer = np.multiply(rthumb_pos[PB_TO_VUER_AXES], PB_TO_VUER_AXES_SIGN)

            # Gripper control
            rmiddl_pos = np.array(event.value["rightLandmarks"][MIDDLE_FINGER_TIP_ID])
            rgrip_dist = np.linalg.norm(rthumb_pos - rmiddl_pos) / PINCH_DIST_OPENED
            _s = EE_S_MIN + rgrip_dist * (EE_S_MAX - EE_S_MIN)

            for slider in EER_CHAIN_HAND:
                self.q[slider] = 0.05 - _s
                p.resetJointState(self.robot_id, self.joint_info[slider]["index"], 0.05 - _s)

        # Left hand
        lthumb_pos = np.array(event.value["leftLandmarks"][THUMB_FINGER_TIP_ID])
        lpinch_dist = np.linalg.norm(np.array(event.value["leftLandmarks"][INDEX_FINGER_TIP_ID]) - lthumb_pos)
        if lpinch_dist < PINCH_DIST_CLOSED:
            self.goal_pos_eel = np.multiply(lthumb_pos[PB_TO_VUER_AXES], PB_TO_VUER_AXES_SIGN)

            # Gripper control
            lmiddl_pos = np.array(event.value["leftLandmarks"][MIDDLE_FINGER_TIP_ID])
            lgrip_dist = np.linalg.norm(lthumb_pos - lmiddl_pos) / PINCH_DIST_OPENED
            _s = EE_S_MIN + lgrip_dist * (EE_S_MAX - EE_S_MIN)

            for slider in EEL_CHAIN_HAND:
                self.q[slider] = 0.05 - _s
                p.resetJointState(self.robot_id, self.joint_info[slider]["index"], 0.05 - _s)

    async def main_loop(self, session: VuerSession, max_fps: int) -> None:
        """Main application loop."""
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

        if self.robot:
            new_positions = {"left_arm": [self.q[pos] for pos in EEL_CHAIN_ARM + EEL_CHAIN_HAND]}

        counter = 0
        while True:
            await asyncio.gather(
                self.inverse_kinematics("left"),
                self.inverse_kinematics("right"),
                asyncio.sleep(1 / max_fps),
            )
            self.update_shared_data()

            # Skip updating positions every UPDATE_RATE frames (adjust if CAN buffer is being overflowed)
            if counter > UPDATE_RATE:
                self.update_positions()
                counter = 0
            counter += 1

            async with self.q_lock:
                session.upsert @ Urdf(
                    src=URDF_WEB,
                    jointValues=self.q,
                    position=START_POS_TRUNK_VUER,
                    rotation=START_EUL_TRUNK_VUER,
                    key="robot",
                )

            if self.robot:
                new_positions["left_arm"] = [self.q[pos] for pos in EEL_CHAIN_ARM + EEL_CHAIN_HAND]
                offset = {"left_arm": OFFSET_LEFT}
                self.robot.set_position(new_positions, offset=offset, radians=True)

    def update_positions(self) -> None:
        if self.robot:
            self.robot.update_motor_data()
            pos = self.robot.get_motor_positions()["left_arm"]
            self.positions = np.array(pos)

    def get_positions(self) -> dict[str, dict[str, NDArray]]:
        if self.robot:
            return {
                "expected": {
                    "left": np.array(
                        [math.degrees(self.q[pos] - START_Q[pos]) for pos in EEL_CHAIN_ARM + EEL_CHAIN_HAND]
                    ),
                },
                "actual": {
                    "left": self.positions,
                },
            }
        else:
            return {
                "expected": {
                    "left": np.array([math.degrees(self.q[pos]) for pos in EEL_CHAIN_ARM + EEL_CHAIN_HAND]),
                },
                "actual": {
                    "left": np.random.rand(6),
                },
            }

    def get_velocities(self) -> Dict[str, NDArray]:
        return {
            "left": np.zeros(6),
        }

    def run(
        self,
        use_gui: bool,
        max_fps: int,
        urdf_path: str = URDF_LOCAL,
    ) -> None:
        self.setup_pybullet(use_gui, urdf_path)

        @self.app.add_handler("HAND_MOVE")
        async def hand_move_wrapper(event: Any, _: Any) -> None:
            self.hand_move_handler(event)

        @self.app.spawn(start=True)
        async def app_main(session: VuerSession) -> None:
            await self.main_loop(session, max_fps)


def run_teleop_app(use_gui: bool, max_fps: int, use_firmware: bool, shared_data: Dict[str, NDArray]) -> None:
    teleop = TeleopRobot(use_firmware=use_firmware, shared_dict=shared_data)
    teleop.run(use_gui, max_fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyBullet and Vuer integration for robot control")
    parser.add_argument("--firmware", action="store_true", help="Enable firmware control")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI mode")
    parser.add_argument("--fps", type=int, default=60, help="Maximum frames per second")
    parser.add_argument("--urdf", type=str, default=URDF_LOCAL, help="Path to URDF file")
    args = parser.parse_args()

    demo = TeleopRobot(use_firmware=args.firmware)
    demo.run(args.gui, args.fps, args.urdf)


if __name__ == "__main__":
    main()
