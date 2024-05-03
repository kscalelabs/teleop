"""
This is a local copy of kscalelabs/sim/sim/stompy/joints.py

Defines a more Pythonic interface for specifying the joint names.

The best way to re-generate this snippet for a new robot is to use the
`sim/scripts/print_joints.py` script. This script will print out a hierarchical
tree of the various joint names in the robot.
"""

import textwrap
from abc import ABC
from typing import Dict, List, Union
from collections import OrderedDict

import numpy as np


class Node(ABC):
    @classmethod
    def children(cls) -> List["Union[Node, str]"]:
        return [
            attr
            for attr in (
                getattr(cls, attr) for attr in dir(cls) if not attr.startswith("__")
            )
            if isinstance(attr, (Node, str))
        ]

    @classmethod
    def joints(cls) -> List[str]:
        return [
            attr
            for attr in (
                getattr(cls, attr) for attr in dir(cls) if not attr.startswith("__")
            )
            if isinstance(attr, str)
        ]

    @classmethod
    def joints_motors(cls) -> List[str]:
        joint_names = []
        for attr in dir(cls):
            if not attr.startswith("__"):
                attr2 = getattr(cls, attr)
                if isinstance(attr2, str):
                    joint_names.append((attr, attr2))

        return joint_names

    @classmethod
    def all_joints(cls) -> List[str]:
        leaves = cls.joints()
        for child in cls.children():
            if isinstance(child, Node):
                leaves.extend(child.all_joints())
        return leaves

    def __str__(self) -> str:
        parts = [str(child) for child in self.children()]
        parts_str = textwrap.indent("\n" + "\n".join(parts), "  ")
        return f"[{self.__class__.__name__}]{parts_str}"


class Head(Node):
    left_right = "joint_head_1_x4_1_dof_x4"
    up_down = "joint_head_1_x4_2_dof_x4"


class Torso(Node):
    pitch = "joint_torso_1_x8_1_dof_x8"


class LeftHand(Node):
    hand_roll = "joint_left_arm_2_hand_1_x4_1_dof_x4"
    hand_grip = "joint_left_arm_2_hand_1_x4_2_dof_x4"
    slider_a = "joint_left_arm_2_hand_1_slider_1"
    slider_b = "joint_left_arm_2_hand_1_slider_2"


class LeftArm(Node):
    shoulder_yaw = "joint_left_arm_2_x8_1_dof_x8"
    shoulder_pitch = "joint_left_arm_2_x8_2_dof_x8"
    shoulder_roll = "joint_left_arm_2_x6_1_dof_x6"
    elbow_yaw = "joint_left_arm_2_x6_2_dof_x6"
    elbow_roll = "joint_left_arm_2_x4_1_dof_x4"
    hand = LeftHand()


class RightHand(Node):
    hand_roll = "joint_right_arm_1_hand_1_x4_1_dof_x4"
    hand_grip = "joint_right_arm_1_hand_1_x4_2_dof_x4"
    slider_a = "joint_right_arm_1_hand_1_slider_1"
    slider_b = "joint_right_arm_1_hand_1_slider_2"


class RightArm(Node):
    shoulder_yaw = "joint_right_arm_1_x8_1_dof_x8"
    shoulder_pitch = "joint_right_arm_1_x8_2_dof_x8"
    shoulder_roll = "joint_right_arm_1_x6_1_dof_x6"
    elbow_yaw = "joint_right_arm_1_x6_2_dof_x6"
    elbow_roll = "joint_right_arm_1_x4_1_dof_x4"
    hand = RightHand()


class LeftLeg(Node):
    hip_roll = "joint_legs_1_x8_2_dof_x8"
    hip_yaw = "joint_legs_1_left_leg_1_x8_1_dof_x8"
    hip_pitch = "joint_legs_1_left_leg_1_x10_1_dof_x10"
    knee_motor = "joint_legs_1_left_leg_1_x10_2_dof_x10"
    knee = "joint_legs_1_left_leg_1_knee_revolute"
    ankle_motor = "joint_legs_1_left_leg_1_x6_1_dof_x6"
    ankle = "joint_legs_1_left_leg_1_ankle_revolute"
    foot_roll = "joint_legs_1_left_leg_1_x4_1_dof_x4"


class RightLeg(Node):
    hip_roll = "joint_legs_1_x8_1_dof_x8"
    hip_yaw = "joint_legs_1_right_leg_1_x8_1_dof_x8"
    hip_pitch = "joint_legs_1_right_leg_1_x10_2_dof_x10"
    knee_motor = "joint_legs_1_right_leg_1_x10_1_dof_x10"
    knee = "joint_legs_1_right_leg_1_knee_revolute"
    ankle_motor = "joint_legs_1_right_leg_1_x6_1_dof_x6"
    ankle = "joint_legs_1_right_leg_1_ankle_revolute"
    foot_roll = "joint_legs_1_right_leg_1_x4_1_dof_x4"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Stompy(Node):
    head = Head()
    torso = Torso()
    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_positions(cls) -> Dict[str, float]:
        return {
            Stompy.head.left_right: np.deg2rad(-54),
            Stompy.head.up_down: 0.0,
            Stompy.torso.pitch: 0.0,
            Stompy.left_arm.shoulder_yaw: np.deg2rad(60),
            Stompy.left_arm.shoulder_pitch: np.deg2rad(60),
            Stompy.right_arm.shoulder_yaw: np.deg2rad(-60),
        }

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            Stompy.head.left_right: np.deg2rad(-2),  # -0.03
            # arms
            Stompy.left_arm.shoulder_yaw: np.deg2rad(-69.5),  # -1.21
            Stompy.left_arm.shoulder_pitch: np.deg2rad(-93),  # 1.61
            Stompy.right_arm.shoulder_yaw: np.deg2rad(85),  # 1.48
            Stompy.right_arm.shoulder_pitch: np.deg2rad(104),  # 1.81
            # legs
            Stompy.legs.left.hip_roll: np.deg2rad(29),  # 0.5
            Stompy.legs.left.hip_yaw: np.deg2rad(-29),  # -0.5
            Stompy.legs.left.hip_pitch: np.deg2rad(56),  # 0.97
            Stompy.legs.right.hip_roll: np.deg2rad(-29),  # -0.5
            Stompy.legs.right.hip_yaw: np.deg2rad(-29),  # -0.5
            Stompy.legs.right.hip_pitch: np.deg2rad(-56),  # -0.97
            Stompy.legs.left.knee: np.deg2rad(-6),  # -0.1
            Stompy.legs.right.knee: np.deg2rad(6),  # 0.1
            Stompy.legs.left.ankle: np.deg2rad(0),  # 0
            Stompy.legs.right.ankle: np.deg2rad(0),  # 0
            Stompy.legs.left.foot_roll: np.deg2rad(0),  # 0
            Stompy.legs.right.foot_roll: np.deg2rad(0),  # 0
        }

    @classmethod
    def default_sitting(cls) -> Dict[str, float]:
        return {
            Stompy.head.left_right: np.deg2rad(-3),
            Stompy.head.up_down: 0.0,
            Stompy.torso.pitch: 0.0,
            # arms
            Stompy.left_arm.shoulder_yaw: np.deg2rad(-88),
            Stompy.left_arm.shoulder_pitch: np.deg2rad(-30),
            Stompy.left_arm.shoulder_roll: np.deg2rad(-190),
            Stompy.left_arm.elbow_yaw: np.deg2rad(-88),
            Stompy.right_arm.shoulder_yaw: np.deg2rad(88),
            Stompy.right_arm.shoulder_pitch: np.deg2rad(30),
            Stompy.right_arm.shoulder_roll: np.deg2rad(190),
            Stompy.right_arm.elbow_yaw: np.deg2rad(88),
            # hands
            Stompy.left_arm.hand.hand_roll: np.deg2rad(-60),
            Stompy.right_arm.hand.hand_roll: np.deg2rad(-60),
            # legs
            Stompy.legs.left.hip_roll: np.deg2rad(29),
            Stompy.legs.left.hip_yaw: np.deg2rad(-29),
            Stompy.legs.left.hip_pitch: np.deg2rad(56),
            Stompy.legs.right.hip_roll: np.deg2rad(-29),
            Stompy.legs.right.hip_yaw: np.deg2rad(-29),
            # check this
            Stompy.legs.right.hip_pitch: np.deg2rad(-56),
            Stompy.legs.left.knee: np.deg2rad(-6),
            Stompy.legs.right.knee: np.deg2rad(6),
        }


def joint_dict_to_list(dicts: List[Dict[str, float]]) -> List[float]:
    # This is the order of joints in the URDF file
    joint_list_urdf = [
        "joint_right_arm_1_x8_1_dof_x8",
        "joint_left_arm_2_x8_1_dof_x8",
        "joint_head_1_x4_1_dof_x4",
        "joint_torso_1_x8_1_dof_x8",
        "joint_right_arm_1_x8_2_dof_x8",
        "joint_left_arm_2_x8_2_dof_x8",
        "joint_legs_1_x8_1_dof_x8",
        "joint_legs_1_x8_2_dof_x8",
        "joint_head_1_x4_2_dof_x4",
        "joint_right_arm_1_x6_1_dof_x6",
        "joint_left_arm_2_x6_1_dof_x6",
        "joint_legs_1_right_leg_1_x8_1_dof_x8",
        "joint_legs_1_left_leg_1_x8_1_dof_x8",
        "joint_right_arm_1_x6_2_dof_x6",
        "joint_left_arm_2_x6_2_dof_x6",
        "joint_legs_1_right_leg_1_x10_2_dof_x10",
        "joint_legs_1_left_leg_1_x10_1_dof_x10",
        "joint_legs_1_left_leg_1_knee_revolute",
        "joint_legs_1_right_leg_1_knee_revolute",
        "joint_right_arm_1_x4_1_dof_x4",
        "joint_left_arm_2_x4_1_dof_x4",
        "joint_legs_1_right_leg_1_x10_1_dof_x10",
        "joint_legs_1_right_leg_1_ankle_revolute",
        "joint_legs_1_left_leg_1_x10_2_dof_x10",
        "joint_legs_1_left_leg_1_ankle_revolute",
        "joint_legs_1_left_leg_1_x6_1_dof_x6",
        "joint_legs_1_right_leg_1_x6_1_dof_x6",
        "joint_legs_1_left_leg_1_x4_1_dof_x4",
        "joint_legs_1_right_leg_1_x4_1_dof_x4",
        "joint_right_arm_1_hand_1_x4_1_dof_x4",
        "joint_left_arm_2_hand_1_x4_1_dof_x4",
        "joint_right_arm_1_hand_1_slider_1",
        "joint_right_arm_1_hand_1_slider_2",
        "joint_left_arm_2_hand_1_slider_1",
        "joint_left_arm_2_hand_1_slider_2",
        "joint_right_arm_1_hand_1_x4_2_dof_x4",
        "joint_left_arm_2_hand_1_x4_2_dof_x4",
    ]
    link_list_rtb = [
        "link_head_1_x4_1_outer_1",
        "link_head_1_x4_2_outer_1",
        "link_right_arm_1_x8_1_outer_1",
        "link_right_arm_1_x8_2_outer_1",
        "link_right_arm_1_x6_1_outer_1",
        "link_right_arm_1_x6_2_outer_1",
        "link_right_arm_1_x4_1_outer_1",
        "link_right_arm_1_hand_1_x4_1_outer_1",
        "link_right_arm_1_hand_1_base_1",
        "link_right_arm_1_hand_1_base_1",
        "link_right_arm_1_hand_1_x4_2_outer_1",
        "link_left_arm_2_x8_1_outer_1",
        "link_left_arm_2_x8_2_outer_1",
        "link_left_arm_2_x6_1_outer_1",
        "link_left_arm_2_x6_2_outer_1",
        "link_left_arm_2_x4_1_outer_1",
        "link_left_arm_2_hand_1_x4_1_outer_1",
        "link_left_arm_2_hand_1_base_1",
        "link_left_arm_2_hand_1_base_1",
        "link_left_arm_2_hand_1_x4_2_outer_1",
        "link_torso_1_x8_1_outer_1",
        "link_legs_1_x8_1_outer_1",
        "link_legs_1_right_leg_1_x8_1_outer_1",
        "link_legs_1_right_leg_1_x10_2_outer_1",
        "link_legs_1_right_leg_1_x10_1_outer_1",
        "link_legs_1_right_leg_1_belt_thigh_right_1",
        "link_legs_1_right_leg_1_x6_1_outer_1",
        "link_legs_1_right_leg_1_51t_hdt5_15mm_belt_knee_right_1",
        "link_legs_1_right_leg_1_x4_1_inner_1",
        "link_legs_1_x8_2_outer_1",
        "link_legs_1_left_leg_1_x8_1_outer_1",
        "link_legs_1_left_leg_1_x10_1_outer_1",
        "link_legs_1_left_leg_1_x10_2_outer_1",
        "link_legs_1_left_leg_1_belt_thigh_left_1",
        "link_legs_1_left_leg_1_51t_hdt5_15mm_belt_knee_left_1",
        "link_legs_1_left_leg_1_x4_1_inner_1",
        "link_legs_1_left_leg_1_x6_1_outer_1",
    ]
    joint_list_rtb = [
        "joint_head_1_x4_1_dof_x4",
        "joint_head_1_x4_2_dof_x4",
        "joint_right_arm_1_x8_1_dof_x8",
        "joint_right_arm_1_x8_2_dof_x8",
        "joint_right_arm_1_x6_1_dof_x6",
        "joint_right_arm_1_x6_2_dof_x6",
        "joint_right_arm_1_x4_1_dof_x4",
        "joint_right_arm_1_hand_1_x4_1_dof_x4",
        "joint_right_arm_1_hand_1_slider_1",
        "joint_right_arm_1_hand_1_slider_2",
        "joint_right_arm_1_hand_1_x4_2_dof_x4",
        "joint_left_arm_2_x8_1_dof_x8",
        "joint_left_arm_2_x8_2_dof_x8",
        "joint_left_arm_2_x6_1_dof_x6",
        "joint_left_arm_2_x6_2_dof_x6",
        "joint_left_arm_2_x4_1_dof_x4",
        "joint_left_arm_2_hand_1_x4_1_dof_x4",
        "joint_left_arm_2_hand_1_slider_1",
        "joint_left_arm_2_hand_1_slider_2",
        "joint_left_arm_2_hand_1_x4_2_dof_x4",
        "joint_torso_1_x8_1_dof_x8",
        "joint_legs_1_x8_1_dof_x8",
        "joint_legs_1_right_leg_1_x8_1_dof_x8",
        "joint_legs_1_right_leg_1_x10_2_dof_x10",
        "joint_legs_1_right_leg_1_knee_revolute",
        "joint_legs_1_right_leg_1_x10_1_dof_x10",
        "joint_legs_1_right_leg_1_ankle_revolute",
        "joint_legs_1_right_leg_1_x6_1_dof_x6",
        "joint_legs_1_right_leg_1_x4_1_dof_x4",
        "joint_legs_1_x8_2_dof_x8",
        "joint_legs_1_left_leg_1_x8_1_dof_x8",
        "joint_legs_1_left_leg_1_x10_1_dof_x10",
        "joint_legs_1_left_leg_1_knee_revolute",
        "joint_legs_1_left_leg_1_x10_2_dof_x10",
        "joint_legs_1_left_leg_1_ankle_revolute",
        "joint_legs_1_left_leg_1_x6_1_dof_x6",
        "joint_legs_1_left_leg_1_x4_1_dof_x4",
    ]
    joint_map = OrderedDict((joint, i) for i, joint in enumerate(joint_list_rtb))
    q = [0] * len(joint_list_rtb)
    for d in dicts:
        for joint, value in d.items():
            q[joint_map[joint]] = value
    return q


class StompyFixed(Stompy):
    head = Head()
    torso = Torso()
    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            Stompy.head.left_right: np.deg2rad(-2),  # -0.03
            # arms
            Stompy.left_arm.shoulder_yaw: np.deg2rad(-69.5),  # -1.21
            Stompy.left_arm.shoulder_pitch: np.deg2rad(-93),  # 1.61
            Stompy.right_arm.shoulder_yaw: np.deg2rad(85),  # 1.48
            Stompy.right_arm.shoulder_pitch: np.deg2rad(104),  # 1.81
            # legs
            Stompy.legs.left.hip_roll: np.deg2rad(29),  # 0.5
            Stompy.legs.left.hip_yaw: np.deg2rad(-29),  # -0.5
            Stompy.legs.left.hip_pitch: np.deg2rad(56),  # 0.97
            Stompy.legs.right.hip_roll: np.deg2rad(-29),  # -0.5
            Stompy.legs.right.hip_yaw: np.deg2rad(-29),  # -0.5
            Stompy.legs.right.hip_pitch: np.deg2rad(-56),  # -0.97
            Stompy.legs.left.knee: np.deg2rad(-6),  # -0.1
            Stompy.legs.right.knee: np.deg2rad(6),  # 0.1
            Stompy.legs.left.ankle: np.deg2rad(0),  # 0
            Stompy.legs.right.ankle: np.deg2rad(0),  # 0
            Stompy.legs.left.foot_roll: np.deg2rad(0),  # 0
            Stompy.legs.right.foot_roll: np.deg2rad(0),  # 0
        }

    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            Stompy.head.left_right: {
                "lower": -0.1,
                "upper": 0.0,
            },
            Stompy.right_arm.shoulder_yaw: {
                "lower": 1.47,
                "upper": 1.50,
            },
            Stompy.left_arm.shoulder_yaw: {
                "lower": -1.23,
                "upper": -1.20,
            },
            Stompy.right_arm.shoulder_pitch: {
                "lower": 1.8,
                "upper": 1.83,
            },
            Stompy.left_arm.shoulder_pitch: {
                "lower": -1.63,
                "upper": -1.60,
            },
            Stompy.legs.right.hip_roll: {
                "lower": -0.75,
                "upper": -0.15,
            },
            Stompy.legs.left.hip_roll: {
                "lower": 0.2,
                "upper": 0.75,
            },
            Stompy.legs.right.hip_yaw: {
                "lower": -1,
                "upper": 0.0,
            },
            Stompy.legs.left.hip_yaw: {
                "lower": -0.71,
                "upper": -0.3,
            },
            Stompy.legs.right.hip_pitch: {
                "lower": -1.1,
                "upper": -0.4,
            },
            Stompy.legs.left.hip_pitch: {
                "lower": -0.1,
                "upper": 1.2,
            },
            Stompy.legs.right.knee: {
                "lower": 0,
                "upper": 1.2,
            },
            Stompy.legs.left.knee: {
                "lower": -1.2,
                "upper": 0,
            },
            Stompy.legs.right.ankle: {
                "lower": -0.3,
                "upper": 0.3,
            },
            Stompy.legs.left.ankle: {
                "lower": -0.3,
                "upper": 0.3,
            },
            Stompy.legs.right.foot_roll: {"lower": -0.3, "upper": 0.3},
            Stompy.legs.left.foot_roll: {"lower": -0.3, "upper": 0.3},
        }


def print_joints() -> None:
    joints = Stompy.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Stompy())


if __name__ == "__main__":
    # python -m sim.stompy.joints
    print_joints()