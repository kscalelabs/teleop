from asyncio import sleep
import math

from vuer import Vuer, VuerSession
from vuer.schemas import Urdf, Scene, Movable, PointLight, Hands

app = Vuer()

stompy_default_pos = {
    "joint_head_1_x4_1_dof_x4": -0.03490658503988659,
    "joint_legs_1_x8_2_dof_x8": 0.5061454830783556,
    "joint_legs_1_left_leg_1_x8_1_dof_x8": -0.5061454830783556,
    "joint_legs_1_left_leg_1_x10_1_dof_x10": 0.9773843811168246,
    "joint_legs_1_x8_1_dof_x8": -0.5061454830783556,
    "joint_legs_1_right_leg_1_x8_1_dof_x8": -0.5061454830783556,
    "joint_legs_1_right_leg_1_x10_2_dof_x10": -0.9773843811168246,
    "joint_legs_1_left_leg_1_knee_revolute": -0.10471975511965978,
    "joint_legs_1_right_leg_1_knee_revolute": 0.10471975511965978,
    "joint_legs_1_left_leg_1_ankle_revolute": 0.0,
    "joint_legs_1_right_leg_1_ankle_revolute": 0.0,
    "joint_legs_1_left_leg_1_x4_1_dof_x4": 0.0,
    "joint_legs_1_right_leg_1_x4_1_dof_x4": 0.0,
    # left arm
    "joint_left_arm_2_x8_1_dof_x8": -1.7,
    "joint_left_arm_2_x8_2_dof_x8": -1.6,
    "joint_left_arm_2_x6_1_dof_x6": -0.34,
    "joint_left_arm_2_x6_2_dof_x6": -1.6,
    "joint_left_arm_2_x4_1_dof_x4": -1.4,
    "joint_left_arm_2_hand_1_x4_1_dof_x4": -1.7,
    # right arm
    "joint_right_arm_1_x8_1_dof_x8": 1.7,
    "joint_right_arm_1_x8_2_dof_x8": 1.6,
    "joint_right_arm_1_x6_1_dof_x6": 0.34,
    "joint_right_arm_1_x6_2_dof_x6": 1.6,
    "joint_right_arm_1_x4_1_dof_x4": 1.4,
    "joint_right_arm_1_hand_1_x4_1_dof_x4": -0.26,
}


@app.spawn(start=True)
async def main(session: VuerSession):
    # session.set @ Scene(
    #     rawChildren=[
    #         Movable(PointLight(intensity=1), position=[0, 0, 2]),
    #     ],
    #     grid=True,
    #     up=[0, 0, 1],
    # )
    session.upsert @ Hands(fps=60, stream=True, key="hands")
    session.upsert @ Urdf(
        src="https://raw.githubusercontent.com/hu-po/webstompy/main/urdf/stompy/robot.urdf",
        jointValues=stompy_default_pos,
        # position=[0, 0, 1],
        position=[0, 1, 0],
        rotation=[-math.pi / 2, 0, 0],
        key="stompy",
    )

    # keep the session alive.
    while True:
        await sleep(10)