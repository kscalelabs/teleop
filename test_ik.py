import os
import time

import roboticstoolbox as rtb

from src.stompy import Stompy

# https://petercorke.github.io/robotics-toolbox-python/arm_superclass.html#roboticstoolbox.robot.Robot.Robot
left_arm_rtb = rtb.robot.Robot.URDF(
    f"{os.path.dirname(__file__)}/urdf/stompy_tiny/robot.urdf",
    gripper="link_left_arm_2_hand_1_x4_1_inner_1",
)
# print(left_arm_rtb)
right_arm_rtb = rtb.robot.Robot.URDF(
    f"{os.path.dirname(__file__)}/urdf/stompy_tiny/robot.urdf",
    gripper="link_right_arm_1_hand_1_x4_2_inner_1",
)
# print(right_arm_rtb)

q0 = [v for k, v in Stompy.default_standing().items()]
print(f"target_joints: {q0}")
Tep = left_arm_rtb.fkine(q0)
print(f"Tep: {Tep}")
# https://petercorke.github.io/robotics-toolbox-python/IK/ik.html
start_time = time.time()
result = left_arm_rtb.ik_LM(
    Tep,
    mask=[1, 1, 1, 0, 0, 0], # which DOF to solve for (this is just xyz)
    q0=q0, # intial guess for faster solve
    joint_limits=True,
)
print(f"elapsed time: {time.time() - start_time}")
q, success, num_iter, num_search, residual = result
print(f"ik joints: {q}")
eepos = left_arm_rtb.fkine(q)
print(f"eepos: {eepos}")
