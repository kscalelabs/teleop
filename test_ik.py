import os

import roboticstoolbox as rtb

from src.stompy import Stompy

# https://petercorke.github.io/robotics-toolbox-python/arm_superclass.html#roboticstoolbox.robot.Robot.Robot
stompy_rtb = rtb.robot.Robot.URDF(
    f"{os.path.dirname(__file__)}/urdf/stompy_tiny_gltf/robot.urdf",
    gripper=[Stompy.right_arm.hand.joints[0], Stompy.left_arm.hand.joints[0]],
)
print(stompy_rtb)

Tep = stompy_rtb.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
# https://petercorke.github.io/robotics-toolbox-python/IK/ik.html
joints = stompy_rtb.ik_LM(Tep)
