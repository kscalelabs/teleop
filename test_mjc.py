from dm_control.utils import inverse_kinematics as ik
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control import mjcf
import mediapy as media
import numpy as np

mjlib = mjbindings.mjlib

_TOL = 1.2
_MAX_STEPS = 100
_MAX_RESETS = 10

STOMPY_XML = "urdf/stompy_new/meshes/robot.xml"

target_pos = np.array([0.0, 0.3, 0.3])
target_quat = np.array([0., 0., 0., 1.])
physics = mujoco.Physics.from_xml_path(STOMPY_XML)
model = mujoco.MjModel.from_xml_path(STOMPY_XML)
data = mujoco.MjData(model)
count = 0
physics2 = physics.copy(share_model=True)

media.write_image("first_position.jpg", physics2.render()) 

_JOINTS = [
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8",
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4",
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4",
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4"
]

_SITE_NAME = "target_site"

# Extract joint name
while True:
    result = ik.qpos_from_site_pose(
        physics=physics2,
        site_name=_SITE_NAME,
        target_pos=target_pos,
        target_quat=target_quat,
        joint_names=_JOINTS,
        tol=_TOL,
        max_steps=_MAX_STEPS,
        inplace=True,
    )
    print(result)
    if result.success:
        break
    
    count += 1

print(result)
# visualize the final position
# media.show(physics2.render(), "final position")


# loop simulating the final position
while True:
    physics2.step()
    media.write_image("final_position.jpg", physics2.render()) 
    if count > _MAX_RESETS:
        break
    count += 1
    print(count)