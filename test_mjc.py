import mediapy as media
import numpy as np
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import inverse_kinematics as ik
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mjlib = mjbindings.mjlib
_TOL = 0.2
_MAX_STEPS = 100
STOMPY_XML = "urdf/stompy_new/meshes/robot.xml"
target_quat = np.array([1., 0., 0., 1.])
physics = mujoco.Physics.from_xml_path(STOMPY_XML)
model = mujoco.MjModel.from_xml_path(STOMPY_XML)
data = mujoco.MjData(model)
count = 0
physics2 = physics.copy(share_model=True)

initial_image = physics2.render()

_JOINTS = [
    'joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8',
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8",
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4",
    "joint_full_arm_5_dof_1_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4",
    "joint_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4"
]

_SITE_NAME = "target_site"

target_pos = np.array([0.0, 0.3, 0.])
target_pos = np.array([-0.2, 0, 0])

# Extract joint name
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


# loop simulating the final position
physics2.step()

print("IK result:", result)
print("Final joint positions:", result.qpos)

# Step the simulation and render final position
final_image = physics2.render()

# Plot the images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.imshow(initial_image)
ax1.set_title('Initial Position')
ax1.axis('off')

ax2.imshow(final_image)
ax2.set_title('Final Position After IK')
ax2.axis('off')

plt.tight_layout()
plt.show()

# Optionally, save the comparison image
plt.savefig('ik_comparison.png')