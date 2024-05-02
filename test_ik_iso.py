import os
import roboticstoolbox as rtb
import spatialmath as sm

# env = swift.Swift()
# env.launch(realtime=True)

stompy_rtb = rtb.robot.Robot.URDF(f"{os.path.dirname(__file__)}/urdf/stompy_tiny/robot.urdf")
Tep = sm.SE3.Trans(-0.2, -0.2, 0) * sm.SE3.OA([0, 1, 0], [0, 0, -1])
q0_right = [1.7, 1.6, 0.34, 1.6, 1.4, -0.26]
result = stompy_rtb.ik_LM(
    Tep,
    end="link_right_arm_1_hand_1_base_1",
    tol=1e-3,
    mask=[1, 1, 1, 0, 0, 0], # which DOF to solve for (this is just xyz)
    q0=q0_right, # intial guess for faster solve
    # joint_limits=True,
)
print(result)

q_pickup = result[0]
print(stompy_rtb.fkine(q_pickup, end="link_right_arm_1_hand_1_base_1"))

joint_list = [
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
joint_map = {joint: i for i, joint in enumerate(joint_list)}
arm_chain = [
    "joint_right_arm_1_x8_1_dof_x8",
    "joint_right_arm_1_x8_2_dof_x8",
    "joint_right_arm_1_x6_1_dof_x6",
    "joint_right_arm_1_x6_2_dof_x6",
    "joint_right_arm_1_x4_1_dof_x4",
    "joint_right_arm_1_hand_1_x4_1_dof_x4",
]
arm_chain = [joint_map[joint] for joint in arm_chain]
pose = stompy_rtb.q.copy()
for i in range(6):
    pose[arm_chain[i]] = q0_right[i]

stompy_rtb.plot(pose, backend='pyplot', block=True)