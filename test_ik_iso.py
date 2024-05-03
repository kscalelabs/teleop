import os
import roboticstoolbox as rtb
import spatialmath as sm

from src.stompy import Stompy, joint_dict_to_list

stompy_rtb = rtb.robot.Robot.URDF(
    f"{os.path.dirname(__file__)}/urdf/stompy_tiny_glb/robot.urdf"
)
q_default = joint_dict_to_list([Stompy().default_standing()])


print("------------------------ Pose 1")
q0 = [1.7, 1.6, 0.34, 1.6, 1.4, -0.26] # TODO: If you jitter this does it still solve it?
Tee = stompy_rtb.fkine(q0, end="link_right_arm_1_hand_1_base_1")
print(f"Manipulation ready ee pose \n {Tee}")
result0 = stompy_rtb.ikine_GN(
    Tee,
    end="link_right_arm_1_hand_1_base_1",
    tol=1e-3,
    # mask=[1, 1, 1, 0, 0, 0],  # which DOF to solve for (this is just xyz)
    q0=q0, # intial guess for faster solve
    joint_limits=True,
)
print(f"IK result is {result0}")
# q_ee0 = result0[0]
q_ee0 = result0.q

print("------------------------ Pose 2")
q1 = [-1.86360008,  0.76775798,  0.36221785,  0.58700685, -1.55511321, -1.90407512] 
Tee = stompy_rtb.fkine(q1, end="link_right_arm_1_hand_1_base_1")
print(f"Manipulation ready ee pose \n {Tee}")
result1 = stompy_rtb.ikine_GN(
    Tee,
    end="link_right_arm_1_hand_1_base_1",
    tol=1e-3,
    # mask=[1, 1, 1, 0, 0, 0],  # which DOF to solve for (this is just xyz)
    q0=q1, # intial guess for faster solve
    joint_limits=True,
)
print(f"IK result is {result1}")
# q_ee1 = result0[0]
q_ee1 = result0.q

def q_to_dict(q):
    return {
    "joint_right_arm_1_x8_1_dof_x8" : q[0],
    "joint_right_arm_1_x8_2_dof_x8" : q[1],
    "joint_right_arm_1_x6_1_dof_x6" : q[2],
    "joint_right_arm_1_x6_2_dof_x6" : q[3],
    "joint_right_arm_1_x4_1_dof_x4" : q[4],
    "joint_right_arm_1_hand_1_x4_1_dof_x4" : q[5],
    }

q_start = joint_dict_to_list([Stompy().default_standing(), q_to_dict(q_ee0)])
q_end = joint_dict_to_list([Stompy().default_standing(), q_to_dict(q_ee1)])
qt = rtb.jtraj(q_start, q_end, 50)
# stompy_rtb.plot(qt.q, backend="pyplot", block=True, loop=True, jointaxes=False)
stompy_rtb.plot(qt.q)

# env = swift.Swift()
# env.launch(realtime=True)



# Tep = sm.SE3.Trans(-0.2, -0.2, 0) * sm.SE3.OA([0, 1, 0], [0, 0, -1])
# q0_right = [1.7, 1.6, 0.34, 1.6, 1.4, -0.26]
# result = stompy_rtb.ik_LM(
#     Tep,
#     end="link_right_arm_1_hand_1_base_1",
#     tol=1e-3,
#     mask=[1, 1, 1, 0, 0, 0],  # which DOF to solve for (this is just xyz)
#     q0=q0_right,  # intial guess for faster solve
#     # joint_limits=True,
# )
# print(result)
# q_pickup = result[0]
# print(stompy_rtb.fkine(q_pickup, end="link_right_arm_1_hand_1_base_1"))



