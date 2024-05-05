import os
import roboticstoolbox as rtb
import spatialmath as sm

import numpy as np
from src.stompy import StompyFixed, joint_dict_to_list, qleft_to_dict, qright_to_dict

END_LINK = "link_right_arm_1_hand_1_x4_2_outer_1"

stompy_rtb = rtb.robot.Robot.URDF(
    # stompy_rtb = rtb.robot.ERobot.URDF(
    f"{os.path.dirname(__file__)}/urdf/stompy_tiny_glb/robot.urdf",
)
# print(stompy_rtb)

# q_grip = joint_dict_to_list([StompyFixed().default_standing(), {
#     "joint_right_arm_1_hand_1_slider_1" : -0.034,
#     "joint_right_arm_1_hand_1_slider_2" : -0.034,
#     "joint_left_arm_2_hand_1_slider_1" : -0.034,
#     "joint_left_arm_2_hand_1_slider_2" : -0.034,
# }])
# stompy_rtb.plot(q_grip)

# q_zeros = [0]*37
# stompy_rtb.plot(q_zeros)

# q_default = joint_dict_to_list([StompyFixed().default_standing()])
# stompy_rtb.plot(q_default)

q0 = [1.7, 1.6, 0.34, 1.6, 1.4, -0.26]
# q0 = [-1.86360008,  0.76775798,  0.36221785,  0.58700685, -1.55511321, -1.90407512]
# q0 = [1.49694154, 1.84065006, -0.0762903, 1.47331632, -3.05554518, 0.43140169]
q0_full = joint_dict_to_list([StompyFixed().default_standing(), qright_to_dict(q0)])
print(f"q0\n{q0}\n")
# stompy_rtb.plot(q0_full)
Tg = stompy_rtb.fkine(q0, end=END_LINK)
print(f"Tg\n{Tg}\n")
Tg *= sm.SE3.Trans(0.001, 0, 0)  # slightly move forward in X
# Tg *= sm.SE3.Trans(0.001, 0.001, 0.001) # slightly move forward in X
print(f"Tg\n{Tg}\n")
result = stompy_rtb.ik_NR(
# result = stompy_rtb.ik_LM(
    # result = stompy_rtb.ik_GN(
    # result = stompy_rtb.ikine_LM(
    Tg,
    pinv=True,
    end=END_LINK,
    # tol=1e-3,
    # tol=1e-6,
    # ilimit=100,
    # ilimit=1000,
    # slimit=3,
    # slimit=1000,
    # mask=[1, 1, 1, 0, 0, 0],  # only xyz
    mask=[8, 8, 8, 1, 1, 1],
    # mask=[100, 100, 100, 0, 0, 1],  # only xyz
    # mask=[100, 100, 100, 1, 1, 1],  # only xyz
    q0=q0,  # intial guess for faster solve
    # q0=[q + 0.1 for q in q0],
    # q0=[q + 0.2 for q in q0],
    # q0=[q0, [q + 0.01 for q in q0], [q - 0.01 for q in q0]], # intial guess for faster solve
    # joint_limits=False,  # this seems to be necessary
    # method='wampler',
    # k = 0.1,
    # method='sugihara',
    # k=0.01,
)
print(f"\n{result}\n")
qr = result[0]
# qr = result.q
qr_full = joint_dict_to_list([StompyFixed().default_standing(), qright_to_dict(qr)])
print(f"qr\n{qr}\n")
Tr = stompy_rtb.fkine(qr, end=END_LINK)
print(f"Tr\n{Tr}\n")
# stompy_rtb.plot(qr_full)
