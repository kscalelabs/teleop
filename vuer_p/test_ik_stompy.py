import os
import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
from typing import List, Dict

clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
  p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot_start_pos = [0, 0, 1]
robot_start_euler = [-math.pi/4, 0, 0]
id = p.loadURDF(
  f"{os.path.dirname(__file__)}/../urdf/stompy_tiny/robot.urdf",
  [0, 0, 0],
  useFixedBase=True,
  )
p.resetBasePositionAndOrientation(id, robot_start_pos, p.getQuaternionFromEuler(robot_start_euler))

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
ee_link_right = 'link_right_arm_1_hand_1_x4_2_outer_1'
ee_link_left = 'link_left_arm_2_hand_1_x4_2_outer_1'

numJoints = p.getNumJoints(id)
jNames = []
jChildNames = []
jUpperLimit = []
jLowerLimit = []
jRestPose = []
jDamping = []
j_dof_idx_map = {}
# https://github.com/bulletphysics/bullet3/blob/e9c461b0ace140d5c73972760781d94b7b5eee53/examples/SharedMemory/SharedMemoryPublic.h#L292
for i in range(numJoints):
  info = p.getJointInfo(id, i)
  joint_name = info[1].decode("utf-8")
  jNames.append(joint_name)
  jChildNames.append(info[12].decode("utf-8"))
  jLowerLimit.append(info[9])
  jUpperLimit.append(info[10])
  if joint_name in stompy_default_pos:
    jRestPose.append(stompy_default_pos[joint_name])
    jDamping.append(0.1)
  else:
    jRestPose.append(0)
    jDamping.append(10)

ee_idx_right = jChildNames.index(ee_link_right)
ee_idx_left = jChildNames.index(ee_link_left)

# IK solver outputs 37 length vector with DOF joints in the following order
ik_joints_names = [
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

def make_full_joint_list(ik_joints: List[float]) -> List[float]:
  full_joints = jRestPose.copy()
  for i, j in enumerate(ik_joints):
    full_joints[jNames.index(ik_joints_names[i])] = j
  return full_joints
    

ll = jLowerLimit
ul = jUpperLimit
jr = [ul[i] - ll[i] for i in range(numJoints)]
rp = jRestPose
jd = jDamping

for i in range(numJoints):
  p.resetJointState(id, i, jRestPose[i])

p.setGravity(0, 0, 0)
t = 0.
# prevPose = [0, 0, 0]
# prevPose1 = [0, 0, 0]
# hasPrevPose = 0
useNullSpace = 1

useOrientation = 1
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 0
useRealTimeSimulation = 0
ikSolver = 0
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15

i=0
while 1:
  i+=1
  #p.getCameraImage(320,
  #                 200,
  #                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
  #                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
  if (useRealTimeSimulation):
    dt = datetime.now()
    t = (dt.second / 60.) * 2. * math.pi
  else:
    t = t + 0.01

  if (useSimulation and useRealTimeSimulation == 0):
    p.stepSimulation()

  for pos, ee_idx in [
    ([0.2 + 0.2 * math.cos(t), 0.3, -0.2 + 0.2 * math.sin(t)], ee_idx_right),
    ([-0.2 + 0.2 * math.cos(t), 0.3, -0.2 + 0.2 * math.sin(t)], ee_idx_left)
  
  ]:
    pos = [pos[0] + robot_start_pos[0], pos[1] + robot_start_pos[1], pos[2] + robot_start_pos[2]]
    #end effector points down, not up (in case useOrientation==1)
    orn = p.getQuaternionFromEuler([-math.pi/2, 0, 0])

    if (useNullSpace == 1):
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(id, ee_idx, pos, orn, ll, ul,
                                                  jr, rp)
      else:
        jointPoses = p.calculateInverseKinematics(id,
                                                  ee_idx,
                                                  pos,
                                                  lowerLimits=ll,
                                                  upperLimits=ul,
                                                  jointRanges=jr,
                                                  restPoses=rp)
    else:
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(id,
                                                  ee_idx,
                                                  pos,
                                                  orn,
                                                  jointDamping=jd,
                                                  solver=ikSolver,
                                                  maxNumIterations=100,
                                                  residualThreshold=.01)
      else:
        jointPoses = p.calculateInverseKinematics(id,
                                                  ee_idx,
                                                  pos,
                                                  solver=ikSolver)

    jointPoses = make_full_joint_list(jointPoses)
    if (useSimulation):
      for i in range(numJoints):
        p.setJointMotorControl2(bodyIndex=id,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(numJoints):
        p.resetJointState(id, i, jointPoses[i])

  # ls = p.getLinkState(id, ee_idx_right)
  # if (hasPrevPose):
  #   p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
  #   p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
  # prevPose = pos
  # prevPose1 = ls[4]
  # hasPrevPose = 1
p.disconnect()