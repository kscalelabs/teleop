import pybullet as p
import pybullet_data

physicsClient = p.connect(p.DIRECT)  # or p.GUI for graphical version

urdf_path = "urdf/stompy_7dof/multiarm.urdf"
robot = p.loadURDF(urdf_path, useFixedBase=True)


num_joints = p.getNumJoints(robot)
join = list()
print("Moving joints in the URDF:")
for i in range(num_joints):
    joint_info = p.getJointInfo(robot, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]


    if joint_type != p.JOINT_FIXED:
            # print(f"Joint index: {i}")
            # print(f"  Name: {joint_name}")

            # # Print joint type
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                print(f"  Name: {joint_name} {i}")
                join.append(joint_name)
                print("  Type: Revolute")
            # elif joint_type == p.JOINT_PRISMATIC:
            #     print("  Type: Prismatic")
            # elif joint_type == p.JOINT_SPHERICAL:
            #     print("  Type: Spherical")
            # elif joint_type == p.JOINT_PLANAR:
            #     print("  Type: Planar")
            # else:
            #     print(f"  Type: Other (type code: {joint_type})")

            # # Print joint limits
            # lower_limit, upper_limit = joint_info[8], joint_info[9]
            # if lower_limit == upper_limit:
            #     print("  Range: Continuous")
            # else:
            #     print(f"  Range: [{lower_limit:.2f}, {upper_limit:.2f}]")

            # # Print max force and max velocity
            # max_force, max_velocity = joint_info[10], joint_info[11]
            # print(f"  Max Force: {max_force}")
            # print(f"  Max Velocity: {max_velocity}")

print(join)
print(len(join))
