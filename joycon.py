from time import sleep, time

import numpy as np
import pybullet as p
import pybullet_data
from pyjoycon import GyroTrackingJoyCon, get_R_id
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from mpl_toolkits.mplot3d import Axes3D

# Initialize PyBullet
# physicsClient = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, 0)
# p.loadURDF("plane.urdf")

# Create visual box
# joycon_visual = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.08, 0.15])
# joycon_body = p.createMultiBody(
#     baseMass=0,
#     baseCollisionShapeIndex=joycon_visual,
#     basePosition=[0, 0, 1],
#     baseOrientation=[0, 0, 0, 1]
# )

# Initialize tracking variables
position = np.array([0., 0., 1.])  # Starting position
velocity = np.array([0., 0., 0.])
last_time = time()

# Initialize JoyCon
joycon = GyroTrackingJoyCon(*get_R_id())
print("Calibrating - keep JoyCon still...")
joycon.calibrate(2)
joycon.reset_orientation()

# Calibration - extended
print("Calibrating - keep JoyCon still...")
accel_samples = []
gyro_samples = []
num_samples = 300  # 5 seconds at 60Hz
for _ in range(num_samples):
    accel_samples.append(np.array(joycon.accel_in_g[-1]))
    gyro_samples.append(np.array(joycon.gyro))
    sleep(1/60)

# Calculate statistics
accel_samples = np.array(accel_samples)
gyro_samples = np.array(gyro_samples)

# Calculate means
accel_mean = np.mean(accel_samples, axis=0)
gyro_mean = np.mean(gyro_samples, axis=0)

# Calculate variances
accel_var = np.var(accel_samples, axis=0)
gyro_var = np.var(gyro_samples, axis=0)

# Calculate biases
accel_bias = accel_mean - np.array([0, 0, -1])
gyro_bias = gyro_mean

print("\nAccelerometer Statistics:")
print(f"Mean (g):     {accel_mean}")
print(f"Variance (g²): {accel_var}")
print(f"Bias (g):     {accel_bias}")

print("\nGyroscope Statistics:")
print(f"Mean (deg/s):     {gyro_mean}")
print(f"Variance (deg²/s²): {gyro_var}")
print(f"Bias (deg/s):     {gyro_bias}")

position_bounds = 2.0
accel_threshold = 0.3

# Filter parameters
alpha = 0.1  # Complementary filter parameter
velocity_threshold = 0.05  # m/s
zero_velocity_count = 0
zero_velocity_threshold = 10  # frames

# Initialize plotting
plt.ion()  # Enable interactive mode
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224, projection='3d')
fig.suptitle('JoyCon Motion Tracking')

# Initialize data storage
window_size = 100
times = deque(maxlen=window_size)
accel_data = deque(maxlen=window_size)
vel_data = deque(maxlen=window_size)
pos_data = deque(maxlen=window_size)

# Initialize lines for each plot
lines_accel = [ax1.plot([], [], label=f'Accel {axis}')[0] for axis in ['X', 'Y', 'Z']]
lines_vel = [ax2.plot([], [], label=f'Velocity {axis}')[0] for axis in ['X', 'Y', 'Z']]
lines_pos = [ax3.plot([], [], label=f'Position {axis}')[0] for axis in ['X', 'Y', 'Z']]
line_3d = ax4.plot([], [], [], 'r-')[0]

# Configure plots
for ax in [ax1, ax2, ax3]:
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, 10)

ax1.set_ylabel('Acceleration (m/s²)')
ax2.set_ylabel('Velocity (m/s)')
ax3.set_ylabel('Position (m)')

ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('3D Position')
ax4.set_xlim(-position_bounds, position_bounds)
ax4.set_ylim(-position_bounds, position_bounds)
ax4.set_zlim(-position_bounds, position_bounds)

# Main Loop
start_time = time()
while True:
    current_time = time()
    dt = current_time - last_time

    # Get sensor data
    quat = -np.array(joycon.direction_Q)
    accel = np.array(joycon.accel_in_g[-1])

    # Normalize quaternion
    quat /= np.linalg.norm(quat)
    rot_matrix = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_matrix()

    # Process accelerometer data
    accel_corrected = accel - accel_bias
    accel_mps2 = accel_corrected * 9.81
    accel_world = rot_matrix @ accel_mps2

    # Remove gravity and apply better threshold
    gravity_vector_world = np.array([0, 0, -9.81])
    world_accel = accel_world - gravity_vector_world
    world_accel = np.where(np.abs(world_accel) < accel_threshold, 0, world_accel)

    # Zero-velocity update (ZUPT)
    if np.all(np.abs(world_accel) < 0.15) and np.all(np.abs(velocity) < velocity_threshold):
        zero_velocity_count += 1
        if zero_velocity_count > zero_velocity_threshold:
            velocity = np.zeros(3)
    else:
        zero_velocity_count = 0

    # Integrate with better numerical method (trapezoidal)
    velocity_new = velocity + world_accel * dt
    velocity = (1 - alpha) * velocity_new + alpha * velocity  # Complementary filter

    # Apply stronger velocity decay when acceleration is low
    if np.all(np.abs(world_accel) < 0.1):
        velocity *= 0.95  # Stronger decay when not accelerating
    else:
        velocity *= 0.99  # Light decay during movement

    # Position update with boundary checking
    position_new = position + velocity * dt
    if np.all(np.abs(position_new) < position_bounds):
        position = position_new

    # Update data collections
    times.append(current_time - start_time)
    accel_data.append(world_accel)
    vel_data.append(velocity)
    pos_data.append(position)

    # Update 2D plots
    for i in range(3):
        lines_accel[i].set_data(list(times), [a[i] for a in accel_data])
        lines_vel[i].set_data(list(times), [v[i] for v in vel_data])
        lines_pos[i].set_data(list(times), [p[i] for p in pos_data])

        # Dynamically adjust y-axis limits with some padding
        accel_data_i = [a[i] for a in accel_data]
        vel_data_i = [v[i] for v in vel_data]
        pos_data_i = [p[i] for p in pos_data]

        if accel_data_i:  # Only adjust if we have data
            accel_min, accel_max = min(accel_data_i), max(accel_data_i)
            vel_min, vel_max = min(vel_data_i), max(vel_data_i)
            pos_min, pos_max = min(pos_data_i), max(pos_data_i)

            padding = 0.1  # 10% padding above and below

            ax1.set_ylim(accel_min - abs(accel_min) * padding, 
                        accel_max + abs(accel_max) * padding)
            ax2.set_ylim(vel_min - abs(vel_min) * padding if vel_min != 0 else -0.1,
                        vel_max + abs(vel_max) * padding if vel_max != 0 else 0.1)
            ax3.set_ylim(pos_min - abs(pos_min) * padding if pos_min != 0 else -0.1,
                        pos_max + abs(pos_max) * padding if pos_max != 0 else 0.1)

    # Update 3D plot
    pos_array = np.array(list(pos_data))
    if len(pos_array) > 0:
        line_3d.set_data(pos_array[:, 0], pos_array[:, 1])
        line_3d.set_3d_properties(pos_array[:, 2])

    # Adjust x-axis limits to show last 10 seconds
    current_time = times[-1]
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(max(0, current_time - 10), current_time + 0.5)

    # Update the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Debug Information
    print(f"Time: {current_time - start_time:.2f} s")
    print(f"Raw accelerometer (g's): {accel}")
    print(f"Corrected accelerometer (g's): {accel_corrected}")
    print(f"Accelerometer (m/s²): {accel_mps2}")
    print(f"World acceleration (m/s²): {world_accel}")
    print(f"Velocity (m/s): {velocity}")
    print(f"Position (m): {position}")
    print("-" * 50)

    # # Update visualization
    # p.resetBasePositionAndOrientation(
    #     joycon_body,
    #     position.tolist(),
    #     [quat[1], quat[2], quat[3], quat[0]]
    # )

    # last_time = current_time

    # p.stepSimulation()
    sleep(1/60)
