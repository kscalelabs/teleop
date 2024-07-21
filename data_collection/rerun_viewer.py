"""Script to convert data from HDF5 file to rerun format."""

import argparse

import h5py
import rerun as rr


def main(args: dict) -> None:
    # Initialize rerun
    rr.init("my_robot_data")

    # Open the HDF5 file
    with h5py.File("data/left_arm/episode_7.hdf5", "r") as f:
        # Get the number of timesteps
        max_timesteps = f["action"].shape[0]

        for t in range(max_timesteps):
            # Log timestamp
            rr.set_time_sequence("timestep", t)

            # Log robot joint positions
            qpos = f["observations/qpos"][t]
            for i, joint_name in enumerate(["joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]):
                rr.log(f"robot/joint/{joint_name}", rr.Scalar(qpos[i]))

            # Log robot actions
            action = f["action"][t]
            rr.log("robot/action", rr.Scalar(action))

            # Log images (if available)
            for cam_name in ["cam1"]:  # adjust based on your camera names
                if f"observations/images/{cam_name}" in f:
                    image = f[f"observations/images/{cam_name}"][t]
                    rr.log(f"camera/{cam_name}", rr.Image(image))
    if args["save"]:
        rr.save("my_recording.rrd")

    if args["view"]:
        rr.spawn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save rerun file", default=False)
    parser.add_argument("--view", action="store_true", help="Open in viewer", default=False)
    main(vars(parser.parse_args()))
