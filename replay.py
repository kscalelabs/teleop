"""Script to replay qpos data from HDF5 file on the real robot."""

import argparse
import os
import sys
import time

import h5py

from data_collection.constants import DT
from firmware.robot.robot import Robot


def main(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    dataset_name = f"episode_{episode_idx}"

    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        sys.exit()

    with h5py.File(dataset_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        action = root["/action"][()]
        timesteps = qpos.shape[0]

    print(f"qpos: {qpos.shape}")
    print(f"action: {action.shape}")
    print(f"timesteps: {timesteps}")

    robot = Robot(config_path="config.yaml", setup="left_arm_replay")

    if True:
        robot.zero_out()

    for t in range(timesteps):
        start_time = time.time()
        robot.set_position({"left_arm": qpos[t]}, radians=False)
        robot.update_motor_data()
        print(f"Robot at {robot.get_motor_positions()}")
        print(f"Moving to {qpos[t]}")
        wait_time = DT - (time.time() - start_time)
        if wait_time > 0:
            time.sleep(wait_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay qpos data from HDF5 file on the real robot.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing HDF5 files.")
    parser.add_argument("--episode_idx", type=int, required=True, help="Index of the episode to replay.")

    main(parser.parse_args())

