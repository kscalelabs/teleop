"""Script to convert data from HDF5 file to rerun format."""

import argparse
import glob
import os

import cv2
import h5py
import rerun as rr


def make_rerun(hdf5: str, cams: list[str]) -> None:
    # Open the HDF5 file
    with h5py.File(hdf5, "r") as f:
        # Get the number of timesteps
        max_timesteps = f["action"].shape[0]

        # Create a dictionary to store video captures for each camera
        video_captures = {}

        joint_names = ["shoulder_pitch", "shoulder_roll", "elbow_roll", "elbow_pitch", "wrist", "gripper"]

        for t in range(max_timesteps):
            # Log timestamp
            rr.set_time_sequence("timestep", t)

            # Log robot joint positions
            qpos = f["observations/qpos"][t]
            action = f["action"][t]
            error = [action[i] - qpos[i] for i in range(len(qpos))]

            for i, joint_name in enumerate(joint_names):
                rr.log(f"robot/{joint_name}/{joint_name}_expected", rr.Scalar(action[i]))
                rr.log(f"robot/{joint_name}/{joint_name}_actual", rr.Scalar(qpos[i]))
                rr.log(f"robot/{joint_name}/{joint_name}_error", rr.Scalar(error[i]))

            # Log images from video files
            for cam_name in cams:
                vid_path = hdf5.split(".")[0] + f" camera_{cam_name}.mp4"
                if os.path.exists(vid_path):
                    if cam_name not in video_captures:
                        video_captures[cam_name] = cv2.VideoCapture(vid_path)

                    ret, frame = video_captures[cam_name].read()
                    if ret:
                        rr.log(f"camera/{cam_name}", rr.Image(frame))
                    else:
                        print(f"Warning: Could not read frame for {cam_name} at timestep {t}")

        # Release all video captures
        for cap in video_captures.values():
            cap.release()

def main(args: dict) -> None:
    # Initialize rerun
    rr.init("my_robot_data")
    cam_list = args['cams'].split(",")
    if args['save_all']:
        hdf5list = glob.glob(f"{args['dataset_dir']}/*.hdf5")
        for hdf5 in hdf5list:
            rr.init(args['dataset_dir'].split("/")[1])
            make_rerun(hdf5, cam_list)
            rr.save(f"{hdf5.split('.')[0]}.rrd")
    elif args['rerun']:
        rr.log_file_from_path(args.rerun)
        rr.spawn()
    else:
        rr.init("my_robot_data")
        make_rerun(args['dataset_dir'], cam_list)
        if args["save"]:
            rr.save(args["dataset_dir"].split(".")[0] + ".rrd")
        if args["view"]:
            rr.spawn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save rerun file", default=False)
    parser.add_argument("--view", action="store_true", help="Open in viewer", default=False)
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--cams", type=str, help="Comma separated cam names", default="cam1")
    parser.add_argument("--save_all", action="store_true", help="Convert all episodes", default=False)
    parser.add_argument("--rerun", type=str, help="pass a rerun file to view", default=None)
    main(vars(parser.parse_args()))
