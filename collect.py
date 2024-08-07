"""Script to collect data from a real robot."""

# Based on https://github.com/tonyzhaozh/aloha/blob/main/aloha_scripts/record_episodes.py

import argparse
import multiprocessing
import os
import sys
import time
from typing import Any

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from data_collection.constants import CAM_HEIGHT, CAM_WIDTH, DT, TASK_CONFIGS
from data_collection.util import ImageRecorder
from demo import run_teleop_app
from env import make_real_env


def capture_one_episode(
    dt: float,
    max_timesteps: int,
    image_recorder: ImageRecorder,
    camera_pseudonyms: list,
    dataset_dir: str,
    dataset_name: str,
    overwrite: bool,
    shared_dict: dict,
    save_mp4: bool = False,
) -> bool:
    print(f"Dataset name: {dataset_name}")

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f"Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.")
        sys.exit()

    env = make_real_env(image_recorder, shared_data=shared_dict, save_mp4=save_mp4)

    # Wait for user input to start collecting data
    print("Press Enter to start collecting data")
    input()

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    for t in tqdm(range(max_timesteps)):
        t0 = time.time()
        action = env.get_action()
        t1 = time.time()
        ts = env.step(t0, action)
        t2 = time.time()
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 1:  # 41
        return False

    data_dict: dict[str, list] = {
        "/observations/qpos": [],
        #'/observations/qvel': [],
        #'/observations/effort': [],
        "/action": [],
    }

    if not save_mp4:
        for cam_name in camera_pseudonyms:
            data_dict[f"/observations/images/{cam_name}"] = []
    else:
        env.write_video()

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        # data_dict['/observations/qvel'].append(ts.observation['qvel'])
        # data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict["/action"].append(action)
        if not save_mp4:
            for cam_name in camera_pseudonyms:
                data_dict[f"/observations/images/{cam_name}"].append(ts.observation["images"][cam_name])

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        obs = root.create_group("observations")
        if not save_mp4:
            image = obs.create_group("images")
            for cam_name in camera_pseudonyms:
                if str(cam_name) not in image:
                    _ = image.create_dataset(
                        str(cam_name),
                        (max_timesteps, CAM_HEIGHT, CAM_WIDTH, 3),
                        dtype="uint8",
                        chunks=(1, CAM_HEIGHT, CAM_WIDTH, 3),
                    )
        _ = obs.create_dataset("qpos", (max_timesteps, 6))
        # _ = obs.create_dataset('qvel', (max_timesteps, 14))
        # _ = obs.create_dataset('effort', (max_timesteps, 14))
        _ = root.create_dataset("action", (max_timesteps, 7))

        for name, array in data_dict.items():
            if name == "/observations/images/" + camera_pseudonyms[0]:
                array = np.array(array)
            root[name][...] = array
    print(f"Saving: {time.time() - t0:.1f} secs")
    return True


def main(args: Any) -> None:
    if args["find_cameras"]:
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        print("Connected camera IDs:", arr)
        return


    task_config = TASK_CONFIGS[args["task_name"]]
    dataset_dir = task_config["dataset_dir"]
    max_timesteps: int = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    camera_pseudonyms = task_config["camera_keys"]
    overwrite = True

    image_recorder = ImageRecorder(camera_names, camera_pseudonyms, args["save_mp4"])

    manager = multiprocessing.Manager()
    shared_data = manager.dict()

    teleop_process = multiprocessing.Process(
            target=run_teleop_app, args=(True, 60, args["use_firmware"], shared_data)
    )

    teleop_process.start()

    time.sleep(5)

    while True:
        if args["episode_idx"] is not None:
            episode_idx = args["episode_idx"]
        else:
            episode_idx = get_auto_index(str(dataset_dir))

        dataset_name = f"episode_{episode_idx}"
        print(dataset_name + "\n")
        image_recorder.set_save_path(os.path.join(str(dataset_dir), dataset_name))
        for camera_id, camera_name in zip(camera_names, camera_pseudonyms):
            image_recorder.make_writer(camera_id, camera_name)

        is_healthy = capture_one_episode(
            DT,
            max_timesteps,
            image_recorder,
            camera_pseudonyms,
            str(dataset_dir),
            dataset_name,
            overwrite,
            shared_data,
            save_mp4=args["save_mp4"],
        )
        if not is_healthy:
            sys.exit()
        print("Next episode")
        image_recorder.close()
        time.sleep(5)


def get_auto_index(dataset_dir: str, dataset_name_prefix: str = "", data_suffix: str = "hdf5") -> int | Exception:
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f"{dataset_name_prefix}episode_{i}.{data_suffix}")):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history: list) -> float:
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    # dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(
        f"Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}"
    )
    return freq_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", action="store", type=str, help="Task name.", required=True)
    parser.add_argument("--episode_idx", action="store", type=int, help="Episode index.", default=None, required=False)
    parser.add_argument("--find_cameras", action="store_true", help="Find available cameras.", default=False)
    parser.add_argument("--use_firmware", action="store_true", help="Use firmware", default=False)
    parser.add_argument("--save_mp4", action="store_true", help="Save directly to mp4", default=False)
    main(vars(parser.parse_args()))
    # debug()
