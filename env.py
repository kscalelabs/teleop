"""Environment for real robot manipulation."""

import collections
import time
from typing import Any

import dm_env
import numpy as np

from data_collection.constants import DT, TIME_OFFSET
from data_collection.util import ImageRecorder


class RealEnv:
    """Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6)]             # absolute joint position.

    Observation space: {"qpos": Concat[ left_arm_qpos (6),]          # absolute joint position
                        "qvel": Concat[ left_arm_qvel (6),]         # absolute joint velocity (rad)
                        "images": {"cam1": (510x910x3),}        # h, w, c, dtype='uint8'
                                   }
    """

    def __init__(
        self, img_recorder: ImageRecorder, shared_data: dict, save_mp4: bool = False) -> None:
        self.image_recorder = img_recorder
        self.save_mp4 = save_mp4

        self.shared_data = shared_data

    def get_qpos(self) -> np.ndarray:
        positions = self.shared_data["positions"]["actual"]["left"]
        return positions

    def get_qvel(self) -> np.ndarray:
        velocities = self.shared_data["velocities"]["left"]
        return velocities

    # def get_effort(self):
    #     left_effort_raw = self.recorder_left.effort
    #     right_effort_raw = self.recorder_right.effort
    #     left_robot_effort = left_effort_raw[:7]
    #     right_robot_effort = right_effort_raw[:7]
    #     return np.concatenate([left_robot_effort, right_robot_effort])

    def get_images(self) -> dict:
        return self.image_recorder.get_images()

    def get_observation(self) -> dict:
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        # obs['effort'] = self.get_effort()
        if not self.save_mp4:
            obs["images"] = self.get_images()
        return obs

    def reset(self, fake: bool = False) -> dm_env.TimeStep:
        if not fake:
            # Reboot puppet robot gripper motors
            # Add logic to reset arm motors
            pass
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0,
            discount=None,
            observation=self.get_observation(),
        )

    def write_video(self) -> None:
        self.image_recorder.close()

    def step(self, ref_time: float, action: np.ndarray) -> dm_env.TimeStep:
        self.image_recorder.update()

        # Maintain desired frequency by adjusting sleep time before next step
        # Calculated sleep time = desired - manual offset (small) - time taken for step
        calc_sleep = DT - (time.time() - ref_time) - TIME_OFFSET
        if calc_sleep > 0:
            time.sleep(calc_sleep)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0,
            discount=None,
            observation=self.get_observation(),
        )

    def get_action(self) -> np.ndarray:
        action = np.zeros(7)  # 5 joint + 2 gripper
        # Arm actions
        action[:7] = self.shared_data["positions"]["expected"]["left"]

        return action


def make_real_env(
    image_recorder: ImageRecorder, shared_data: dict = {}, save_mp4: bool = False
) -> RealEnv:
    env = RealEnv(image_recorder, shared_data=shared_data, save_mp4=save_mp4)
    return env

