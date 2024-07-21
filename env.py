"""Environment for real robot manipulation."""

import collections
import multiprocessing
import time
from typing import Any

import dm_env
import numpy as np

from data_collection.constants import DT, TIME_OFFSET
from data_collection.util import ImageRecorder
from demo import run_teleop_app


class RealEnv:
    """Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position.

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                        "images": {"cam1": (540x960x3),        # h, w, c, dtype='uint8'
                                   }
    """  # noqa: D205

    def __init__(
        self, cameras: list[Any], pseudonyms: list[str], firmware: bool, save_mp4: bool = False, save_path: str = ""
    ) -> None:
        print(cameras[0])
        self.image_recorder = ImageRecorder(cameras, pseudonyms, save_mp4, save_path=save_path)
        self.save_mp4 = save_mp4

        self.manager = multiprocessing.Manager()
        self.shared_data = self.manager.dict()

        self.stop_event = multiprocessing.Event()
        self.teleop_process = multiprocessing.Process(
            target=run_teleop_app, args=(True, 60, firmware, self.stop_event, self.shared_data)
        )
        self.teleop_process.start()
        time.sleep(5)

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
    cameras: list[Any], pseudonyms: list[str], firmware: bool = False, save_mp4: bool = False, save_path: str = ""
) -> RealEnv:
    env = RealEnv(cameras, pseudonyms, firmware=firmware, save_mp4=save_mp4, save_path=save_path)
    return env


# def test_real_teleop():
#     """
#     Test bimanual teleoperation and show image observations onscreen.
#     It first reads joint poses from both master arms.
#     Then use it as actions to step the environment.
#     The environment returns full observations including images.

#     An alternative approach is to have separate scripts for teleoperation and observation recording.
#     This script will result in higher fidelity (obs, action) pairs
#     """

#     onscreen_render = True
#     render_cam = 'cam_left_wrist'

#     # source of data
#     master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
#                                               robot_name=f'master_left', init_node=True)
#     master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
#                                                robot_name=f'master_right', init_node=False)
#     setup_master_bot(master_bot_left)
#     setup_master_bot(master_bot_right)

#     # setup the environment
#     env = make_real_env(init_node=False)
#     ts = env.reset(fake=True)
#     episode = [ts]
#     # setup visualization
#     if onscreen_render:
#         ax = plt.subplot()
#         plt_img = ax.imshow(ts.observation['images'][render_cam])
#         plt.ion()

#     for t in range(1000):
#         action = get_action(master_bot_left, master_bot_right)
#         ts = env.step(action)
#         episode.append(ts)

#         if onscreen_render:
#             plt_img.set_data(ts.observation['images'][render_cam])
#             plt.pause(DT)
#         else:
#             time.sleep(DT)


# if __name__ == '__main__':
#     test_real_teleop()
