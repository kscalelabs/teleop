"""Constants for data collection."""

DATA_DIR = "data"
TASK_CONFIGS: dict[str, dict[str, int | str | list]] = {
    "left_arm": {
        "dataset_dir": DATA_DIR + "/left_arm",
        "episode_len": 300,
        "camera_names": ["/dev/video0"],
        "camera_keys": ["cam1"],
    },
    "left_arm_mac": {
        "dataset_dir": DATA_DIR + "/left_arm",
        "episode_len": 600,
        "camera_names": [0],
        "camera_keys": ["cam1"],
    },
    "left_arm_long": {
        "dataset_dir": DATA_DIR + "/left_arm",
        "episode_len": 1000,
        "camera_names": ["/dev/video0"],
        "camera_keys": ["cam1"],
    },
}
DT = 0.04
TIME_OFFSET = 0.002
CAM_HEIGHT = 510
CAM_WIDTH = 910
