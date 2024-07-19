DATA_DIR = 'data'
TASK_CONFIGS: dict[str, dict[str, int | str | list]] = {
    'left_arm':{
        'dataset_dir': DATA_DIR + '/left_arm',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['/dev/video0'],
        'camera_keys': ['cam1']
    },
    'left_arm_mac':{
        'dataset_dir': DATA_DIR + '/left_arm',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': [0],
        'camera_keys': ['cam1']
    },
}
DT = 0.04
