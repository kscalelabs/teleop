DATA_DIR = 'data'
TASK_CONFIGS: dict[str, dict[str, int | str | list]] = {
    'left_arm':{
        'dataset_dir': DATA_DIR + '/left_arm',
        'num_episodes': 50,
        'episode_len': 100,
        'camera_names': ['/dev/video1'],
        'camera_keys': ['cam1']
    },
}
DT = 0.02
