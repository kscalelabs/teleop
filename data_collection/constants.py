
DATA_DIR = 'data'
TASK_CONFIGS: dict[str, dict[str, int | str | list]] = {
    'aloha_wear_shoe':{
        'dataset_dir': DATA_DIR + '/aloha_wear_shoe',
        'num_episodes': 50,
        'episode_len': 100,
        'camera_names': [0]
    },
}
DT = 0.02
