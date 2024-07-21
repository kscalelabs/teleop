# Teleop Data Collection for Stompy

For convenience, install the following on Jetson to view connected cameras:
```
sudo apt install v4l-utils
```
and run 
```
v4l2-ctl --list-devices
```
to view the camera devices that you slot into camera_names in the config

### Collecting Data

To collect data, run the following command:
```
python data_collection.py --task_name CONFIG_TASK_NAME --use_firmware True --save_mp4
```
where CONFIG_TASK_NAME is the name of the task you want to collect data for. The config can be found in constants.py.\
--use_firmware is an optional argument that can be set to True if you want to use the firmware to control the robot. If set to False, images will be collected without any robot control.\
--save_mp4 is an optional argument that can be set to True if you want to save images directly to an mp4 file. Note that this slows down frequency of data collection (to around 20hz on Jetson NX)

To view the collected data, run the following command:
```
python view_data.py --dataset_dir PATH/TO/DATASET --episode_idx EPISODE_IDX
```
where PATH/TO/DATASET is the path to the dataset you want to view and EPISODE_IDX is the index of the episode you want to view.

### Script Structure

The data_collection.py script is structured as follows:
1. Initialize the env (robot + cameras)
2. Load the config for the task
3. Collect data for an episode
    1. Start teleop app
    2. Collect data for each step
        1. Collect images
        2. Collect robot state from app
        3. Collect action from app
    3. Save episode

