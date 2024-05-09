<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/teleop/blob/master/LICENSE)
[![Version](https://img.shields.io/pypi/v/kscale-onshape-library)](https://pypi.org/project/kscale-onshape-library/)
[![Discord](https://dcbadge.limes.pink/api/server/k5mSvCkYQh?style=flat)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)

</div>
<p align="center">
  <picture>
    <img alt="dalle3" src="assets/cover.png" style="max-width: 100%;">
  </picture>
  <br/>
  <picture>
    <img alt="demo" src="https://giphy.com/gifs/GyOOrsqLv77JgJiSBT" style="max-width: 100%;">
  </picture>
  <br/>
</p>


<h1 align="center">
    <p>Bi-Manual Remote Robotic Teleoperation</p>
</h1>

A minimal implementation of a bi-manual remote robotic teleoperation system using VR hand tracking and camera streaming.

✅ VR and browser visualization

✅ bi-manual hand gesture control

✅ camera streaming (mono + stereo)

✅ inverse kinematics

✅ Meta Quest Pro HMD + NVIDIA® Jetson AGX Orin™ Developer Kit

✅ `.urdf` robot model

✅ 3dof end effector control

⬜️ debug 6dof end effector control

⬜️ resets to various default pose 

⬜️ tested on real world robot

⬜️ record & playback trajectories


### Setup

```bash
git clone https://github.com/kscalelabs/teleop.git && cd teleop
conda create -y -n teleop python=3.8 && conda activate teleop
pip install -r requirements.txt
```

### Usage

Start the server on the robot computer.

```bash
python demo_hands_stereo_ik3dof.py
```

Start ngrok on the robot computer.

```bash
ngrok http 8012
```

Open the browser app on the HMD and go to the ngrok URL.

### Dependencies

- [Vuer](https://github.com/vuer-ai/vuer) is used for visualization
- [PyBullet](https://pybullet.org/wordpress/) is used for IK
- [ngrok](https://ngrok.com/download) is used for networking


### Citation

```
@misc{teleop-2024,
  title={Bi-Manual Remote Robotic Teleoperation},
  author={Hugo Ponte},
  year={2024},
  url={https://github.com/kscalelabs/teleop}
}
```
