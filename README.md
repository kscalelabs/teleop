<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/cover.png">
    <img alt="Teleop" src="assets/cover.png" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>
<h3 align="center">
    <p>Bi-Manual Remote Robotic Teleoperation</p>
</h3>

---

A minimal implementation of a bi-manual remote robotic teleoperation system using VR hand tracking and camera streaming.

✅ VR and browser visualization

✅ bi-manual hand gesture control

✅ camera streaming

✅ inverse kinematics

✅ Meta Quest Pro HMD + NVIDIA® Jetson AGX Orin™ Developer Kit

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
python demo_ik_hands_stereo.py
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
