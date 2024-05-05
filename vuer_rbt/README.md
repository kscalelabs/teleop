# Vuer (Python) + Robotics Toolbox (Python/C)

✅ vr hands

✅ working server on pc

⬜️ working server on orin

✅ vr hand interactions

✅ movable objects

⬜️ debug ik issues

⬜️ async launch script


### Setup

```
conda create -n teleop python=3.8
conda activate teleop
pip install vuer[all]
pip install opencv-python
pip install roboticstoolbox-python
```

### Usage

#### Local (Browser)

run one of the examples, e.g. the urdf loader

```
python test_urdf.py
```

then go to `http://localhost:8012`

#### Remote (VR)

run an example

```
python test_urdf.py
```

Run [ngrok](https://ngrok.com/download).

```
ngrok http 8012
```

open the browser app in the VR headset and go to the ngrok URL.


### References

This repo is based on the following repos:

- [Vuer](https://github.com/vuer-ai/vuer)
- [Teleop](https://github.com/OpenTeleop/Teleop)
- [kscalelabs/sim](https://github.com/kscalelabs/sim)
