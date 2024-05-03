from asyncio import sleep
import os
import math

from vuer import Vuer, VuerSession
from vuer.schemas import Scene, Urdf, Movable, PointLight, AmbientLight

from src.stompy import StompyFixed

app = Vuer(static_root=f"{os.path.dirname(__file__)}/urdf/stompy_tiny_glb")

amplitudes = {
    k: abs(v["upper"] - v["lower"])
    for k, v in StompyFixed.default_limits(StompyFixed).items()
}


@app.spawn(start=True)
async def main(app: VuerSession):
    app.set @ Scene(
        rawChildren=[
            AmbientLight(intensity=1),
            Movable(PointLight(intensity=1), position=[0, 0, 2]),
            Movable(PointLight(intensity=3), position=[0, 1, 2]),
        ],
        grid=True,
        up=[0, 0, 1],
    )
    await sleep(0.1)
    app.upsert @ Urdf(
        src="http://localhost:8012/static/robot.urdf",
        jointValues=StompyFixed.default_standing(),
        position=[0, 0, 1],
        key="stompy",
    )
    await sleep(0.1)
    i = 0
    while True:
        app.upsert @ Urdf(
            src="http://localhost:8012/static/robot.urdf",
            jointValues={
                value + amplitudes[joint_name] * math.sin(i * 0.1) for joint_name, value in StompyFixed.default_standing().items()
            },
            position=[0, 0, 1],
            key="stompy",
        )
        await sleep(0.016)
        i += 1
