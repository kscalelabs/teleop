from setuptools import setup

# Read the contents of requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="teleop",
    version="0.1",
    packages=["data_collection"],
    author="KScale Labs",
    description="Bi-Manual Remote Robotic Teleoperation",
    url="https://github.com/kscalelabs/teleop",
    # Use the requirements read from requirements.txt
    install_requires=requirements,
    python_requires=">=3.6",
    # Additional classifiers for package indexing
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
