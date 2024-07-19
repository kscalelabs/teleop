# Teleop Data Collection for Stompy

### Install

Installing h5py on Jetson is weird, so to install do the following:
```
git clone https://github.com/h5py/h5py.git
git checkout 3.1.0
git cherry-pick 3bf862daa4ebeb2eeaf3a0491e05f5415c1818e4
```

and run the bash file:
```source dev-install.sh```

If all else fails, refer to: https://forums.developer.nvidia.com/t/failed-building-wheel-of-h5py/263322/5 (solution pasted below)
```sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
pip3 install Cython==0.29.36
pip3 install pkgconfig

H5PY_SETUP_REQUIRES=0 pip3 install . --no-deps --no-build-isolation
sudo pip3 install -U numpy==1.19.4 future mock keras_preprocessing keras_applications gast==0.2.1 protobuf pybind11 packaging
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow
```


After installing h5py, you should install the following on the Jetson to view connected cameras:
```
sudo apt install v4l-utils
```

and run 
```
v4l2-ctl --list-devices
```

to view the camera devices that you slot into camera_names in the config

