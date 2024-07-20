# Makefile

define HELP_MESSAGE
teleop


# Installing

1. Create a new Conda environment: `conda create -y -n teleop python=3.11`
2. Activate the environment: `conda activate teleop`
3. Install the package: `make install-dev`

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#          Train           #
# ------------------------ #

demo:
	@python -m 

# ------------------------ #
#          Build           #
# ------------------------ #

install:
	@pip install --verbose -e .
.PHONY: install

h5py:
	@echo "Cloning and installing h5py"
	sudo apt-get update
	sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
	sudo apt-get install python3-pip
	sudo pip3 install -U pip testresources setuptools
	sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
	pip3 install Cython==0.29.36
	pip3 install pkgconfig
	git clone https://github.com/h5py/h5py.git
	git checkout 3.1.0
	git cherry-pick 3bf862daa4ebeb2eeaf3a0491e05f5415c1818e4
	H5PY_SETUP_REQUIRES=0 pip3 install . --no-deps --no-build-isolation
	cd h5py && source dev-install.sh
.PHONY: h5py

install-dependencies:
	@git submodule update --init --recursive
	@cd firmware/ && pip install --verbose -e .

build-ext:
	@python setup.py build_ext --inplace
.PHONY: build-ext

clean:
	rm -rf build dist *.so **/*.so **/*.pyi **/*.pyc **/*.pyd **/*.pyo **/__pycache__ *.egg-info .eggs/ .ruff_cache/
.PHONY: clean

# ------------------------ #
#       Static Checks      #
# ------------------------ #

format:
	@isort --profile black demo.py data_collection
	@black demo.py data_collection
	@ruff format demo.py data_collection
.PHONY: format

static-checks:
	@isort --profile black --check --diff .
	@black --diff --check .
	@ruff check .
.PHONY: lint
