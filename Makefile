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

install: h5py
	@pip install --verbose -e .
.PHONY: install

h5py:
	@echo "Cloning and installing h5py"
	@git clone https://github.com/h5py/h5py.git
	@cd h5py && \
		git checkout 3.1.0 && \
		git cherry-pick 3bf862daa4ebeb2eeaf3a0491e05f5415c1818e4 && \
		source dev-install.sh
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
