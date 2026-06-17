# SWECast (Snow Water Equivalent Forecasting)

> **Development Status**
> This is an early-stage development module and is **not yet functional**. APIs, behavior, and outputs will change.

<!-- vim-markdown-toc GFM -->

* [Micromamba environment](#micromamba-environment)
  * [PyPI TensorFlow](#pypi-tensorflow)
    * [Creating a new micromamba environment](#creating-a-new-micromamba-environment)
    * [Using an existing micromamba environment](#using-an-existing-micromamba-environment)
  * [Micromamba TensorFlow (not working)](#micromamba-tensorflow-not-working)
* [Installation](#installation)
* [Authentication](#authentication)
* [Quick start](#quick-start)
* [Core concepts](#core-concepts)
  * [Manifest](#manifest)
* [Main functions](#main-functions)
  * [`build_stacks(manifest, output_dir)`](#build_stacksmanifest-output_dir)
  * [`build_swe_stacks(manifest, output_dir)`](#build_swe_stacksmanifest-output_dir)
  * [`fill_stacks(stacks)`](#fill_stacksstacks)
  * [`fetch_stations(...)`](#fetch_stations)
  * [`stations_to_csv(...)`](#stations_to_csv)
* [Output](#output)
* [Development notes](#development-notes)

<!-- vim-markdown-toc -->

## Micromamba environment

### PyPI TensorFlow

#### Creating a new micromamba environment

```bash
# install micromamba
curl -L https://micro.mamba.pm/install.sh | env \
  BIN_FOLDER="$HOME/local/bin" \
  PREFIX_LOCATION="$HOME/opt/micromamba" \
  sh

# create an alias
echo "alias mm=micromamba" >> ~/.bashrc

# source micromamba
. ~/.bashrc

mkdir -p ~/work/projects/swecast
cd ~/work/projects/swecast
git clone git@github.com:hydrocslab/swecast.git

# the latest tensorflow supports python 3.13
mm create -p ./env -y python=3.13

# add cuda lib paths to LD_LIBRARY_PATH automatically when activating the
# environment
mkdir -p env/etc/conda/activate.d
cat > env/etc/conda/activate.d/tensorflow-cuda.sh <<'EOF'
export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH="$(python - <<'PY'
import site
from pathlib import Path

for sp in site.getsitepackages():
    nvidia = Path(sp) / "nvidia"
    if nvidia.exists():
        for lib in nvidia.glob("*/lib"):
            print(lib, end=":")
PY
)$LD_LIBRARY_PATH"
EOF

mkdir -p env/etc/conda/deactivate.d
cat > env/etc/conda/deactivate.d/tensorflow-cuda.sh <<'EOF'
export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
EOF

mm activate ./env

pip install tensorflow[and-cuda] rasterio shapely xarray scipy netCDF4 matplotlib rioxarray

. env/etc/conda/activate.d/tensorflow-cuda.sh

# make sure GPU is recognized
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

mkdir -p runs/example
cd runs/example
python ../../swecast/example.py

mm deactivate
```

#### Using an existing micromamba environment

```bash
cd ~/work/projects/swecast
mm activate ./env

mkdir -p runs/example
cd runs/example
python ../../swecast/example.py

mm deactivate
```

### Micromamba TensorFlow (not working)

```bash
# install micromamba
curl -L https://micro.mamba.pm/install.sh | env \
  BIN_FOLDER="$HOME/local/bin" \
  PREFIX_LOCATION="$HOME/opt/micromamba" \
  sh

# create an alias
echo "alias mm=micromamba" >> ~/.bashrc

# source micromamba
. ~/.bashrc

mkdir -p ~/work/projects/swecast
cd ~/work/projects/swecast
git clone git@github.com:hydrocslab/swecast.git

# install Python through tensorflow because tensorflow is usually a couple
# versions behind Python in conda-forge
mm create -p ./env -y tensorflow rasterio shapely xarray scipy netCDF4 matplotlib rioxarray
mm activate ./env

# executable stack needed to be cleared on Slackware
patchelf --clear-execstack $CONDA_PREFIX/lib/libtensorflow_cc.so.2

# make sure GPU is recognized
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

mkdir -p runs/example
cd runs/example
python ../../swecast/example.py

# after all this, it still didn't work
```

## Installation

For now, install locally:
pip install -e .

* *(This will change once published to PyPI)*

## Authentication

`swecast` requires access to external data services. You must set your Earthdata credentials as environment variables before running the module:

```bash
export EARTHDATA_USERNAME='your_earthdata_username'
export EARTHDATA_PASSWORD='your_earthdata_password'
```

Without these, data downloads will fail.

## Quick start

```python
import swecast
from datetime import date
from swecast import Manifest, build_stacks, build_swe_stacks
from swecast import fetch_stations, stations_to_csv, fill_stacks

manifest = Manifest(
    start=date(2001, 1, 1),
    end=date(2002, 1, 1),
    bbox=(-121.9, 36.08, -109, 41.98),
)

# Build general data stacks
outputs = build_stacks(manifest, output_dir="./output")

# Build SWE stacks
swe_outputs = build_swe_stacks(manifest, output_dir="./output")

# Gap-fill SWE stacks
fill_stacks(swe_outputs)
```

## Core concepts

### Manifest

The `Manifest` object defines the scope of your data processing job:

* **start / end**: Date range for the dataset
* **bbox**: Geographic bounding box `(min_lon, min_lat, max_lon, max_lat)`

## Main functions

### `build_stacks(manifest, output_dir)`

Builds general geospatial data stacks for the given manifest.

### `build_swe_stacks(manifest, output_dir)`

Builds Snow Water Equivalent (SWE) stacks.

### `fill_stacks(stacks)`

Performs gap-filling on SWE stacks to handle missing data.

### `fetch_stations(...)`

Fetches station metadata (usage still evolving).

### `stations_to_csv(...)`

Exports station data to CSV format.

## Output

All outputs are written to the specified `output_dir`. The structure and formats may change as the project evolves.

## Development notes

* This project is **actively under development**
* Interfaces are **not yet stable**
