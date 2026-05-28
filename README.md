# SWECast (Snow Water Equivalent Forecasting)


> **Development Status**
> This is an early-stage development module and is **not yet functional**. APIs, behavior, and outputs will change.

---

## Installation

For now, install locally:
pip install -e .

* *(This will change once published to PyPI)*

---

## Authentication

`swecast` requires access to external data services. You must set your Earthdata credentials as environment variables before running the module:

```bash
export EARTHDATA_USERNAME='your_earthdata_username'
export EARTHDATA_PASSWORD='your_earthdata_password'
```

Without these, data downloads will fail.

---

## Quick Start

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

---

## Core Concepts

### Manifest

The `Manifest` object defines the scope of your data processing job:

* **start / end**: Date range for the dataset
* **bbox**: Geographic bounding box `(min_lon, min_lat, max_lon, max_lat)`

---

## Main Functions

### `build_stacks(manifest, output_dir)`

Builds general geospatial data stacks for the given manifest.

---

### `build_swe_stacks(manifest, output_dir)`

Builds Snow Water Equivalent (SWE) stacks.

---

### `fill_stacks(stacks)`

Performs gap-filling on SWE stacks to handle missing data.

---

### `fetch_stations(...)`

Fetches station metadata (usage still evolving).

---

### `stations_to_csv(...)`

Exports station data to CSV format.

---

## Output

All outputs are written to the specified `output_dir`. The structure and formats may change as the project evolves.

---

## Development Notes

* This project is **actively under development**
* Interfaces are **not yet stable**
