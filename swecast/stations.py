"""
Fetch active SWE stations from the USDA AWDB REST API and export to CSV.
"""

import csv
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .nsidc import _GRID_RES, swe_grid
from .prism import Manifest

API_URL = (
    "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations"
    "?elements=WTEQ"
    "&returnForecastPointMetadata=false"
    "&returnReservoirMetadata=false"
    "&returnStationElements=false"
    "&activeOnly=true"
)

FIELDS = [
    "stationTriplet",
    "stationId",
    "stateCode",
    "networkCode",
    "name",
    "dcoCode",
    "countyName",
    "huc",
    "elevation",
    "latitude",
    "longitude",
    "beginDate",
    "endDate",
    "associatedHucs",
]


def fetch_stations():
    """
    Fetch all active SWE stations from the USDA AWDB API.
    """
    print("[swecast] Fetching SWE stations from USDA AWDB API...")
    with urllib.request.urlopen(API_URL) as response:
        stations = json.loads(response.read())
    print(f"[swecast] Fetched {len(stations)} stations.")
    return stations


def stations_to_csv(stations, output_path):
    """
    Write a list of station dicts to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for station in stations:
            row = dict(station)
            if "associatedHucs" in row and isinstance(row["associatedHucs"], list):
                row["associatedHucs"] = "|".join(row["associatedHucs"])
            writer.writerow(row)


def get_stations(manifest: Manifest, output_dir: Path) -> Path:
    """
    Resolve the stations CSV, fetching from the API if needed.

    - If manifest.fetch_stations is True, always fetch fresh data.
    - Otherwise, use the cached CSV if it exists, or fetch if it doesn't.

    Returns the path to the stations CSV.
    """
    output_dir = Path(output_dir)
    cache_path = output_dir / ".cache" / "swe_stations.csv"

    if manifest.fetch_stations:
        stations = fetch_stations()
        stations_to_csv(stations, cache_path)
    elif cache_path.exists():
        print(f"[swecast] Using cached stations: {cache_path}")
    else:
        stations = fetch_stations()
        stations_to_csv(stations, cache_path)

    return cache_path


def identify_station_cells(stations_csv, bbox, output_path="station_cells.npy"):
    """
    Forklift of Identify_Station_Cells.py.

    Converts a station list (with replicates) to unique grid cells in the study
    region defined by ``bbox`` and writes the result to ``output_path``.

    The input CSV must have no header and columns matching the original
    ../../data/stations.csv layout: column 1 is latitude, column 2 is longitude.

    Cells are (row, col) indices into the bbox window of the 4km grid, with
    rows running south-to-north to match the .npy stacks.
    """
    # this part converts the 104 stations(with replicates) to 75 stations without
    # replicates,and returns the coordinates of this stations in the study region
    # defined by bbox

    stations = pd.read_csv(stations_csv, header=None).to_numpy()
    lat, lon = swe_grid(bbox)
    south_edge = lat[0] - _GRID_RES / 2
    west_edge = lon[0] - _GRID_RES / 2

    # the grid is regular, so the containing cell is just floor(offset / res)
    lonlat = np.zeros((stations.shape[0], 2))
    lonlat[:, 0] = np.floor((stations[:, 1].astype(float) - south_edge) / _GRID_RES)
    lonlat[:, 1] = np.floor((stations[:, 2].astype(float) - west_edge) / _GRID_RES)

    latlon = np.unique(lonlat, axis=0)

    # save the station data
    np.save(output_path, latlon)
    return output_path
