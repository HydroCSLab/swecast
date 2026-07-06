"""
Fetch active SWE stations from the USDA AWDB REST API and export to CSV.
"""

import csv
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset as ncdataset

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


#Context: This function reaches out to a goverment weather website (the USDA AWDB API)
#Context: over the internet and downloads the list of all active snow stations.
#Context: It takes no inputs. It gives back a big list of stations, where each station
#Context: is a dictionary of facts like its name, latitude and longitude.
#Context: Example output: [{"name":"Mt Rose","latitude":39.3,...}, {"name":"Echo Peak",...}]
def fetch_stations():
    """
    Fetch all active SWE stations from the USDA AWDB API.
    """
    print("[swecast] Fetching SWE stations from USDA AWDB API...")
    with urllib.request.urlopen(API_URL) as response:
        stations = json.loads(response.read())
    print(f"[swecast] Fetched {len(stations)} stations.")
    return stations


#Context: This takes the list of stations (the dictionarys from fetch_stations) and
#Context: saves them into a CSV file, which is just a simple spreadsheet type text file.
#Context: It recieves the stations list and a file path to write to. It makes the
#Context: folder if it dosnt exist yet, then writes one row per station.
#Context: It dosnt return anything, it just creates the file on disk.
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


#Context: This is the smart helper that decides where to get the stations CSV from.
#Context: If the settings say to fetch fresh, or we have no saved copy, it downloads
#Context: new data and saves it. Otherwise it just reuses the cached file to save time.
#Context: It recieves the settings (manifest) and a folder, and returns the file path
#Context: to the CSV. Example output: output_dir/.cache/swe_stations.csv
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


#Context: A weather map is split into a grid of little square cells. This function
#Context: figures out which grid cell each station sits inside, based on its lat/lon.
#Context: Many stations can land in the same cell, so it removes the duplicates and
#Context: shifts the numbers so they line up with our smaller study area.
#Context: It recieves the stations CSV and a NetCDF map file, and saves the cell
#Context: coordinates to a .npy file. Example output: [[12,5],[12,6],[40,21]..etc]
def identify_station_cells(
    stations_csv, nc_path, output_path="station_cells.npy", bbox=None
):
    """
    Forklift of Identify_Station_Cells.py.

    Converts a station list (with replicates) to unique grid cells in the study
    region defined by ``bbox`` and writes the result to ``output_path``.

    The input CSV must have no header and columns matching the original
    ../../data/stations.csv layout: column 1 is latitude, column 2 is longitude.

    The corner offset that clips global NetCDF grid indices into local study-region
    indexing is derived from ``bbox`` via ``_nc_bbox_indices``. If ``bbox`` is None,
    falls back to (288, 75), the original California study-area constants from
    Identify_Station_Cells.py.
    """
    # Avoid an import cycle: nsidc imports stations for get_stations,
    # so we import _nc_bbox_indices lazily here.
    from .nsidc import _nc_bbox_indices

    # this part converts the 104 stations(with replicates) to 75 stations without
    # replicates,and returns the coordinates of this stations in the study region
    # defined by bbox

    stations = pd.read_csv(stations_csv, header=None).to_numpy()
    num_stations = stations.shape[0]

    # preprocess .nc file, extract SWE and DEPTH data
    ds = ncdataset(str(nc_path))
    swe = ds.variables["SWE"]
    depth = ds.variables["DEPTH"]

    # understand the region
    lat = ds.variables["lat"]
    lon = ds.variables["lon"]
    num_lats = len(lat)
    num_lons = len(lon)

    # Derive the corner offset (was hardcoded (288, 75) in the original)
    if bbox is not None:
        lat_lo, _, lon_lo, _ = _nc_bbox_indices(nc_path, bbox)
        corner_lat, corner_lon = lat_lo, lon_lo
    else:
        corner_lat, corner_lon = 288, 75

    lonlat = np.zeros((num_stations, 2))

    # XXX: consider 2D version of np.abs(lon - stations).argmin()
    for i in range(0, num_stations):
        for j in range(0, num_lons - 1):
            # stations[i, 2]: the longitude of station i
            # lon[j]: j'th longitude in NC
            # find the cell j or j + 1 that contains station i
            if (stations[i, 2] - lon[j]) * (lon[j + 1] - stations[i, 2]) > 0:
                # if station i is in cell j + 1
                if (stations[i, 2] - lon[j]) >= (lon[j + 1] - stations[i, 2]):
                    lonlat[i, 1] = j + 1
                    break
                else:  # else if station i is in cell j
                    lonlat[i, 1] = j
                    break
            elif lon[j + 1] - stations[i, 2] == 0:
                # XXX: why not just j?
                lonlat[i, 1] = j + 1
                break

        for j in range(0, num_lats - 1):
            if (stations[i, 1] - lat[j]) * (lat[j + 1] - stations[i, 1]) > 0:
                if (stations[i, 1] - lat[j]) >= (lat[j + 1] - stations[i, 1]):
                    lonlat[i, 0] = j + 1
                    break
                else:
                    lonlat[i, 0] = j
            elif lat[j + 1] - stations[i, 1] == 0:
                # XXX: why not just j?
                lonlat[i, 0] = j + 1
                break

    unique_lonlats = np.unique(lonlat, axis=0)
    num_unique_lonlats = len(unique_lonlats)

    latlon = np.zeros((num_unique_lonlats, 2))

    # clip the NC region to our study area
    for i in range(0, num_unique_lonlats):
        # (corner_lat, corner_lon) is one corner of our study area, derived from manifest.bbox
        latlon[i, 0] = lonlat[i, 0] - corner_lat
        latlon[i, 1] = lonlat[i, 1] - corner_lon

    # save the station data
    np.save(output_path, latlon)
    return output_path
