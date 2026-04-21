"""Fetch active SWE stations from the USDA AWDB REST API and export to CSV."""

import csv
import json
import urllib.request
from pathlib import Path

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
    """Fetch all active SWE stations from the USDA AWDB API."""
    print("[sweforecast] Fetching SWE stations from USDA AWDB API...")
    with urllib.request.urlopen(API_URL) as response:
        stations = json.loads(response.read())
    print(f"[sweforecast] Fetched {len(stations)} stations.")
    return stations


def stations_to_csv(stations, output_path):
    """Write a list of station dicts to CSV."""
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
    """Resolve the stations CSV, fetching from the API if needed.

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
        print(f"[sweforecast] Using cached stations: {cache_path}")
    else:
        stations = fetch_stations()
        stations_to_csv(stations, cache_path)

    return cache_path
