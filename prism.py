"""Download PRISM daily ppt/tmean BIL files and build stacked GeoTIFFs."""

import io
import zipfile
from .preflight import preflight_prism
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import requests

PRISM_BASE = "https://services.nacse.org/prism/data/get/us/4km/{variable}/{date}?format=bil"
VARIABLES = ("ppt", "tmean")


@dataclass
class Manifest:
    start: date
    end: date
    bbox: tuple  # (minx, miny, maxx, maxy) in WGS84
    fetch_stations: bool = False


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _download_bil(variable: str, d: date, cache_dir: Path) -> Path:
    date_str = d.strftime("%Y%m%d")
    bil_path = cache_dir / variable / f"{date_str}.bil"

    if bil_path.exists():
        print(f"[sweforecast] {variable} {d}: using cached {bil_path}")
        return bil_path

    url = PRISM_BASE.format(variable=variable, date=date_str)
    print(f"[sweforecast] {variable} {d}: downloading {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    if resp.content[:4] != b"PK\x03\x04":
        preview = resp.content[:500].decode("utf-8", errors="replace")
        raise ValueError(
            f"Expected zip for {variable} {d}.\nResponse preview:\n{preview}"
        )

    bil_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for member in zf.namelist():
            suffix = Path(member).suffix
            zf.extract(member, bil_path.parent)
            if suffix == ".bil":
                extracted = bil_path.parent / member
                extracted.rename(bil_path)
            elif suffix in (".hdr", ".prj", ".stx", ".aux"):
                extracted = bil_path.parent / member
                extracted.rename(bil_path.with_suffix(suffix))

    if not bil_path.exists():
        raise FileNotFoundError(f"No .bil file in archive for {variable} {d}")
    return bil_path


def build_stacks(
    manifest: Manifest,
    output_dir: Path,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Download PRISM ppt and tmean for each day in manifest, clip to bbox,
    and write a band-per-day stacked GeoTIFF for each variable.

    Returns a dict mapping variable name -> output path.
    """
    from .stations import get_stations

    preflight_prism()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".cache"

    get_stations(manifest, output_dir)

    geom = [box(*manifest.bbox).__geo_interface__]
    dates = list(_date_range(manifest.start, manifest.end))
    outputs = {}

    for variable in VARIABLES:
        arrays: list[np.ndarray] = []
        profile = None

        for d in dates:
            bil = _download_bil(variable, d, cache_dir)
            with rasterio.open(bil) as src:
                    clipped, transform = mask(src, geom, crop=True, nodata=src.nodata)
                    if profile is None:
                        profile = src.profile.copy()
                        profile.update(
                            driver="GTiff",
                            count=len(dates),
                            transform=transform,
                            height=clipped.shape[1],
                            width=clipped.shape[2],
                            compress="lzw",
                        )
            arrays.append(clipped[0])

        out_path = output_dir / f"{variable}_stack.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, (arr, d) in enumerate(zip(arrays, dates), start=1):
                dst.write(arr, i)
                dst.update_tags(i, date=d.isoformat())

        outputs[variable] = out_path
        print(f"[sweforecast] {variable}: wrote {len(dates)}-band stack -> {out_path}")

    return outputs
