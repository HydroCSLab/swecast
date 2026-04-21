"""Download NSIDC-0719 SWE/Depth NetCDF files and build stacked GeoTIFFs.

Requires NASA Earthdata credentials (username + password).
Credentials can be set via manifest fields or environment variables
EARTHDATA_USERNAME / EARTHDATA_PASSWORD.

Files are organized by water year (Oct 1 – Sep 30).
  URL: https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/
  Name: 4km_SWE_Depth_WY{yyyy}_v01.nc
  Variables: SWE (mm H2O), DEPTH (mm)
"""

import os
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rioxarray  # noqa: F401  — registers .rio accessor on xarray
import xarray as xr
import requests
import rasterio
from scipy.ndimage import convolve

_EARTHDATA_AUTH_HOST = "urs.earthdata.nasa.gov"


class _EarthdataSession(requests.Session):
    """requests.Session that keeps Basic Auth credentials through Earthdata OAuth redirects.

    The DAAC redirects to urs.earthdata.nasa.gov for OAuth; by sending Basic Auth
    credentials there, the redirect loop completes without a browser.
    """

    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        # Only drop auth when redirecting away from both the origin and Earthdata hosts
        redirect_host = urlparse(prepared_request.url).hostname or ""
        if redirect_host == _EARTHDATA_AUTH_HOST:
            return  # keep credentials for the OAuth login step
        super().rebuild_auth(prepared_request, response)

from .prism import Manifest, _date_range
from .preflight import preflight_nsidc

NSIDC_BASE = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/"
    "nsidc0719_SWE_Snow_Depth_v1/4km_SWE_Depth_WY{wy}_v01.nc"
)
NC_VARIABLES = ("SWE", "DEPTH")


def _water_year(d: date) -> int:
    """Return the water year (Oct 1 – Sep 30) for a given date."""
    return d.year + 1 if d.month >= 10 else d.year


def _water_years(start: date, end: date) -> list[int]:
    """Return sorted list of water years spanned by [start, end]."""
    wys = set()
    for d in _date_range(start, end):
        wys.add(_water_year(d))
    return sorted(wys)


# NetCDF4 (HDF5) magic bytes
_NC_MAGIC = b"\x89HDF\r\n\x1a\n"


def _is_valid_nc(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == _NC_MAGIC
    except OSError:
        return False


def _download_nc(wy: int, cache_dir: Path, username: str, password: str) -> Path:
    dest = cache_dir / f"4km_SWE_Depth_WY{wy}_v01.nc"
    if dest.exists():
        if _is_valid_nc(dest):
            print(f"[sweforecast] SWE WY{wy}: using cached {dest}")
            return dest
        print(f"[sweforecast] SWE WY{wy}: cached file is corrupt, re-downloading")
        dest.unlink()

    url = NSIDC_BASE.format(wy=wy)
    print(f"[sweforecast] SWE WY{wy}: downloading {url}")
    session = _EarthdataSession(username, password)
    with session.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        # Check magic bytes from first chunk before committing to disk
        first_chunk = next(resp.iter_content(chunk_size=8), b"")
        if first_chunk[:8] != _NC_MAGIC:
            preview = first_chunk.decode("utf-8", errors="replace")
            raise ValueError(
                f"Expected NetCDF4 for WY{wy}, got unexpected content.\n"
                f"Response preview: {preview!r}"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(first_chunk)
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def _bbox_slice(ds: xr.Dataset, bbox: tuple) -> xr.Dataset:
    """Clip dataset to bbox (minx, miny, maxx, maxy) using lat/lon coords."""
    minx, miny, maxx, maxy = bbox
    lat = ds["lat"].values
    lat_slice = slice(miny, maxy) if lat[0] < lat[-1] else slice(maxy, miny)
    return ds.sel(lon=slice(minx, maxx), lat=lat_slice)


def build_swe_stacks(
    manifest: Manifest,
    output_dir: Path,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Download NSIDC-0719 SWE and DEPTH for each day in manifest, clip to bbox,
    and write a band-per-day stacked GeoTIFF for each variable.

    Requires $EARTHDATA_USERNAME and $EARTHDATA_PASSWORD to be set.

    Returns:
        dict mapping variable name -> output path  {"SWE": ..., "DEPTH": ...}
    """
    from .stations import get_stations

    preflight_nsidc()
    username = os.environ["EARTHDATA_USERNAME"]
    password = os.environ["EARTHDATA_PASSWORD"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".cache"

    get_stations(manifest, output_dir)

    dates = list(_date_range(manifest.start, manifest.end))
    wys = _water_years(manifest.start, manifest.end)

    # Download and open all needed water-year files
    nc_by_wy: dict[int, xr.Dataset] = {}
    for wy in wys:
        nc_path = _download_nc(wy, cache_dir, username, password)
        nc_by_wy[wy] = xr.open_dataset(nc_path, engine="netcdf4")

    outputs = {}
    for variable in NC_VARIABLES:
        arrays = []
        profile = None

        for d in dates:
            ds = nc_by_wy[_water_year(d)]

            da = ds[variable].sel(time=np.datetime64(d))
            clipped = _bbox_slice(da.to_dataset(name=variable), manifest.bbox)[variable]

            if profile is None:
                from rasterio.transform import from_bounds
                lons = clipped.lon.values
                lats = clipped.lat.values
                transform = from_bounds(
                    lons.min(), lats.min(), lons.max(), lats.max(),
                    len(lons), len(lats),
                )
                profile = {
                    "driver": "GTiff",
                    "dtype": str(clipped.values.dtype),
                    "width": len(lons),
                    "height": len(lats),
                    "count": len(dates),
                    "crs": "EPSG:4269",
                    "transform": transform,
                    "compress": "lzw",
                    "nodata": ds[variable].attrs.get("_FillValue", None),
                }

            arrays.append(clipped.values)

        import rasterio
        out_path = output_dir / f"{variable.lower()}_stack.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, (arr, d) in enumerate(zip(arrays, dates), start=1):
                dst.write(arr, i)
                dst.update_tags(i, date=d.isoformat())

        outputs[variable] = out_path
        print(f"[sweforecast] {variable}: wrote {len(dates)}-band stack -> {out_path}")

    # Clean up open datasets
    for ds in nc_by_wy.values():
        ds.close()

    return outputs


import numpy as np
import rasterio
from scipy.ndimage import convolve


def filled_data(data):
    """Fill NaNs in a 3D array by averaging valid neighbors in a 3x3x3 kernel."""
    kernel = np.ones((3, 3, 3)) / 27
    data_filled = data.copy()
    nan_mask = np.isnan(data_filled)

    data_convolved = convolve(np.nan_to_num(data), kernel, mode="constant", cval=0)
    neighbor_count = convolve((~nan_mask).astype(float), kernel, mode="constant", cval=0)
    neighbor_count[neighbor_count == 0] = 1  # avoid div-by-zero

    data_filled[nan_mask] = data_convolved[nan_mask] / neighbor_count[nan_mask]
    return data_filled


def _read_stack(tif_path):
    """Read a multi-band GeoTIFF as (bands, rows, cols) float array + profile."""
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
    nodata = profile.get("nodata")
    if nodata is not None and not np.isnan(nodata):
        data[data == nodata] = np.nan
    return data, profile


def _write_stack(out_path, data, profile):
    """Write a 3D array to a multi-band GeoTIFF, preserving georeferencing."""
    profile.update(dtype="float32", count=data.shape[0], nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


def fill_stacks(stack_paths, output_dir=None, suffix="_filled"):
    """Fill NaN values in SWE and DEPTH raster stacks.

    Parameters
    ----------
    stack_paths : dict[str, str | Path]
        Mapping of variable name to input GeoTIFF path, e.g.
        {"SWE": "swe.tif", "DEPTH": "depth.tif"}.
    output_dir : str | Path, optional
        Directory for outputs. Defaults to each input's parent directory.
    suffix : str
        Suffix appended to output filenames (before the extension).

    Returns
    -------
    dict[str, str]
        Mapping of variable name to the written output path.
    """
    outputs = {}
    for var, in_path in stack_paths.items():
        in_path = Path(in_path)
        data, profile = _read_stack(in_path)
        filled = filled_data(data)

        out_dir = Path(output_dir) if output_dir else in_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{in_path.stem}{suffix}{in_path.suffix}"

        _write_stack(out_path, filled, profile)
        outputs[var] = str(out_path)
    return outputs


