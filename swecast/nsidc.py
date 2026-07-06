"""
Download NSIDC-0719 SWE/Depth NetCDF files and build stacked GeoTIFFs.

Needs NASA Earthdata credentials. Set them on the manifest or via the
EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment variables.

Files are organized by water year:
  URL:  https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/
  Name: 4km_SWE_Depth_WY{yyyy}_v01.nc
  Vars: SWE (mm H2O), DEPTH (mm)
"""

import os
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import xarray as xr
import requests
import rasterio
from scipy.ndimage import convolve
from netCDF4 import Dataset as ncdataset
from netCDF4 import num2date
from .prism import Manifest, _date_range, _date_range_no_leap, _resolve
from .preflight import preflight_nsidc
from rasterio.transform import from_bounds
from scipy.ndimage import convolve
import math

_EARTHDATA_AUTH_HOST = (
    "urs.earthdata.nasa.gov"  # Thish might be a config option in the future.
)


class _EarthdataSession(requests.Session):
    """
    requests.Session that survives Earthdata's OAuth redirect chain.

    The DAAC bounces us to urs.earthdata.nasa.gov for OAuth. Keeping
    Basic Auth attached during that hop lets the login complete without
    a browser.
    """

    #Context: This sets up the web session when we first make one. It recieves a
    #Context: username and password and stores them so every request we send later
    #Context: will automatically log us in. No example output, it just saves the login.
    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = (username, password)

    #Context: When the website bounces us to a diffrent login page (a redirect), this
    #Context: decides if we keep sending our password along. If we are going to the
    #Context: Earthdata login host we keep it, otherwise we drop it for safety so we
    #Context: dont leak our password to some other website by accident.
    def rebuild_auth(self, prepared_request, response):
        # Only drop auth when redirecting away from both the origin and Earthdata hosts
        redirect_host = urlparse(prepared_request.url).hostname or ""
        if redirect_host == _EARTHDATA_AUTH_HOST:
            return  # keep credentials for the OAuth login step
        super().rebuild_auth(prepared_request, response)


NSIDC_BASE = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/"
    "nsidc0719_SWE_Snow_Depth_v1/4km_SWE_Depth_WY{wy}_v01.nc"
)
NC_VARIABLES = ("SWE", "DEPTH")

# Forklifted from Extract_SWE.py.
# Leap years are not considered, matching the original (num_days = 365 per WY).
# The original script used a hardcoded region slice [:, 288:430, 75:385] giving a
# 142x310 array; that was the California study bbox in the .nc grid. We now
# derive the slice indices from manifest.bbox at runtime instead.
num_days = 365

# .npy filename per NetCDF variable, matching Extract_SWE.py -> swe.npy
_NPY_NAME = {"SWE": "swe", "DEPTH": "depth"}


#Context: This takes a map box (a left,bottom,right,top corner area) and figures out
#Context: which rows and columns of the data grid fall inside it. So instead of real
#Context: world coordinates you get simple counting numbers to slice the array with.
#Context: Example output: (288, 430, 75, 385)  meaning lat rows 288-430, lon cols 75-385.
def _nc_bbox_indices(nc_path, bbox):
    """
    Convert (minx, miny, maxx, maxy) into integer lat/lon slice indices.

    Used by build_npy_swe_stacks for the SWE region slice and by
    identify_station_cells for the corner offset that maps global grid
    indices into the local region. Indices come back low-to-high so
    callers can use them directly: ``data[:, lat_lo:lat_hi, lon_lo:lon_hi]``.

    Size is computed as ``round((max - min) / cell_size)`` to match how
    _prism_bbox_indices derives PRISM's H/W from .hdr metadata. That's
    important: the two grids have to come out the same shape, otherwise
    np.stack downstream will blow up. Origin comes from searchsorted; hi
    is then origin + size.

    For the original California bbox (-121.9, 36.08, -109, 41.98) on the
    4km NSIDC grid this returns ~(288, 430, 75, 385), matching the
    hardcoded slice in Extract_SWE.py.
    """
    ds = ncdataset(str(nc_path))
    # lat_arr and lon_arr are centroids
    lat_arr = np.asarray(ds.variables["lat"][:])
    lon_arr = np.asarray(ds.variables["lon"][:])
    ds.close()

    dy = float(np.abs(lat_arr[1] - lat_arr[0]))
    dx = float(np.abs(lon_arr[1] - lon_arr[0]))

    # bbox has edges
    minx, miny, maxx, maxy = bbox

    lon_edge0 = lon_arr[0] - dx / 2

    if lat_arr[0] < lat_arr[-1]:  # increasing (south-to-north)
        lat_edge0 = lat_arr[0] - dy / 2
        lat_lo = int(max(0, math.floor((miny - lat_edge0) / dy)))
        lat_hi = int(min(len(lat_arr), math.ceil((maxy - lat_edge0) / dy)))
    else:  # decreasing (north-to-south)
        lat_edge0 = lat_arr[0] + dy / 2
        lat_lo = int(max(0, math.floor((lat_edge0 - maxy) / dy)))
        lat_hi = int(min(len(lat_arr), math.ceil((lat_edge0 - miny) / dy)))

    lon_lo = int(max(0, math.floor((minx - lon_edge0) / dx)))
    lon_hi = int(min(len(lon_arr), math.ceil((maxx - lon_edge0) / dx)))
    return lat_lo, lat_hi, lon_lo, lon_hi


#Context: A water year runs from Oct 1 to Sep 30, not Jan to Dec. This tells you which
#Context: water year a given date belongs too. If the month is October or later it
#Context: counts as next years water year.
#Context: Example: a date in Nov 2020 -> output 2021. A date in Mar 2020 -> output 2020.
def _water_year(d: date) -> int:
    """
    Water year (Oct 1 to Sep 30) for a date.
    """
    return d.year + 1 if d.month >= 10 else d.year


#Context: Given a start and end date, this gives back every water year that the range
#Context: covers, sorted and with no duplicates. Usefull for knowing which yearly
#Context: files we need to download.
#Context: Example: start in late 2019 thru mid 2021 -> output [2020, 2021, 2022]
def _water_years(start: date, end: date) -> list[int]:
    """
    Sorted list of water years that [start, end] touches.
    """
    wys = set()
    for d in _date_range(start, end):
        wys.add(_water_year(d))
    return sorted(wys)


# NetCDF4 (HDF5) magic bytes
_NC_MAGIC = b"\x89HDF\r\n\x1a\n"


#Context: This checks if a file is a real NetCDF file or just junk. It peeks at the
#Context: first 8 bytes and compares them to the known magic signature these files
#Context: always start with. Helps us catch a download that got corrupted.
#Context: Example output: True if its a good file, False if not.
def _is_valid_nc(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == _NC_MAGIC
    except OSError:
        return False


#Context: This downloads the data file for one water year from NASA, but first it
#Context: checks if we already have a good copy saved in the cache folder so we dont
#Context: download it twice. If the cached one is broken it deletes it and tries again.
#Context: It returns the path to the file on disk once its ready.
def _download_nc(wy: int, cache_dir: Path, username: str, password: str) -> Path:
    dest = cache_dir / f"4km_SWE_Depth_WY{wy}_v01.nc"
    if dest.exists():
        if _is_valid_nc(dest):
            print(f"[swecast] SWE WY{wy}: using cached {dest}")
            return dest
        print(f"[swecast] SWE WY{wy}: cached file is corrupt, re-downloading")
        dest.unlink()

    url = NSIDC_BASE.format(wy=wy)
    print(f"[swecast] SWE WY{wy}: downloading {url}")
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


#Context: This trims a big dataset down to just the area inside the box you give it
#Context: (left, bottom, right, top). It uses the lat/lon labels to cut out only the
#Context: part of the map we care about, like cropping a photo. It has to flip the
#Context: cut direction depending on wheter latitude goes up or down in the data.
def _bbox_slice(ds: xr.Dataset, bbox: tuple) -> xr.Dataset:
    """
    Clip ``ds`` to bbox (minx, miny, maxx, maxy) using its lat/lon coords.
    """
    minx, miny, maxx, maxy = bbox
    lat = ds["lat"].values
    lat_slice = slice(miny, maxy) if lat[0] < lat[-1] else slice(maxy, miny)
    return ds.sel(lon=slice(minx, maxx), lat=lat_slice)


#Context: This is the big main job. It downloads the snow data for each day, crops it
#Context: to our area, and saves it all stacked together into GeoTIFF image files (one
#Context: layer/band per day). It needs your NASA login in the environment variables.
#Context: It returns a little dictionary telling you where each output file landed,
#Context: for example {"SWE": path, "DEPTH": path}.
def build_swe_stacks(
    manifest: Manifest,
    output_dir: Path,
    cache_dir: Path | None = None,
    write_npy: bool | None = None,
    variables: tuple | None = None,
) -> dict[str, Path]:
    """
    Download NSIDC-0719 SWE/DEPTH per day, clip to bbox, write band-per-day GeoTIFFs.

    Needs $EARTHDATA_USERNAME and $EARTHDATA_PASSWORD set.

    With ``write_npy=True`` (default) we also drop the script-faithful
    swe.npy stack via build_npy_swe_stacks, matching Extract_SWE.py.

    ``cache_dir``, ``write_npy``, and ``variables`` (mapped to
    ``manifest.nc_variables``) fall back to the manifest if not passed.

    Returns a dict mapping variable name to output path,
    e.g. {"SWE": ..., "DEPTH": ...}.
    """
    cache_dir = _resolve(cache_dir, manifest, "cache_dir", None)
    write_npy = _resolve(write_npy, manifest, "write_npy", True)
    variables = _resolve(variables, manifest, "nc_variables", NC_VARIABLES)

    preflight_nsidc()
    username = os.environ["EARTHDATA_USERNAME"]
    password = os.environ["EARTHDATA_PASSWORD"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".cache"

    dates = list(_date_range_no_leap(manifest.start, manifest.end))
    wys = _water_years(manifest.start, manifest.end)

    # Download and open all needed water-year files
    nc_by_wy: dict[int, xr.Dataset] = {}
    for wy in wys:
        nc_path = _download_nc(wy, cache_dir, username, password)
        nc_by_wy[wy] = xr.open_dataset(nc_path, engine="netcdf4")

    outputs = {}
    for variable in variables:
        arrays = []
        profile = None

        for d in dates:
            ds = nc_by_wy[_water_year(d)]

            da = ds[variable].sel(time=np.datetime64(d))
            clipped = _bbox_slice(da.to_dataset(name=variable), manifest.bbox)[variable]

            if profile is None:

                lons = clipped.lon.values
                lats = clipped.lat.values
                transform = from_bounds(
                    lons.min(),
                    lats.min(),
                    lons.max(),
                    lats.max(),
                    len(lons),
                    len(lats),
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

        out_path = output_dir / f"{variable.lower()}_stack.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, (arr, d) in enumerate(zip(arrays, dates), start=1):
                dst.write(arr, i)
                dst.update_tags(i, date=d.isoformat())

        outputs[variable] = out_path
        print(f"[swecast] {variable}: wrote {len(dates)}-band stack -> {out_path}")

    # Clean up open datasets
    for ds in nc_by_wy.values():
        ds.close()

    if write_npy:
        build_npy_swe_stacks(manifest, output_dir, cache_dir=cache_dir)

    return outputs


#Context: Simlar to the function above but it saves the data as plain numpy .npy files
#Context: instead of GeoTIFF images. It walks every calendar day (skipping Feb 29 since
#Context: the data has no leap days) and stacks them into a 3D block shaped like
#Context: (number of days, height, width). This keeps the days lined up with the PRISM
#Context: data so they match row for row. Returns where each .npy file was saved.
def build_npy_swe_stacks(
    manifest: Manifest,
    output_dir: Path,
    cache_dir: Path | None = None,
    variables: tuple | None = None,
) -> dict[str, Path]:
    """
    Forklift of Extract_SWE.py, retimed so the time axis lines up with PRISM.

    Walks calendar dates from manifest.start to manifest.end (skipping
    Feb 29 since NSIDC-0719 has no leap days) and writes (T, H, W) stacks:
        swe.npy   (always)
        depth.npy (only if "DEPTH" is in ``variables``)

    Row i corresponds to the same calendar date as row i in pcp.npy /
    tmp.npy from build_npy_stacks, since both walk _date_range_no_leap.
    The original Extract_SWE.py iterated water years starting Oct 1 of
    (start_year - 1); the underlying NSIDC convention is unchanged, but
    the .npy time axis now starts at manifest.start so PRISM lines up.

    ``cache_dir`` and ``variables`` (mapped to ``manifest.npy_nc_variables``)
    fall back to the manifest if not passed.
    """
    cache_dir = _resolve(cache_dir, manifest, "cache_dir", None)
    variables = _resolve(variables, manifest, "npy_nc_variables", ("SWE",))

    preflight_nsidc()
    username = os.environ["EARTHDATA_USERNAME"]
    password = os.environ["EARTHDATA_PASSWORD"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".cache"

    # Iterate calendar dates, skipping Feb 29 (NSIDC-0719 doesn't have it).
    # This is the same iteration build_npy_stacks (PRISM) uses, so the time
    # axes align row-for-row across all .npy stacks.
    dates = list(_date_range_no_leap(manifest.start, manifest.end))
    needed_wys = sorted({_water_year(d) for d in dates})

    # Download / locate every water-year .nc the date range touches
    nc_paths = {
        wy: _download_nc(wy, cache_dir, username, password) for wy in needed_wys
    }

    # Region slice from manifest.bbox (replaces the original hardcoded
    # [:, 288:430, 75:385] in Extract_SWE.py).
    sample_nc = nc_paths[needed_wys[0]]
    lat_lo, lat_hi, lon_lo, lon_hi = _nc_bbox_indices(sample_nc, manifest.bbox)
    height = lat_hi - lat_lo
    width = lon_hi - lon_lo
    print(
        f"[swecast] SWE region: lat[{lat_lo}:{lat_hi}], lon[{lon_lo}:{lon_hi}] -> ({height}, {width})"
    )

    # Build a (year, month, day) -> time-axis index map for each .nc, so we
    # can look up any calendar date in O(1).
    day_indices = {}
    for wy, nc_path in nc_paths.items():
        with ncdataset(str(nc_path)) as ds:
            times = ds.variables["time"]
            date_objs = num2date(
                times[:], times.units, getattr(times, "calendar", "standard")
            )
        day_indices[wy] = {(d.year, d.month, d.day): i for i, d in enumerate(date_objs)}

    outputs = {}
    for variable in variables:
        # XXX: if your computer's capability is limited, don't extract ds_swe and ds2
        # simultanuously
        ds_swe = np.zeros((len(dates), height, width))
        nc_handles = {wy: ncdataset(str(p)) for wy, p in nc_paths.items()}
        try:
            for p, d in enumerate(dates):  # p = running data-row counter
                wy = _water_year(d)
                key = (d.year, d.month, d.day)
                if key not in day_indices[wy]:
                    raise ValueError(
                        f"Date {d.isoformat()} not present in {nc_paths[wy]}"
                    )
                day_idx = day_indices[wy][key]
                arr = nc_handles[wy].variables[variable][
                    day_idx, lat_lo:lat_hi, lon_lo:lon_hi
                ]
                # NSIDC variables come back as masked arrays with a fill_value
                # like -999.0; convert masked elements to NaN so filled_data
                # can gap-fill them downstream (it checks np.isnan, not the
                # fill_value).
                ds_swe[p] = np.ma.filled(np.ma.asarray(arr).astype(np.float64), np.nan)
        finally:
            for h in nc_handles.values():
                h.close()
        ds_swe = np.array(ds_swe)
        # ds_swe.shape: (len(dates), height, width)
        out_path = output_dir / f"{_NPY_NAME[variable]}.npy"
        np.save(out_path, ds_swe)
        outputs[variable] = out_path
        print(f"[swecast] {variable}: wrote {len(dates)}-day .npy stack -> {out_path}")

    return outputs


#Context: Sometimes the data has holes in it (NaN means "no value here"). This fills
#Context: those holes by looking at the neigboring cells around them in a little 3x3x3
#Context: cube and taking the average. Cells that already have a value are left alone.
#Context: Example: an array with some NaN holes goes in, the same array comes back out
#Context: but with the holes filled by the average of thier neighbors.
def filled_data(data):
    # this function fill nan data using kernel of size 3x3x3. It averages data from kernel and fill it up
    kernel = (
        np.ones((3, 3, 3)) / 27
    )  # kernel is 3d and it will give equal weightage to each cell
    data_filled = data.copy()
    nan_mask = np.isnan(data_filled)
    # Convolve while keeping NaN values outside the kernel
    data_convolved = convolve(np.nan_to_num(data), kernel, mode="constant", cval=0)

    # Normalize by the valid neighbor count
    neighbor_count = convolve(~nan_mask, kernel, mode="constant", cval=0)
    neighbor_count[neighbor_count == 0] = 1  # Avoid division by zero

    data_filled[nan_mask] = data_convolved[nan_mask] / neighbor_count[nan_mask]

    return data_filled


#Context: This loads a saved .npy data file, fills in its missing holes using the
#Context: filled_data helper above, and then saves a new copy next to the orignal with
#Context: "_filled" added to the name. The first file is left untouched.
#Context: Example: input "swe.npy" -> output a new file called "swe_filled.npy".
def fill_npy(file_path):
    """
    Forklift of Fill_Missing_Data.py.

    Loads a .npy stack, gap-fills NaNs with a 3x3x3 kernel, and writes
    the result to ``{stem}_filled{ext}`` next to the input.
    """
    file_path = str(file_path)
    data = np.load(file_path)
    data_filled = filled_data(data)
    filename, ext = os.path.splitext(
        os.path.basename(file_path)
    )  # Extract the filename without extension

    # Create the new filename
    new_filename = f"{filename}_filled{ext}"
    out_path = os.path.join(os.path.dirname(file_path), new_filename)

    np.save(out_path, data_filled)
    return out_path


#Context: This opens a multi layer GeoTIFF image file and reads it into a numbers array
#Context: shaped (bands, rows, cols). It also turns any "nodata" placeholder values into
#Context: NaN holes so they can be filled later. It hands back both the numbers and the
#Context: profile (the map info like size and location).
def _read_stack(tif_path):
    """
    Read a multi-band GeoTIFF as (bands, rows, cols) float + rasterio profile.
    """
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
    nodata = profile.get("nodata")
    if nodata is not None and not np.isnan(nodata):
        data[data == nodata] = np.nan
    return data, profile


#Context: This is the opposite of _read_stack. It takes a 3D block of numbers and saves
#Context: it back out as a multi layer GeoTIFF image, reusing the same map info (profile)
#Context: so the new file knows where it sits on the globe. Nothing is returned, it just
#Context: writes the file to disk.
def _write_stack(out_path, data, profile):
    """
    Write a 3D array as a multi-band GeoTIFF, keeping the input's georeferencing.
    """
    profile.update(dtype="float32", count=data.shape[0], nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


#Context: This goes through several GeoTIFF stacks (like SWE and DEPTH), reads each one,
#Context: fills its holes, and writes out a new "_filled" version. If theres a matching
#Context: .npy file sitting next to it, it fills that one too so everthing stays in sync.
#Context: It returns a dictionary mapping each variable name to where its filled file went.
def fill_stacks(stack_paths, output_dir=None, suffix="_filled"):
    """
    Fill NaN values in raster stacks (SWE, DEPTH, ...).

    Parameters
    ----------
    stack_paths : dict
        Variable name to input GeoTIFF path, e.g.
        {"SWE": "swe.tif", "DEPTH": "depth.tif"}.
    output_dir : str | Path, optional
        Where to write outputs. Defaults to each input's own parent dir.
    suffix : str
        Appended to the output filename, before the extension.

    Returns
    -------
    dict
        Variable name to written output path.
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

        # If a sibling .npy stack exists (from build_npy_swe_stacks), fill it too
        # so downstream training has e.g. swe_filled.npy alongside the GeoTIFF.
        sibling_npy = in_path.parent / f"{_NPY_NAME.get(var, var.lower())}.npy"
        if sibling_npy.exists():
            fill_npy(sibling_npy)

    return outputs
