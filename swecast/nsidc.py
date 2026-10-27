"""
Download SNODAS SWE/Depth grids and build stacked GeoTIFFs on the PRISM 4km grid.

SNODAS (NOAA NOHRSC) is a ~1km (30 arc-second) daily snow model for the
conterminous US, archived by NSIDC as G02158. No credentials are needed.

Files are organized by day:
  URL:  https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{yyyy}/{mm_Mon}/
  Name: SNODAS_{yyyymmdd}.tar
  Vars: SWE (product 1034, mm H2O), DEPTH (product 1036, mm)

Each day is cached as a native-resolution CONUS GeoTIFF, then area-averaged
onto the PRISM 4km grid (the same grid NSIDC-0719 used) so SWE, PCP, and TMP
stack cell-for-cell.
"""

import gzip
import io
import math
import os
import tarfile
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import convolve

from .prism import Manifest, _date_range_no_leap, _resolve
from .preflight import preflight_nsidc

SNODAS_BASE = (
    "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/"
    "{d:%Y}/{d:%m_%b}/SNODAS_{d:%Y%m%d}.tar"
)
# First day in the masked archive
SNODAS_START = date(2003, 9, 30)
SNODAS_NODATA = -9999
NC_VARIABLES = ("SWE", "DEPTH")

# SNODAS product code per variable, as it appears in the tar member names
_PRODUCT = {"SWE": "ssmv11034tS", "DEPTH": "ssmv11036tS"}

# .npy filename per variable, matching Extract_SWE.py -> swe.npy
_NPY_NAME = {"SWE": "swe", "DEPTH": "depth"}

# Target grid: PRISM 4km (2.5 arc-minute), edges rather than centroids. This is
# also the NSIDC-0719 grid, so outputs line up with stacks built from it.
_GRID_RES = 1 / 24
_GRID_WEST = -125.0 - _GRID_RES / 2
_GRID_NORTH = 49.916666666667 + _GRID_RES / 2
_GRID_ROWS = 621
_GRID_CRS = "EPSG:4269"


def _grid_window(bbox):
    """
    Convert (minx, miny, maxx, maxy) into a window on the PRISM 4km grid.

    Returns (row_north, col_west, height, width, transform), where transform
    is the north-up affine of the window. Uses the same floor/ceil edge math
    as _prism_bbox_indices, so SWE comes out exactly the shape of the PRISM
    stacks for the same bbox.
    """
    minx, miny, maxx, maxy = bbox

    col_west = int(math.floor((minx - _GRID_WEST) / _GRID_RES))
    col_east_excl = int(math.ceil((maxx - _GRID_WEST) / _GRID_RES))
    row_north = int(math.floor((_GRID_NORTH - maxy) / _GRID_RES))
    row_south_excl = int(math.ceil((_GRID_NORTH - miny) / _GRID_RES))

    height = row_south_excl - row_north
    width = col_east_excl - col_west
    transform = from_origin(
        _GRID_WEST + col_west * _GRID_RES,
        _GRID_NORTH - row_north * _GRID_RES,
        _GRID_RES,
        _GRID_RES,
    )
    return row_north, col_west, height, width, transform


def swe_grid(bbox):
    """
    Cell-centre (lat, lon) arrays for the bbox window.

    lat is ascending (south-to-north), matching the row order of the .npy
    stacks; lon is ascending (west-to-east).
    """
    _, _, height, width, transform = _grid_window(bbox)
    west, north = transform.c, transform.f
    lat = north - (np.arange(height)[::-1] + 0.5) * _GRID_RES
    lon = west + (np.arange(width) + 0.5) * _GRID_RES
    return lat, lon


def _check_start(start: date):
    if start < SNODAS_START:
        raise ValueError(
            f"SNODAS starts on {SNODAS_START.isoformat()}; "
            f"manifest starts on {start.isoformat()}."
        )


def _parse_header(text: str) -> dict[str, str]:
    """
    Parse a NOHRSC 'key: value' header file.
    """
    header = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip()] = value.strip()
    return header


def _fetch_snodas_day(d: date, variables) -> dict | None:
    """
    Download one day's tar and decode ``variables`` into {var: (grid, header)}.

    Returns None if the day is missing from the archive (HTTP 404).
    """
    url = SNODAS_BASE.format(d=d)
    print(f"[swecast] SNODAS {d}: downloading {url}")
    resp = requests.get(url, timeout=300)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    try:
        tar = tarfile.open(fileobj=io.BytesIO(resp.content))
    except tarfile.TarError as e:
        preview = resp.content[:500].decode("utf-8", errors="replace")
        raise ValueError(
            f"Expected tar for SNODAS {d}.\nResponse preview:\n{preview}"
        ) from e

    data, header_text = {}, {}
    with tar:
        for member in tar.getmembers():
            var = next((v for v in variables if _PRODUCT[v] in member.name), None)
            if var is None:
                continue
            content = gzip.decompress(tar.extractfile(member).read())
            if member.name.endswith(".dat.gz"):
                data[var] = content
            elif member.name.endswith((".txt.gz", ".Hdr.gz")):
                header_text[var] = content.decode("ascii", errors="replace")

    grids = {}
    for var in variables:
        # Skip variables that are missing either the data or the header
        if var not in data or var not in header_text:
            print(f"[swecast] SNODAS {d}: no {var} product in {url}")
            continue
        header = _parse_header(header_text[var])
        rows = int(header["Number of rows"])
        cols = int(header["Number of columns"])
        grid = np.frombuffer(data[var], dtype=">i2").reshape(rows, cols)
        grids[var] = (grid.astype(np.int16), header)
    return grids


def _write_snodas_tif(grid, header, out: Path):
    """
    Write a native-resolution SNODAS grid, georeferenced from its own header.

    The masked grid's extent shifts slightly between eras (e.g. pre/post
    2013), so the transform has to come from each day's header.
    """
    transform = from_origin(
        float(header["Minimum x-axis coordinate"]),
        float(header["Maximum y-axis coordinate"]),
        float(header["X-axis resolution"]),
        float(header["Y-axis resolution"]),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
        nodata=SNODAS_NODATA,
        compress="deflate",
    ) as dst:
        dst.write(grid, 1)
        dst.update_tags(units="mm")
    # Rename last so an interrupted write never looks like a cache hit
    tmp.replace(out)


def _download_snodas_day(d: date, cache_dir: Path, variables) -> dict | None:
    """
    Return {var: cached native-resolution GeoTIFF | None} for one day.

    Returns None (after a warning) if the day is missing from the archive;
    callers write an all-NaN frame that filled_data gap-fills later.
    """
    paths = {
        var: Path(cache_dir) / "snodas" / _NPY_NAME[var] / f"{d:%Y%m%d}.tif"
        for var in variables
    }
    if all(p.exists() for p in paths.values()):
        return paths

    grids = _fetch_snodas_day(d, variables)
    if grids is None:
        print(f"[swecast] SNODAS {d}: missing from archive, leaving it as NaN")
        return None
    for var, (grid, header) in grids.items():
        _write_snodas_tif(grid, header, paths[var])
    return {var: paths[var] if var in grids else None for var in variables}


def _read_day_on_grid(tif_path, bbox) -> np.ndarray:
    """
    Area-average a cached SNODAS GeoTIFF onto the bbox window of the 4km grid.

    Returns a north-up (H, W) float64 array with NaN where every source cell
    was nodata. The ~1km cells don't line up with the 4km edges, so this is a
    proper area-weighted average rather than a 5x5 block mean.
    """
    _, _, height, width, transform = _grid_window(bbox)
    out = np.full((height, width), np.nan, dtype=np.float64)
    with rasterio.open(tif_path) as src:
        reproject(
            rasterio.band(src, 1),
            out,
            src_nodata=SNODAS_NODATA,
            dst_transform=transform,
            dst_crs=_GRID_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return out


def _read_stacks(manifest, cache_dir, variables) -> tuple[list, dict]:
    """
    Read every day in the manifest for ``variables``, north-up.

    Walks _date_range_no_leap (like the PRISM builders) so the time axis lines
    up row-for-row with pcp.npy / tmp.npy. Returns (dates, {var: (T, H, W)}).
    """
    _check_start(manifest.start)
    dates = list(_date_range_no_leap(manifest.start, manifest.end))
    _, _, height, width, _ = _grid_window(manifest.bbox)

    stacks = {var: np.full((len(dates), height, width), np.nan) for var in variables}
    for i, d in enumerate(dates):
        paths = _download_snodas_day(d, cache_dir, variables)
        if paths is None:
            continue
        for var in variables:
            if paths[var] is not None:
                stacks[var][i] = _read_day_on_grid(paths[var], manifest.bbox)
    return dates, stacks


def build_swe_stacks(
    manifest: Manifest,
    output_dir: Path,
    cache_dir: Path | None = None,
    write_npy: bool | None = None,
    variables: tuple | None = None,
) -> dict[str, Path]:
    """
    Download SNODAS SWE/DEPTH per day, resample to the 4km grid, clip to bbox,
    and write band-per-day GeoTIFFs.

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".cache"

    dates, stacks = _read_stacks(manifest, cache_dir, variables)
    _, _, height, width, transform = _grid_window(manifest.bbox)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": len(dates),
        "crs": _GRID_CRS,
        "transform": transform,
        "compress": "lzw",
        "nodata": np.nan,
    }

    outputs = {}
    for variable in variables:
        out_path = output_dir / f"{variable.lower()}_stack.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, (arr, d) in enumerate(zip(stacks[variable], dates), start=1):
                dst.write(arr.astype(np.float32), i)
                dst.update_tags(i, date=d.isoformat())

        outputs[variable] = out_path
        print(f"[swecast] {variable}: wrote {len(dates)}-band stack -> {out_path}")

    if write_npy:
        build_npy_swe_stacks(manifest, output_dir, cache_dir=cache_dir)

    return outputs


def build_npy_swe_stacks(
    manifest: Manifest,
    output_dir: Path,
    cache_dir: Path | None = None,
    variables: tuple | None = None,
) -> dict[str, Path]:
    """
    Forklift of Extract_SWE.py, retimed so the time axis lines up with PRISM.

    Walks calendar dates from manifest.start to manifest.end (skipping
    Feb 29, like build_npy_stacks) and writes (T, H, W) stacks:
        swe.npy   (always)
        depth.npy (only if "DEPTH" is in ``variables``)

    Rows run south-to-north (row 0 is the southern edge), the same order as
    the PRISM .npy stacks and the original NSIDC-0719 slice, so station_cells
    and multi-channel stacking stay aligned. Days missing from SNODAS are NaN.

    ``cache_dir`` and ``variables`` (mapped to ``manifest.npy_nc_variables``)
    fall back to the manifest if not passed.
    """
    cache_dir = _resolve(cache_dir, manifest, "cache_dir", None)
    variables = _resolve(variables, manifest, "npy_nc_variables", ("SWE",))

    preflight_nsidc()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".cache"

    dates, stacks = _read_stacks(manifest, cache_dir, variables)
    print(f"[swecast] SWE region: {stacks[variables[0]].shape[1:]}")

    outputs = {}
    for variable in variables:
        # flip north-up to south-to-north rows
        ds_swe = stacks[variable][:, ::-1, :]
        out_path = output_dir / f"{_NPY_NAME[variable]}.npy"
        np.save(out_path, ds_swe)
        outputs[variable] = out_path
        print(f"[swecast] {variable}: wrote {len(dates)}-day .npy stack -> {out_path}")

    return outputs


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


def _write_stack(out_path, data, profile):
    """
    Write a 3D array as a multi-band GeoTIFF, keeping the input's georeferencing.
    """
    profile.update(dtype="float32", count=data.shape[0], nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


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
