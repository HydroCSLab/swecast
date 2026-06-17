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

PRISM_BASE = (
    "https://services.nacse.org/prism/data/get/us/4km/{variable}/{date}?format=bil"
)
VARIABLES = ("ppt", "tmean")

# Forklifted from Extract_PCP.py / Extract_TEMP.py.
# The original scripts hardcoded h_grid=75, v_grid=333, width=310, height=142 for
# the California study area; we now derive them from manifest.bbox + the .hdr
# geographic metadata at runtime so any bbox produces correct indices.

# .npy filename per PRISM variable, matching Extract_PCP.py -> pcp.npy and Extract_TEMP.py -> tmp.npy
_NPY_NAME = {"ppt": "pcp", "tmean": "tmp"}


def _read_bil_hdr(bil):
    """Parse the .hdr sidecar of a PRISM .bil file and return its full metadata dict."""
    bil = str(bil)
    hdr = bil.replace(".bil", ".hdr")
    meta = {}
    with open(hdr) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                meta[parts[0].upper()] = parts[1]
    return meta


def _prism_bbox_indices(bil, bbox):
    """Convert a (minx, miny, maxx, maxy) bbox to (h_grid, v_grid, width, height)
    using the geographic metadata (ULXMAP, ULYMAP, XDIM, YDIM) in the .bil's .hdr.

    h_grid = column index of minx (west edge of bbox)
    v_grid = row index of miny (south edge of bbox; PRISM rows count north-to-south,
             so v_grid is the BOTTOM row; the inner loop reads bil_ds[v_grid - j, ...]
             to walk south-up, which is what the original scripts produced)
    width  = number of columns between minx and maxx
    height = number of rows between miny and maxy

    For the original California bbox (-121.9, 36.08, -109, 41.98) on the standard
    PRISM 4km grid, this returns ~(75, 333, 310, 142), matching the hardcoded
    constants in Extract_PCP.py.
    """
    meta = _read_bil_hdr(bil)
    ulxmap = float(meta["ULXMAP"])
    ulymap = float(meta["ULYMAP"])
    xdim = float(meta["XDIM"])
    ydim = float(meta["YDIM"])

    minx, miny, maxx, maxy = bbox
    h_grid = int(round((minx - ulxmap) / xdim))
    v_grid = int(round((ulymap - miny) / ydim))
    width = int(round((maxx - minx) / xdim))
    height = int(round((maxy - miny) / ydim))
    return h_grid, v_grid, width, height


@dataclass
class Manifest:
    """Job specification for a swecast run.

    Required fields define scope:
        start, end : date range
        bbox       : (minx, miny, maxx, maxy) in WGS84

    Optional fields control workflow and training. All have sensible defaults
    matching the original research scripts in files_to_forklift/, so the
    minimal call ``Manifest(start, end, bbox)`` reproduces the paper's setup.

    Workflow knobs (data acquisition):
        cache_dir         : where to cache downloaded files (None -> output_dir/.cache)
        write_npy         : also emit .npy stacks alongside .tif (default True)
        nc_variables      : NetCDF variables to extract for .tif (default both)
        npy_nc_variables  : NetCDF variables to extract for .npy (default SWE-only,
                            matching Extract_SWE.py)
        fetch_stations    : force-refresh the USDA AWDB stations CSV

    Training hyperparameters (forklifted from ConvLSTM_*.py):
        num_days_train    : sequence length (5 = forecast day 5 from days 1-4)
        num_data_used     : how many leading days from the .npy stack to use
        epochs, batch_size: keras.fit args
        train_split       : train/val split ratio (0.8)
        log_norm_divisor  : the "/3.5" in log10(1+x)/3.5 normalization
        early_stopping_patience, reduce_lr_patience : callback patience values
        num_stations      : number of stations in the per-station NS loop (75)

    Optuna tuning:
        n_trials          : number of Optuna trials (20)
    """

    start: date
    end: date
    bbox: tuple  # (minx, miny, maxx, maxy) in WGS84

    # Stations
    fetch_stations: bool = False

    # Workflow
    cache_dir: "Path | None" = None
    write_npy: bool = True
    nc_variables: tuple = ("SWE", "DEPTH")
    npy_nc_variables: tuple = ("SWE",)

    # Training hyperparameters (forklifted from ConvLSTM_*.py defaults)
    num_days_train: int = 5
    num_data_used: int = 7300
    epochs: int = 50
    batch_size: int = 16
    train_split: float = 0.8
    log_norm_divisor: float = 3.5
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    num_stations: int = 75

    # Optuna
    n_trials: int = 20

    # Model serialization (each train_* function picks a variant-specific default
    # filename when model_filename is None, so multiple variants in one output_dir
    # don't clobber each other).
    save_model: bool = True
    model_format: str = "keras"  # "keras" (recommended) or "h5"
    model_filename: "str | None" = None


def _resolve(value, manifest, attr, default):
    """Resolve a parameter using the precedence: explicit value > manifest field > default.

    Returns ``value`` if not None; else ``getattr(manifest, attr)`` if a manifest
    is provided; else ``default``.
    """
    if value is not None:
        return value
    if manifest is not None:
        return getattr(manifest, attr, default)
    return default


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _date_range_no_leap(start: date, end: date):
    """Yield dates from start to end (inclusive), skipping Feb 29.

    NSIDC-0719 uses a 365-day calendar (omits Feb 29). For PRISM and NSIDC
    .npy stacks to share a time axis, both builders iterate this filtered
    range so row ``i`` of every stack corresponds to the same calendar date.
    """
    for d in _date_range(start, end):
        if not (d.month == 2 and d.day == 29):
            yield d


def _download_bil(variable: str, d: date, cache_dir: Path) -> Path:
    date_str = d.strftime("%Y%m%d")
    bil_path = cache_dir / variable / f"{date_str}.bil"

    if bil_path.exists():
        print(f"[swecast] {variable} {d}: using cached {bil_path}")
        return bil_path

    url = PRISM_BASE.format(variable=variable, date=date_str)
    print(f"[swecast] {variable} {d}: downloading {url}")
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


def read_bil_file(bil):
    """Forklifted from Extract_PCP.py.

    Reads a PRISM .bil file by parsing its sibling .hdr metadata and returns a 2D numpy array.
    The numpy-based reader works for both ppt and tmean (Extract_TEMP.py used GDAL but
    produces the same shape; the numpy version avoids the GDAL dependency).
    """
    bil = str(bil)
    hdr = bil.replace(".bil", ".hdr")
    meta = {}
    with open(hdr) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                meta[parts[0].upper()] = parts[1]

    rows = int(meta["NROWS"])
    cols = int(meta["NCOLS"])
    dtype_map = {"16": np.int16, "32": np.float32, "8": np.uint8}
    nbits = meta.get("NBITS", "16")
    dtype = dtype_map.get(nbits, np.int16)

    data = np.fromfile(bil, dtype=dtype).reshape(rows, cols)
    return data


def build_npy_stacks(
    manifest: Manifest,
    output_dir: Path,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """Forklift of Extract_PCP.py and Extract_TEMP.py.

    Downloads PRISM ppt and tmean BIL files for each day in manifest and writes
    flat .npy stacks matching the original scripts:
        pcp.npy   shape: (num_days, height=142, width=310)
        tmp.npy   shape: (num_days, height=142, width=310)

    The hardcoded grid offset (v_grid, h_grid) and the v_grid - j flip are preserved
    verbatim so the resulting arrays are byte-comparable to the standalone scripts.
    """
    preflight_prism()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _resolve(cache_dir, manifest, "cache_dir", None)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".cache"

    dates = list(_date_range_no_leap(manifest.start, manifest.end))
    outputs = {}

    # Derive h_grid, v_grid, width, height from the first .bil's .hdr + manifest.bbox
    # (replaces the original hardcoded h_grid=75, v_grid=333, width=310, height=142
    # in Extract_PCP.py / Extract_TEMP.py).
    sample_bil = _download_bil(VARIABLES[0], dates[0], cache_dir)
    h_grid, v_grid, width, height = _prism_bbox_indices(sample_bil, manifest.bbox)
    print(
        f"[swecast] PRISM region: h_grid={h_grid}, v_grid={v_grid}, "
        f"size=({height}, {width})"
    )

    for variable in VARIABLES:
        ds = []
        for d in dates:
            bil = _download_bil(variable, d, cache_dir)
            print("Processing file:", bil)
            bil_ds = read_bil_file(bil)
            ds1day = np.zeros((height, width))
            for j in range(0, height):
                for k in range(0, width):
                    ds1day[j, k] = bil_ds[v_grid - j, h_grid + k]
            ds.append(ds1day)
        print("ds data shape:", np.array(ds).shape)
        out_path = output_dir / f"{_NPY_NAME[variable]}.npy"
        np.save(out_path, np.array(ds))
        outputs[variable] = out_path
        print(f"[swecast] {variable}: wrote {len(dates)}-day .npy stack -> {out_path}")

    return outputs


def build_stacks(
    manifest: Manifest,
    output_dir: Path,
    cache_dir: Path | None = None,
    write_npy: bool | None = None,
) -> dict[str, Path]:
    """
    Download PRISM ppt and tmean for each day in manifest, clip to bbox,
    and write a band-per-day stacked GeoTIFF for each variable.

    If ``write_npy`` is True (default), also emits the script-compatible
    pcp.npy and tmp.npy stacks via build_npy_stacks. ``cache_dir`` and
    ``write_npy`` fall back to the same-named fields on the Manifest when
    not passed explicitly.

    Returns a dict mapping variable name -> GeoTIFF output path.
    """
    cache_dir = _resolve(cache_dir, manifest, "cache_dir", None)
    write_npy = _resolve(write_npy, manifest, "write_npy", True)

    preflight_prism()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".cache"

    geom = [box(*manifest.bbox).__geo_interface__]
    dates = list(_date_range_no_leap(manifest.start, manifest.end))
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
        print(f"[swecast] {variable}: wrote {len(dates)}-band stack -> {out_path}")

    if write_npy:
        build_npy_stacks(manifest, output_dir, cache_dir=cache_dir)

    return outputs
