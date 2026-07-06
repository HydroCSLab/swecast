"""
Run a trained ConvLSTM on a target date and compare against the actual SWE.

Retrospective validation: pick a date covered by NSIDC-0719, have the
model forecast that day's SWE from the (num_days_train - 1) days
before it, then compare the prediction with the actual NSIDC value.
This is what the QGIS validation plugin sits on top of.
"""

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset as ncdataset
from netCDF4 import num2date

from .preflight import preflight_models
from .prism import (
    Manifest,
    _resolve,
    _download_bil,
    _prism_bbox_indices,
    read_bil_file,
)
from .nsidc import (
    _download_nc,
    _water_year,
    _nc_bbox_indices,
    filled_data,
)

# Channel order per variant. Must match the order the model was trained with
# (see train_swe_pcp / train_swe_tmp / train_swe_tmp_pcp / train_tmp_pcp
# in models.py; they all stack as np.stack((swe, ...), axis=3) in this order).
_VARIANTS = {
    "swe": ("SWE",),
    "swe_pcp": ("SWE", "PCP"),
    "swe_tmp": ("SWE", "TMP"),
    "swe_tmp_pcp": ("SWE", "TMP", "PCP"),
    "tmp_pcp": ("TMP", "PCP"),  # swe is only used for the actual comparison
}


#Context: This function looks inside a NetCDF weather file and finds which slot (the index)
#Context: in its list of days matches the date you are after. It reads the files own time
#Context: list so we dont have to guess how the calander works. It returns a number like 3,
#Context: meaning the 4th day in the file (counting starts at 0). If the date isnt there it errors out.
#Context: Example output: 3
def _find_day_index(nc_path, target):
    """
    Index along ``nc_path``'s time axis that matches ``target``.

    Uses the file's own time variable and units so we don't have to
    assume a particular water-year start or leap-day handling.
    """
    with ncdataset(str(nc_path)) as ds:
        times = ds.variables["time"]
        date_objs = num2date(
            times[:], times.units, getattr(times, "calendar", "standard")
        )
    for i, d_obj in enumerate(date_objs):
        if (d_obj.year, d_obj.month, d_obj.day) == (
            target.year,
            target.month,
            target.day,
        ):
            return i
    raise ValueError(f"Date {target.isoformat()} not found in {nc_path}")


#Context: This grabs one single day of snow water (called SWE, basicly how much water is in the snow)
#Context: from the right yearly file, and crops it down to just the map box we care about using
#Context: the lat/lon cutoffs. Missing or bad pixels get turned into NaN (a "no value" marker) so
#Context: they can be filled in later instead of showing a weird fake number like -999.
#Context: Example output: [[0.0, 1.2, NaN], [3.4, 0.0, 2.1]]
def _fetch_swe_day(d, nc_paths, lat_lo, lat_hi, lon_lo, lon_hi):
    """
    Read one day of SWE from the NSIDC NetCDF for water_year(d).
    """
    wy = _water_year(d)
    if wy not in nc_paths:
        raise ValueError(f"No NSIDC file cached for water year {wy}")
    day_idx = _find_day_index(nc_paths[wy], d)
    with ncdataset(str(nc_paths[wy])) as ds:
        arr = ds.variables["SWE"][day_idx, lat_lo:lat_hi, lon_lo:lon_hi]
    # Convert masked elements (NSIDC fill_value, e.g. -999) to NaN so filled_data
    # can gap-fill them. np.asarray would strip the mask and expose -999.
    return np.ma.filled(np.ma.asarray(arr).astype(np.float64), np.nan)


#Context: This reads one day of PRISM weather data (like rainfall or temperature) for the
#Context: requested variable, downloads the file if needed, and copies it into a clean grid
#Context: cropped to our box. It flips the rows so the map comes back top-to-bottom (north on top),
#Context: the same way the rest of the pipeline expects it. Its a bit slow becuase it loops cell by cell.
#Context: Example output: [[10.5, 11.0], [9.8, 12.3]]
def _fetch_prism_day(prism_var, d, cache_dir, bbox):
    """
    Read one day of PRISM ``prism_var`` and clip to bbox.

    Flips through v_grid / h_grid the same way the build pipeline does so
    rows come back north-to-south.
    """
    bil = _download_bil(prism_var, d, cache_dir)
    bil_ds = read_bil_file(bil)
    h_grid, v_grid, width, height = _prism_bbox_indices(bil, bbox)
    out = np.zeros((height, width))
    for j in range(0, height):
        for k in range(0, width):
            out[j, k] = bil_ds[v_grid - j, h_grid + k]
    return out


#Context: This is the main job. It takes a trained AI model and a date you want to forecast, feeds
#Context: it the few days of snow/weather data right before that date, and asks the model to guess the
#Context: snow water for the target day. Then it compares that guess to the real measured value to see
#Context: how close it was. It handles downloads, cropping to the map box, cleaning up the data, and
#Context: undoing the math scaling so the numbers make sense again.
#Context: It returns a dictionary with the prediction, the actual value, the difference (residual), and
#Context: some extra info like the dates used and the map coordinates.
#Context: Example output: {"predicted":[[..]], "actual":[[..]], "residual":[[..]], "target_date": ...}
def predict(
    model_path,
    target_date,
    manifest,
    *,
    variant="swe",
    cache_dir=None,
    num_days_train=None,
    swe_scaling_factor=None,
    earthdata_username=None,
    earthdata_password=None,
):
    """
    Run a trained ConvLSTM on the inputs preceding ``target_date``.

    Returns the predicted SWE for ``target_date`` along with the actual
    NSIDC-0719 SWE for that day and the residual.

    Parameters
    ----------
    model_path : str | Path
        Trained Keras model (.keras or .h5).
    target_date : datetime.date
        Day to predict. The model uses the (num_days_train - 1) days
        immediately before it as input.
    manifest : Manifest
        Supplies bbox and (via ``_resolve``) defaults for cache_dir,
        num_days_train, and swe_scaling_factor.
    variant : str
        One of "swe", "swe_pcp", "swe_tmp", "swe_tmp_pcp",
        "tmp_pcp". Has to match the channel layout the model was
        trained on.
    earthdata_username, earthdata_password : str, optional
        Override $EARTHDATA_USERNAME / $EARTHDATA_PASSWORD. Handy when
        the QGIS plugin holds creds in QSettings rather than the env.

    Returns
    -------
    dict with keys:
        predicted    : (H, W), denormalized SWE prediction
        actual       : (H, W), gap-filled NSIDC SWE for target_date
        actual_raw   : (H, W), raw NSIDC SWE for target_date (NaNs kept)
        residual     : predicted - actual
        target_date  : echoed back
        input_dates  : list of dates fed to the model
        bbox         : (minx, miny, maxx, maxy) used
        bbox_indices : (lat_lo, lat_hi, lon_lo, lon_hi) into the .nc grid
        nc_lat       : 1D latitudes covered
        nc_lon       : 1D longitudes covered
        variant      : echoed back
    """
    preflight_models()

    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {tuple(_VARIANTS)}; got {variant!r}")
    channels = _VARIANTS[variant]

    cache_dir = _resolve(cache_dir, manifest, "cache_dir", None)
    cache_dir = Path(cache_dir) if cache_dir else Path("./.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    num_days_train = _resolve(num_days_train, manifest, "num_days_train", 5)
    swe_scaling_factor = _resolve(swe_scaling_factor, manifest, "swe_scaling_factor", 3.5)

    seq_len = num_days_train - 1  # number of input frames (e.g. 4 for default)
    input_dates = [target_date - timedelta(days=seq_len - i) for i in range(seq_len)]

    # Earthdata creds (kwargs override env vars)
    user = earthdata_username or os.environ.get("EARTHDATA_USERNAME")
    pw = earthdata_password or os.environ.get("EARTHDATA_PASSWORD")
    if not user or not pw:
        raise RuntimeError(
            "Earthdata credentials missing. Set EARTHDATA_USERNAME and "
            "EARTHDATA_PASSWORD, or pass earthdata_username/password kwargs."
        )

    # Download / locate the NSIDC water-year files needed to cover both the
    # input window and the target date.
    needed_wys = sorted({_water_year(d) for d in [*input_dates, target_date]})
    nc_paths = {wy: _download_nc(wy, cache_dir, user, pw) for wy in needed_wys}

    # Bbox-to-grid-indices using one .nc as the reference grid. All NSIDC-0719
    # files share the same lat/lon arrays so any of them works.
    sample_nc = nc_paths[needed_wys[0]]
    lat_lo, lat_hi, lon_lo, lon_hi = _nc_bbox_indices(sample_nc, manifest.bbox)
    H = lat_hi - lat_lo
    W = lon_hi - lon_lo

    with ncdataset(str(sample_nc)) as ds:
        nc_lat = np.asarray(ds.variables["lat"][:])[lat_lo:lat_hi]
        nc_lon = np.asarray(ds.variables["lon"][:])[lon_lo:lon_hi]

    # Build the (seq_len, H, W, n_channels) input stack
    frames = []
    for d in input_dates:
        per_channel = []
        for ch in channels:
            if ch == "SWE":
                arr = _fetch_swe_day(d, nc_paths, lat_lo, lat_hi, lon_lo, lon_hi)
            elif ch == "PCP":
                arr = _fetch_prism_day("ppt", d, cache_dir, manifest.bbox)
            elif ch == "TMP":
                arr = _fetch_prism_day("tmean", d, cache_dir, manifest.bbox)
            else:
                raise ValueError(f"Unknown channel {ch!r}")
            per_channel.append(arr)
        frames.append(np.stack(per_channel, axis=-1))
    x = np.stack(frames, axis=0)  # (seq_len, H, W, n_channels)

    # Gap-fill each channel independently with the same 3x3x3 kernel used in
    # training. The temporal context is much smaller here (seq_len frames vs.
    # the full 7300-day stack), so filled values won't be byte-identical to
    # what training saw, but close enough for retrospective validation.
    for ch_idx in range(x.shape[-1]):
        x[..., ch_idx] = filled_data(x[..., ch_idx])

    # Replicate training normalization: log10(1 + x) / swe_scaling_factor on SWE.
    # PCP/TMP are fed raw (matching ConvLSTM_SWE_PCP / TMP / TMP_PCP scripts).
    if "SWE" in channels:
        swe_idx = channels.index("SWE")
        x[..., swe_idx] = np.log10(1 + x[..., swe_idx]) / swe_scaling_factor

    # Add batch dim
    x_input = x[np.newaxis, ...].astype(np.float32)

    # Lazy-load Keras to keep the rest of swecast importable without TF
    from tensorflow import keras

    model = keras.models.load_model(str(model_path))
    y_pred_norm = model.predict(x_input, verbose=0)
    y_pred_norm = np.squeeze(y_pred_norm)  # (H, W)

    # Inverse of log10(1 + y) / swe_scaling_factor; clamp negatives to 0 to match
    # the per-station postprocessing in the original ConvLSTM scripts.
    predicted = 10 ** (y_pred_norm * swe_scaling_factor) - 1
    predicted = np.where(predicted < 0, 0, predicted)

    # Actual SWE for target_date (raw + gap-filled, the latter being apples-to-
    # apples with what the model was trained against).
    actual_raw = _fetch_swe_day(target_date, nc_paths, lat_lo, lat_hi, lon_lo, lon_hi)
    actual_filled = filled_data(actual_raw[np.newaxis, ...])[0]

    residual = predicted - actual_filled

    return {
        "predicted": predicted,
        "actual": actual_filled,
        "actual_raw": actual_raw,
        "residual": residual,
        "target_date": target_date,
        "input_dates": input_dates,
        "bbox": manifest.bbox,
        "bbox_indices": (lat_lo, lat_hi, lon_lo, lon_hi),
        "nc_lat": nc_lat,
        "nc_lon": nc_lon,
        "variant": variant,
    }
