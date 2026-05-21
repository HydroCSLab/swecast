from .prism import Manifest, build_stacks, build_npy_stacks, read_bil_file
from .nsidc import (
    build_swe_stacks,
    build_npy_swe_stacks,
    fill_stacks,
    fill_npy,
    filled_data,
)
from .preflight import (
    preflight,
    preflight_prism,
    preflight_nsidc,
    preflight_models,
)
from .stations import (
    fetch_stations,
    stations_to_csv,
    get_stations,
    identify_station_cells,
)


def __getattr__(name):
    """Lazy-import TF-dependent functions so importing swecast does not pull in TensorFlow."""
    _models_exports = {
        "train_swe_forecast",
        "train_swe_only",
        "train_swe_pcp",
        "train_swe_temp",
        "train_swe_temp_pcp",
        "train_temp_pcp",
        "optimize_hyper_parameters",
    }
    if name in _models_exports:
        from . import models
        return getattr(models, name)
    if name == "predict":
        from . import predict as _predict_mod
        return _predict_mod.predict
    raise AttributeError(f"module 'swecast' has no attribute {name!r}")


__all__ = [
    "Manifest",
    "build_stacks",
    "build_npy_stacks",
    "read_bil_file",
    "build_swe_stacks",
    "build_npy_swe_stacks",
    "fill_stacks",
    "fill_npy",
    "filled_data",
    "preflight",
    "preflight_prism",
    "preflight_nsidc",
    "preflight_models",
    "fetch_stations",
    "stations_to_csv",
    "get_stations",
    "identify_station_cells",
    "train_swe_forecast",
    "train_swe_only",
    "train_swe_pcp",
    "train_swe_temp",
    "train_swe_temp_pcp",
    "train_temp_pcp",
    "optimize_hyper_parameters",
    "predict",
]
