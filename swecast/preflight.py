"""Environment preflight checks for swecast."""

import os
import sys

# Packages required for all functionality
_PACKAGES_PRISM = {
    "requests": "requests",
    "numpy": "numpy",
    "rasterio": "rasterio",
    "shapely": "shapely",
}

# Additional packages required for NSIDC SWE downloads
_PACKAGES_NSIDC = {
    "xarray": "xarray",
    "rioxarray": "rioxarray",
    "netCDF4": "netcdf4",
}

# Environment variables required for NSIDC SWE downloads
_ENV_NSIDC = {
    "EARTHDATA_USERNAME": "NASA Earthdata username (https://urs.earthdata.nasa.gov)",
    "EARTHDATA_PASSWORD": "NASA Earthdata password",
}

# additional required
_PACKAGES_MODELS = {
    "tensorflow": "tensorflow",
    "keras": "keras",
    "optuna": "optuna",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "scipy": "scipy",
}


def _check_packages(packages: dict, failures: list) -> None:
    '''Check that required packages are installed, and report any missing ones.'''
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
            print(f"  Found package '{import_name}' (pip install {pip_name})")
        except ImportError:
            print(f"  Missing package '{import_name}' (pip install {pip_name})")
            failures.append(f"Missing package '{import_name}' (pip install {pip_name})")


def _check_env(env_vars: dict, failures: list) -> None:
    '''Check that required environment variables are SET, and report any missing ones.'''
    missing = []
    for var, description in env_vars.items():
        if os.environ.get(var):
            print(f" Found environment variable ${var} ({description})")
        else:
            print(f" Missing environment variable ${var} ({description})")
            failures.append(f"Missing environment variable ${var} ({description})")
            missing.append(var)
    if missing:
        print("\n  To set them, run:")
        for var in missing:
            print(f'    export {var}="your-value"')


def _finish(failures: list, raise_on_error: bool) -> bool:
    '''Print summary of preflight checks, and exit with error if any failed and raise_on_error is True.'''
    print()
    if failures:
        print(f"  {len(failures)} check(s) failed.\n")
        if raise_on_error:
            sys.exit(1)
        return False
    print("  All checks passed.\n")
    return True


def preflight(raise_on_error: bool = True) -> bool:
    """
    Run all preflight checks (packages + environment variables).
    Called automatically by build_stacks and build_swe_stacks.
    Add new checks here as functionality grows.
    """
    failures = []
    print("swecast preflight checks")
    print("=" * 40)
    print("\nPackages:")
    _check_packages(_PACKAGES_PRISM, failures)
    _check_packages(_PACKAGES_NSIDC, failures)
    _check_packages(_PACKAGES_MODELS, failures)
    print("\nEnvironment variables:")
    _check_env(_ENV_NSIDC, failures)
    return _finish(failures, raise_on_error)


def preflight_models(raise_on_error: bool = True) -> bool:
    """Preflight checks for training"""
    failures = []
    print("swecast preflight checks (models)")
    print("=" * 40)
    print("\nPackages:")
    _check_packages(_PACKAGES_MODELS, failures)
    return _finish(failures, raise_on_error)


def preflight_prism(raise_on_error: bool = True) -> bool:
    """Preflight checks for PRISM-only functionality (no Earthdata required)."""
    failures = []
    print("swecast preflight checks (PRISM)")
    print("=" * 40)
    print("\nPackages:")
    _check_packages(_PACKAGES_PRISM, failures)
    return _finish(failures, raise_on_error)


def preflight_nsidc(raise_on_error: bool = True) -> bool:
    """Preflight checks for NSIDC SWE functionality (includes Earthdata credentials)."""
    failures = []
    print("swecast preflight checks (NSIDC)")
    print("=" * 40)
    print("\nPackages:")
    _check_packages(_PACKAGES_PRISM, failures)
    _check_packages(_PACKAGES_NSIDC, failures)
    print("\nEnvironment variables:")
    _check_env(_ENV_NSIDC, failures)
    return _finish(failures, raise_on_error)
