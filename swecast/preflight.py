"""
Environment preflight checks for swecast.
"""

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


#Context: This function looks at a list of python packages and checks if each one is installed.
#Context: If a package is missing it adds a note to the failures list so we can tell the user later.
#Context: The 'packages' input is a dict like {"numpy": "numpy"} where the left side is what you import.
#Context: It dosent return anything, it just prints messages and fills up the failures list.
def _check_packages(packages: dict, failures: list) -> None:
    """
    Check that required packages are installed, and report any missing ones.
    """
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
            print(f"  Found package '{import_name}' (pip install {pip_name})")
        except ImportError:
            print(f"  Missing package '{import_name}' (pip install {pip_name})")
            failures.append(f"Missing package '{import_name}' (pip install {pip_name})")


#Context: This checks that certain environment variables (settings stored outside the code) are set.
#Context: For example it makes sure your NASA username and password exist before we try to use them.
#Context: If some are missing it prints a helpful 'export NAME=value' tip so you know how to set them.
#Context: Example input: {"EARTHDATA_USERNAME": "your NASA login"} and a failures list to add problems too.
def _check_env(env_vars: dict, failures: list) -> None:
    """
    Check that required environment variables are SET, and report any missing ones.
    """
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


#Context: This wraps up the checks and tells you the final result.
#Context: If the failures list has stuff in it, it prints how many things failed.
#Context: When raise_on_error is True and there were problems, it stops the program right away.
#Context: It returns True if everything passed, or False if something went wrong (and we didnt stop).
def _finish(failures: list, raise_on_error: bool) -> bool:
    """
    Print summary of preflight checks, and exit with error if any failed and raise_on_error is True.
    """
    print()
    if failures:
        print(f"  {len(failures)} check(s) failed.\n")
        if raise_on_error:
            sys.exit(1)
        return False
    print("  All checks passed.\n")
    return True


#Context: This is the main check that runs everything at once before the real work starts.
#Context: It checks all the packages (PRISM, NSIDC and model ones) plus the needed environment variables.
#Context: Think of it like a pre-flight checklist a pilot does before takeoff, making sure your all set up.
#Context: Returns True if everything is good. Example output: True (or False if somethings missing).
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


#Context: This is a smaller check just for the model training part of swecast.
#Context: It only looks at the packages you need for training (like tensorflow, keras, pandas etc).
#Context: Use this when you only care about training and dont need the data download stuff.
#Context: Returns True when the training packages are all present. Example output: True
def preflight_models(raise_on_error: bool = True) -> bool:
    """
    Preflight checks for training
    """
    failures = []
    print("swecast preflight checks (models)")
    print("=" * 40)
    print("\nPackages:")
    _check_packages(_PACKAGES_MODELS, failures)
    return _finish(failures, raise_on_error)


#Context: This check is for when you only want to work with PRISM weather data.
#Context: PRISM dosent need a NASA Earthdata login, so this skips the username/password checks.
#Context: It just makes sure the basic packages like requests, numpy and rasterio are installed.
#Context: Returns True if the PRISM packages are ready to go. Example output: True
def preflight_prism(raise_on_error: bool = True) -> bool:
    """
    Preflight checks for PRISM-only functionality (no Earthdata required).
    """
    failures = []
    print("swecast preflight checks (PRISM)")
    print("=" * 40)
    print("\nPackages:")
    _check_packages(_PACKAGES_PRISM, failures)
    return _finish(failures, raise_on_error)


#Context: This check is for the NSIDC snow data (SWE means snow water equivalent).
#Context: That data comes from NASA so this one DOES need your Earthdata username and password set.
#Context: It checks the base packages, the extra NSIDC ones, and the environment variables too.
#Context: Returns True if packages and credentials are all there. Example output: True
def preflight_nsidc(raise_on_error: bool = True) -> bool:
    """
    Preflight checks for NSIDC SWE functionality (includes Earthdata credentials).
    """
    failures = []
    print("swecast preflight checks (NSIDC)")
    print("=" * 40)
    print("\nPackages:")
    _check_packages(_PACKAGES_PRISM, failures)
    _check_packages(_PACKAGES_NSIDC, failures)
    print("\nEnvironment variables:")
    _check_env(_ENV_NSIDC, failures)
    return _finish(failures, raise_on_error)
