"""
ConvLSTM training and hyperparameter tuning.

Mostly forklifted from the original per-variant scripts.
"""

from dataclasses import dataclass
import gc
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input
from .prism import _resolve


@dataclass
class TrainingInputs:
    swe_filled: str
    pcp_filled: str
    tmp_filled: str
    station_cells: str


#Context: This little helper clears out tensorflow/keras memory after a model
#Context: is done training. If we dont do this the computer slowly fills up its
#Context: memory and can crash when we train several models one after another.
#Context: It takes no inputs and gives nothing back, it just cleans house.
def _cleanup_keras_state():
    """
    Reset TF state between back-to-back train_* calls so we don't OOM.
    """
    # Without this the OOM killer eventually catches up with us when
    # several variants run in the same process.
    keras.backend.clear_session()
    gc.collect()


#Context: This takes a bunch of 3d arrays (time, height, width) and trims them all
#Context: down so they share the exact same height and width. Sometimes two datasets
#Context: are off by a row or column at the edges, so we cut to the smallest one so
#Context: they line up and can be stacked together without errors.
#Context: Example: an array of shape (10,142,310) and one of (10,141,309) both
#Context: come back as shape (10,141,309).
def _align_shapes(*arrays):
    """
    Crop (T, H, W) arrays to the smallest common (H, W).

    NSIDC and PRISM resolve the same bbox via different code paths
    (searchsorted vs. round-to-cell) so they can disagree by a row or
    column at the edges. Original scripts hardcoded h=142, w=310 and never
    hit this; bbox-driven extraction does, and np.stack barfs without it.
    """
    H = min(a.shape[-2] for a in arrays)
    W = min(a.shape[-1] for a in arrays)
    return tuple(a[..., :H, :W] for a in arrays)


#Context: This saves a trained model to disk so we can load and reuse it later.
#Context: If saving is turned off it just returns None and does nothing. You can
#Context: give it a custom file name, otherwise it uses a default name. It figures
#Context: out the right file extension (.keras or .h5) and returns the full path
#Context: where the model was written.
#Context: Example output: "/home/me/output/swe_model.keras"
def _save_model(
    model, output_dir, variant_default, save_model, model_format, model_filename
):
    """
    Save ``model`` and return the path, or None if disabled.

    ``model_filename`` (if given) overrides ``variant_default``; an explicit
    .keras / .h5 extension is respected as-is.
    """
    if not save_model:
        return None
    base = model_filename if model_filename else variant_default
    if base.endswith(".keras") or base.endswith(".h5"):
        path = os.path.join(output_dir, base)
    else:
        path = os.path.join(output_dir, f"{base}.{model_format}")
    model.save(path)
    print(f"[swecast] Saved trained model -> {path}")
    return path


#Context: This is the big prep step that gets all the data ready before any model
#Context: training happens. It downloads weather/snow station info, builds the data
#Context: stacks for snow water (SWE), precipitation and temperature, fills in any
#Context: missing gaps, and figures out which grid cell each station sits in.
#Context: It does NOT train anything, it just makes the files the trainers need and
#Context: returns there file paths bundled up in a TrainingInputs object.
def prepare_training_inputs(
    output_dir, *, manifest=None, cache_dir=None
) -> TrainingInputs:
    """
    Prepare inputs required for ConvLSTM training and evaluation.

    This function executes the SWECAST data-preparation workflow and generates
    the intermediate files required by the ConvLSTM training routines. The
    workflow includes station acquisition, PRISM and NSIDC stack generation,
    gap filling, and construction of station-to-grid-cell mappings used for
    station-based model evaluation.

    Parameters
    ----------
    output_dir : str | Path
        Directory where intermediate datasets, training inputs, and evaluation
        files are written. The directory is created if it does not already
        exist.
    manifest : Manifest, optional
        SWECAST manifest specifying the study period, spatial domain, data
        acquisition settings, and training configuration.
    cache_dir : str | Path, optional
        Directory containing cached NSIDC NetCDF files used to obtain the
        latitude-longitude grid required for station-cell identification.
        If not specified, defaults to ``manifest.cache_dir`` when available,
        or ``<output_dir>/.cache``.

    Returns
    -------
    TrainingInputs
        Dataclass containing paths required by the training routines:

        * ``swe_filled`` -- gap-filled SWE stack (``swe_filled.npy``).
        * ``pcp_filled`` -- gap-filled precipitation stack (``pcp_filled.npy``).
        * ``tmp_filled`` -- gap-filled temperature stack (``tmp_filled.npy``).
        * ``station_cells`` -- station-to-grid-cell mapping
          (``station_cells.npy``).

    Notes
    -----
    This function performs data acquisition and preprocessing but does not
    perform model training. It prepares the inputs required by
    ``train_swe()``, ``train_swe_pcp()``, ``train_swe_tmp()``, and
    ``train_swe_tmp_pcp()``.

    The generated workflow includes:

    1. Acquisition of station metadata.
    2. Construction of PRISM and NSIDC data stacks.
    3. Gap filling of SWE, precipitation, and temperature datasets.
    4. Filtering of stations to the study domain.
    5. Identification of station grid-cell locations for evaluation.
    """
    from pathlib import Path
    import pandas as pd
    import rasterio

    from .nsidc import build_swe_stacks, fill_stacks, fill_npy
    from .prism import build_stacks
    from .stations import get_stations, identify_station_cells

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stations_csv = get_stations(manifest, output_dir)

    # build stacks
    outputs = build_stacks(manifest, output_dir=output_dir)
    # swe stacks
    swe_outputs = build_swe_stacks(manifest, output_dir=output_dir)

    # Gap-fill SWE stacks (also fills sibling swe.npy -> swe_filled.npy)
    filled_stacks = fill_stacks(swe_outputs)

    # Gap-fill the PRISM .npy stacks too, since the multi-channel ConvLSTM
    # variants expect pcp_filled.npy and tmp_filled.npy.
    pcp_filled = fill_npy(output_dir / "pcp.npy")  # -> ./output/pcp_filled.npy
    tmp_filled = fill_npy(output_dir / "tmp.npy")  # -> ./output/tmp_filled.npy

    swe_tif = Path(filled_stacks["SWE"])
    swe_filled = swe_tif.parent / "swe_filled.npy"
    if not swe_filled.exists():
        raise FileNotFoundError(
            f"Expected {swe_filled} alongside the filled SWE GeoTIFF. "
            "Re-run build_swe_stacks (write_npy=True default) and fill_stacks."
        )

    # Find an NSIDC .nc file for the lat/lon grid used by identify_station_cells
    cache_dir = _resolve(cache_dir, manifest, "cache_dir", None)
    cache_dir = Path(cache_dir) if cache_dir else swe_tif.parent / ".cache"
    nc_files = sorted(cache_dir.glob("4km_SWE_Depth_WY*_v01.nc"))
    if not nc_files:
        raise FileNotFoundError(
            f"No NSIDC NetCDF files found in {cache_dir}. "
            "Run build_swe_stacks first to populate the cache."
        )
    nc_path = nc_files[0]

    # Filter the AWDB-style stations CSV to those inside the SWE stack's bbox,
    # then write a 3-column (id, lat, lon) CSV without header, which is what
    # identify_station_cells expects (stations[i, 1] = lat, stations[i, 2] = lon).
    with rasterio.open(swe_tif) as src:
        bounds = src.bounds  # (left, bottom, right, top)

    df = pd.read_csv(stations_csv)
    in_bbox = (
        (df["longitude"] >= bounds.left)
        & (df["longitude"] <= bounds.right)
        & (df["latitude"] >= bounds.bottom)
        & (df["latitude"] <= bounds.top)
    )
    df = df.loc[in_bbox, ["stationId", "latitude", "longitude"]]

    formatted_csv = output_dir / "stations_for_cells.csv"
    df.to_csv(formatted_csv, header=False, index=False)
    print(
        f"[swecast] Filtered {in_bbox.sum()} stations within SWE bbox -> {formatted_csv}"
    )

    station_cells = output_dir / "station_cells.npy"
    bbox = getattr(manifest, "bbox", None) if manifest is not None else None
    identify_station_cells(
        formatted_csv, nc_path, output_path=str(station_cells), bbox=bbox
    )

    return TrainingInputs(
        swe_filled=swe_filled,
        pcp_filled=pcp_filled,
        tmp_filled=tmp_filled,
        station_cells=station_cells,
    )


@dataclass
class _HParams:
    """Resolved ConvLSTM hyperparameters shared by every train_* variant."""

    num_days_train: int
    num_data_used: int
    epochs: int
    batch_size: int
    train_split: float
    swe_scaling_factor: float
    es_patience: int
    rlr_patience: int
    num_stations: int
    save_model: bool
    model_format: str
    model_filename: str


def _resolve_hparams(
    manifest,
    *,
    num_days_train,
    num_data_used,
    epochs,
    batch_size,
    train_split,
    swe_scaling_factor,
    early_stopping_patience,
    reduce_lr_patience,
    num_stations,
    save_model,
    model_format,
    model_filename,
):
    """
    Resolve the hyperparameters common to every ConvLSTM variant.

    Each value falls back to ``manifest.<field>`` when the kwarg is None, then
    to the original script default if neither is set.
    """
    return _HParams(
        # Number of previous time steps used to forecast the next day's SWE; 5
        # means day 5 is forecast from the previous 4 days of data.
        num_days_train=_resolve(num_days_train, manifest, "num_days_train", 5),
        num_data_used=_resolve(num_data_used, manifest, "num_data_used", 7300),  # 8 years (8*365)
        epochs=_resolve(epochs, manifest, "epochs", 50),
        batch_size=_resolve(batch_size, manifest, "batch_size", 16),
        train_split=_resolve(train_split, manifest, "train_split", 0.8),
        swe_scaling_factor=_resolve(swe_scaling_factor, manifest, "swe_scaling_factor", 3.5),
        es_patience=_resolve(
            early_stopping_patience, manifest, "early_stopping_patience", 10
        ),
        rlr_patience=_resolve(reduce_lr_patience, manifest, "reduce_lr_patience", 5),
        num_stations=_resolve(num_stations, manifest, "num_stations", 75),
        save_model=_resolve(save_model, manifest, "save_model", True),
        model_format=_resolve(model_format, manifest, "model_format", "keras"),
        model_filename=_resolve(model_filename, manifest, "model_filename", None),
    )


def _build_convlstm(input_shape):
    """
    Build and compile the 3-layer ConvLSTM2D network shared by every variant.

    Three ``ConvLSTM2D`` layers (32 filters, 3x3) with batch normalization
    between them, followed by a ``Conv2D`` head for the spatiotemporal output.
    The only thing that differs between variants is the channel count, which is
    carried by ``input_shape``.
    """
    inp = layers.Input(shape=input_shape)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False,
        activation="relu",
    )(x)
    x = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation="sigmoid",
        padding="same",
    )(x)
    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
    )
    return model


def _fit_convlstm(model, x_train, y_train, x_val, y_val, hp):
    """
    Fit a compiled ConvLSTM with the shared early-stopping / LR-plateau
    callbacks and return the Keras ``History``.
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=hp.es_patience
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=hp.rlr_patience
    )
    return model.fit(
        x_train,
        y_train,
        batch_size=hp.batch_size,
        epochs=hp.epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
    )


def _evaluate_stations(
    model, x_val, y_val, station_cells, output_dir, hp, tag, *, ns_filename=None
):
    """
    Compare model predictions against the in-situ stations and write outputs.

    Writes ``Actual_<tag>.npy`` and ``model_output_<tag>.npy`` (de-normalized
    to the original SWE scale) plus the per-station Nash-Sutcliffe efficiencies
    to ``ns_filename`` (default ``NS_stations_<tag>.csv``). NS is computed on
    the original SWE scale; negative predictions are clamped to zero first.
    """
    num_stations = hp.num_stations
    swe_scaling_factor = hp.swe_scaling_factor
    if ns_filename is None:
        ns_filename = f"NS_stations_{tag}.csv"

    # Predict frames for all x_val.
    y_val_prediction = model.predict(x_val)

    # read in coordinates of the stations in our study region
    latlon = np.load(station_cells).astype(int)

    # extract observed swe for the stations
    station_swe_origin = []
    y_val1 = np.squeeze(y_val)
    np.save(
        os.path.join(output_dir, f"Actual_{tag}.npy"),
        (10 ** (y_val1 * swe_scaling_factor) - 1),
    )
    for j in range(0, num_stations):
        station_swe_origin.append(y_val1[:, latlon[j, 0], latlon[j, 1]])
    station_swe_origin = np.array(station_swe_origin)

    # extract predicted swe for the stations
    station_swe_predict = []
    y_val_prediction1 = np.squeeze(y_val_prediction)
    np.save(
        os.path.join(output_dir, f"model_output_{tag}.npy"),
        (10 ** (y_val_prediction1 * swe_scaling_factor) - 1),
    )
    for j in range(0, num_stations):
        station_swe_predict.append(y_val_prediction1[:, latlon[j, 0], latlon[j, 1]])
    station_swe_predict = np.array(station_swe_predict)

    # data adjustment: change negative SWE predicts to be 0
    for j in range(0, station_swe_predict.shape[1]):
        for i in range(0, num_stations):
            if station_swe_predict[i, j] < 0:
                station_swe_predict[i, j] = 0

    # recover the SWE data to original scales, calculate NS for the stations
    station_swe_origin1 = 10 ** (swe_scaling_factor * station_swe_origin) - 1
    station_swe_predict1 = 10 ** (swe_scaling_factor * station_swe_predict) - 1

    variances = np.var(station_swe_origin1, axis=1)
    mse = ((station_swe_origin1 - station_swe_predict1) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances
    NS_stations = np.round(NS_stations, 3)

    np.savetxt(os.path.join(output_dir, ns_filename), NS_stations, delimiter=",")


#Context: This trains an AI model (a type called ConvLSTM) that looks at the last few
#Context: days of snow water (SWE) and tries to guess the next days snow. So if you
#Context: give it 4 days it guesses day 5. It loads the data, shrinks the numbers down
#Context: so theyre easier to learn from, splits them into a training pile and a
#Context: testing pile, builds the model, trains it, then saves a chart of how the
#Context: error shrank, the guesses, and an accuracy score (NS) for the real stations.
#Context: It only uses snow water here, no rain or temperature.
def train_swe(
    swe_filled,
    station_cells,
    output_dir,
    *,
    manifest=None,
    num_days_train=None,
    num_data_used=None,
    epochs=None,
    batch_size=None,
    train_split=None,
    swe_scaling_factor=None,
    early_stopping_patience=None,
    reduce_lr_patience=None,
    num_stations=None,
    save_model=None,
    model_format=None,
    model_filename=None,
):
    """
    SWE-only ConvLSTM: previous N-1 days of SWE predict day N.

    Forklift of ConvLSTM_SWE_only.py. NS is reported against the 75 in-situ
    stations from the original California study.

    Hyperparameters fall back to ``manifest.<field>`` if the kwarg is None,
    then to script defaults if neither is set.

    Writes loss_curve.{png,pdf}, Actual_swe.npy, model_output_swe.npy,
    and NS_stations.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    hp = _resolve_hparams(
        manifest,
        num_days_train=num_days_train,
        num_data_used=num_data_used,
        epochs=epochs,
        batch_size=batch_size,
        train_split=train_split,
        swe_scaling_factor=swe_scaling_factor,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        num_stations=num_stations,
        save_model=save_model,
        model_format=model_format,
        model_filename=model_filename,
    )

    # read in SWE data (cast to fp32; Keras weights are fp32 anyway, and this
    # halves the memory footprint of the windowed dataset built below)
    ds = np.load(swe_filled).astype(np.float32, copy=False)
    ds1 = ds[0 : hp.num_data_used, :, :]

    # prepare for model input
    dataset = []
    for i in range(0, hp.num_data_used - hp.num_days_train):
        dataset.append(ds1[i : i + hp.num_days_train, :, :])
    dataset = np.array(dataset)

    # split the dataset into training and validation parts
    dataset = np.expand_dims(dataset, axis=-1)
    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(hp.train_split * dataset.shape[0])]
    val_index = indexes[int(hp.train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # Normalize the data to the 0-1 range.The study used log normalization
    train_dataset = (
        np.log10(1 + train_dataset) / hp.swe_scaling_factor
    )  # +1 ensures that zero values do not cause issue and /3.5 scales values to approximately 0-1
    val_dataset = np.log10(1 + val_dataset) / hp.swe_scaling_factor

    # prepare for x and y for the model.`x` is frames 0 to n - 1, and `y` is frames n-1 to n.
    # For example, if data from 5 days were used, days 1 to 4 would be used as input features, and day 5 would be used as the target data.
    #Context: This small helper splits each little stack of days into the input (x)
    #Context: and the answer (y). x is all the days except the last, y is just the
    #Context: last day which is the thing we want the model to predict.
    #Context: Example: input [[1,2,3,4,5]] gives x=[[1,2,3,4]] and y=[[5]].
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, data.shape[1] - 1 : data.shape[1], :, :]
        return x, y

    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # this is the trickest part! for the many-to-one ConvLSTM, you have to
    # change the array shape of y_train

    #    Input shape: (batch_size, time_steps, height, width, channels)
    #    Output shape for many-to-one: (batch_size, height, width, channels)
    y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
    y_val = np.transpose(y_val, [0, 2, 3, 1, 4])

    model = _build_convlstm((None, *x_train.shape[2:]))

    start_time = time.time()
    history = _fit_convlstm(model, x_train, y_train, x_val, y_val, hp)
    end_time = time.time()
    training_time = end_time - start_time
    print("Total training time:", training_time, "seconds")
    print("Best val_loss:", min(history.history["val_loss"]))

    plt.figure(figsize=(6, 4), dpi=300)  # high resolution for journals

    plt.plot(history.history["loss"], linewidth=2)
    plt.plot(history.history["val_loss"], linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Binary Crossentropy)", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=12)

    plt.legend(["Training Loss", "Validation Loss"], fontsize=10)
    plt.grid(True, linewidth=0.4, alpha=0.6)

    plt.tight_layout()

    # save as high-quality PNG and PDF
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "loss_curve.pdf"), dpi=300)

    plt.close()

    _save_model(
        model, output_dir, "model", hp.save_model, hp.model_format, hp.model_filename
    )

    # following part aims at comparing the in-situ station SWE observations with
    # predictions from the above model over the validation period.
    _evaluate_stations(
        model,
        x_val,
        y_val,
        station_cells,
        output_dir,
        hp,
        "swe",
        ns_filename="NS_stations.csv",
    )

    _cleanup_keras_state()


#Context: Almost the same as train_swe but this one feeds the model TWO things at
#Context: once: snow water (SWE) and precipitation (PCP, basicaly rain/snowfall).
#Context: The idea is that knowing how much it precipitated helps predict tomorrows
#Context: snow better. It stacks both maps on top of each other (each stacked map is
#Context: called a channel), trains the model, and writes out its own prediction and
#Context: accuracy files with a _swe_pcp tag.
def train_swe_pcp(
    swe_filled,
    pcp_filled,
    station_cells,
    output_dir,
    *,
    manifest=None,
    num_days_train=None,
    num_data_used=None,
    epochs=None,
    batch_size=None,
    train_split=None,
    swe_scaling_factor=None,
    early_stopping_patience=None,
    reduce_lr_patience=None,
    num_stations=None,
    save_model=None,
    model_format=None,
    model_filename=None,
):
    """
    SWE + PCP variant. Forklift of ConvLSTM_SWE_PCP.py.

    Writes Actual_swe_pcp.npy, model_output_swe_pcp.npy and
    NS_stations_swe_pcp.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    hp = _resolve_hparams(
        manifest,
        num_days_train=num_days_train,
        num_data_used=num_data_used,
        epochs=epochs,
        batch_size=batch_size,
        train_split=train_split,
        swe_scaling_factor=swe_scaling_factor,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        num_stations=num_stations,
        save_model=save_model,
        model_format=model_format,
        model_filename=model_filename,
    )

    # read in SWE data
    # Cast to fp32 to halve windowed-dataset memory; Keras weights are fp32 anyway
    ds_swe = np.load(swe_filled).astype(np.float32, copy=False)
    ds_pcp = np.load(pcp_filled).astype(np.float32, copy=False)

    ds1_swe = ds_swe[0 : hp.num_data_used, :, :]
    ds1_pcp = ds_pcp[0 : hp.num_data_used, :, :]
    ds1_swe, ds1_pcp = _align_shapes(ds1_swe, ds1_pcp)
    ds1_swe = np.log10(1 + ds1_swe) / hp.swe_scaling_factor
    ds1 = np.stack((ds1_swe, ds1_pcp), axis=3)

    # prepare for model input
    dataset = []
    for i in range(0, hp.num_data_used - hp.num_days_train):
        dataset.append(ds1[i : i + hp.num_days_train, :, :, :])
    dataset = np.array(dataset)

    # split the dataset into training and validation parts
    # dataset = np.expand_dims(dataset, axis=-1)
    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(hp.train_split * dataset.shape[0])]
    val_index = indexes[int(hp.train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # Normalize the data to the 0-1 range.The study used log normalization
    # train_dataset = (
    #    np.log10(1 + train_dataset) / 3.5
    # )  # +1 ensures that zero values do not cause issue and /3.5 scales values to approximately 0-1
    # val_dataset = np.log10(1 + val_dataset) / 3.5

    # prepare for x and y for the model.`x` is frames 0 to n - 1, and `y` is frames n-1 to n.
    # For example, if data from 5 days were used, days 1 to 4 would be used as input features, and day 5 would be used as the target data.
    #Context: Same idea as before but now each day has more than one map stacked
    #Context: together (snow water plus the other thing, each map is a channel). x is
    #Context: all the days except the last with every map, and y is just the last days
    #Context: snow water map (the :1 grabs only the first map) since snow is what we
    #Context: are actualy trying to guess.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :, :]
        y = data[:, data.shape[1] - 1 : data.shape[1], :, :, :1]
        return x, y

    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # this is the trickest part! for the many-to-one ConvLSTM, you have to
    # change the array shape of y_train

    #    Input shape: (batch_size, time_steps, height, width, channels)
    #    Output shape for many-to-one: (batch_size, height, width, channels)
    y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
    y_val = np.transpose(y_val, [0, 2, 3, 1, 4])

    model = _build_convlstm((None, *x_train.shape[2:]))

    _fit_convlstm(model, x_train, y_train, x_val, y_val, hp)

    _save_model(
        model,
        output_dir,
        "model_swe_pcp",
        hp.save_model,
        hp.model_format,
        hp.model_filename,
    )

    # following part aims at comparing the in-situ station SWE observations with
    # predictions from the above model over the validation period.
    _evaluate_stations(model, x_val, y_val, station_cells, output_dir, hp, "swe_pcp")

    _cleanup_keras_state()


#Context: Like train_swe but it feeds the model snow water (SWE) together with
#Context: temperature (TMP). Temperature matters alot for snow because warm days
#Context: melt it, so giving the model temperature can help it predict tomorrows
#Context: snow. It stacks the two maps on top of each other (each one is a channel),
#Context: trains the model and saves its own prediction and accuracy files tagged
#Context: with swe_tmp.
def train_swe_tmp(
    swe_filled,
    tmp_filled,
    station_cells,
    output_dir,
    *,
    manifest=None,
    num_days_train=None,
    num_data_used=None,
    epochs=None,
    batch_size=None,
    train_split=None,
    swe_scaling_factor=None,
    early_stopping_patience=None,
    reduce_lr_patience=None,
    num_stations=None,
    save_model=None,
    model_format=None,
    model_filename=None,
):
    """
    SWE + TMP variant. Forklift of ConvLSTM_SWE_TEMP.py.

    Writes Actual_swe_tmp.npy, model_output_swe_tmp.npy and
    NS_stations_swe_tmp.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    hp = _resolve_hparams(
        manifest,
        num_days_train=num_days_train,
        num_data_used=num_data_used,
        epochs=epochs,
        batch_size=batch_size,
        train_split=train_split,
        swe_scaling_factor=swe_scaling_factor,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        num_stations=num_stations,
        save_model=save_model,
        model_format=model_format,
        model_filename=model_filename,
    )

    # read in SWE data (fp32 to halve windowed-dataset memory)
    ds_swe = np.load(swe_filled).astype(np.float32, copy=False)
    ds_tmp = np.load(tmp_filled).astype(np.float32, copy=False)

    ds1_swe = ds_swe[0 : hp.num_data_used, :, :]
    ds1_tmp = ds_tmp[0 : hp.num_data_used, :, :]
    ds1_swe, ds1_tmp = _align_shapes(ds1_swe, ds1_tmp)
    ds1_swe = np.log10(1 + ds1_swe) / hp.swe_scaling_factor
    ds1 = np.stack((ds1_swe, ds1_tmp), axis=3)

    # prepare for model input
    dataset = []
    for i in range(0, hp.num_data_used - hp.num_days_train):
        dataset.append(ds1[i : i + hp.num_days_train, :, :, :])
    dataset = np.array(dataset)

    # split the dataset into training and validation parts
    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(hp.train_split * dataset.shape[0])]
    val_index = indexes[int(hp.train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    #Context: Splits each stack of days into the input (x) and the answer (y) we want
    #Context: it to guess. x is all the earlier days with both maps (snow water and
    #Context: temperature), y is just the last days snow water (the first map, grabbed
    #Context: by :1) which is what we want to predict.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :, :]
        y = data[:, data.shape[1] - 1 : data.shape[1], :, :, :1]
        return x, y

    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # this is the trickest part! for the many-to-one ConvLSTM, you have to
    # change the array shape of y_train
    y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
    y_val = np.transpose(y_val, [0, 2, 3, 1, 4])

    model = _build_convlstm((None, *x_train.shape[2:]))

    _fit_convlstm(model, x_train, y_train, x_val, y_val, hp)

    _save_model(
        model,
        output_dir,
        "model_swe_tmp",
        hp.save_model,
        hp.model_format,
        hp.model_filename,
    )

    _evaluate_stations(model, x_val, y_val, station_cells, output_dir, hp, "swe_tmp")

    _cleanup_keras_state()


#Context: This is the kitchen-sink version: it feeds the model all three things at
#Context: once, snow water (SWE), temperature (TMP) and precipitation (PCP). The
#Context: hope is that using every clue gives the best forecast of tomorrows snow.
#Context: It stacks the three maps on top of each other (each map is a channel),
#Context: trains the model, and writes out its predictions and accuracy scores
#Context: tagged with swe_tmp_pcp.
def train_swe_tmp_pcp(
    swe_filled,
    tmp_filled,
    pcp_filled,
    station_cells,
    output_dir,
    *,
    manifest=None,
    num_days_train=None,
    num_data_used=None,
    epochs=None,
    batch_size=None,
    train_split=None,
    swe_scaling_factor=None,
    early_stopping_patience=None,
    reduce_lr_patience=None,
    num_stations=None,
    save_model=None,
    model_format=None,
    model_filename=None,
):
    """
    SWE + TMP + PCP variant. Forklift of ConvLSTM_SWE_TEMP_PCP.py.

    Writes Actual_swe_tmp_pcp.npy, model_output_swe_tmp_pcp.npy and
    NS_stations_swe_tmp_pcp.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    hp = _resolve_hparams(
        manifest,
        num_days_train=num_days_train,
        num_data_used=num_data_used,
        epochs=epochs,
        batch_size=batch_size,
        train_split=train_split,
        swe_scaling_factor=swe_scaling_factor,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        num_stations=num_stations,
        save_model=save_model,
        model_format=model_format,
        model_filename=model_filename,
    )

    # Cast to fp32 to halve windowed-dataset memory; Keras weights are fp32 anyway
    ds_swe = np.load(swe_filled).astype(np.float32, copy=False)
    ds_tmp = np.load(tmp_filled).astype(np.float32, copy=False)
    ds_pcp = np.load(pcp_filled).astype(np.float32, copy=False)

    ds1_swe = ds_swe[0 : hp.num_data_used, :, :]
    ds1_tmp = ds_tmp[0 : hp.num_data_used, :, :]
    ds1_pcp = ds_pcp[0 : hp.num_data_used, :, :]
    ds1_swe, ds1_tmp, ds1_pcp = _align_shapes(ds1_swe, ds1_tmp, ds1_pcp)
    ds1_swe = np.log10(1 + ds1_swe) / hp.swe_scaling_factor
    ds1 = np.stack((ds1_swe, ds1_tmp, ds1_pcp), axis=3)

    dataset = []
    for i in range(0, hp.num_data_used - hp.num_days_train):
        dataset.append(ds1[i : i + hp.num_days_train, :, :, :])
    dataset = np.array(dataset)

    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(hp.train_split * dataset.shape[0])]
    val_index = indexes[int(hp.train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    #Context: Splits each stack of days into the input (x) and the answer (y) we want
    #Context: it to guess. Here each day has three maps (snow water, temperature and
    #Context: precipitation). x keeps all the earlier days and every map, y is just the
    #Context: last days snow water (the first map via :1), thats the value we want the
    #Context: model to guess.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :, :]
        y = data[:, data.shape[1] - 1 : data.shape[1], :, :, :1]
        return x, y

    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # this is the trickest part! for the many-to-one ConvLSTM, you have to
    # change the array shape of y_train
    y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
    y_val = np.transpose(y_val, [0, 2, 3, 1, 4])

    model = _build_convlstm((None, *x_train.shape[2:]))

    _fit_convlstm(model, x_train, y_train, x_val, y_val, hp)

    _save_model(
        model,
        output_dir,
        "model_swe_tmp_pcp",
        hp.save_model,
        hp.model_format,
        hp.model_filename,
    )

    _evaluate_stations(
        model, x_val, y_val, station_cells, output_dir, hp, "swe_tmp_pcp"
    )

    _cleanup_keras_state()


#Context: This one is a bit diffrent: it predicts tomorrows snow water (SWE) using
#Context: ONLY temperature and precipitation as inputs, not past SWE at all. It
#Context: still loads SWE but only to use as the answer (target) the model trains
#Context: against. So its testing wheter weather alone can forecast the snow. Saves
#Context: its predictions and accuracy files tagged with tmp_pcp.
def train_tmp_pcp(
    swe_filled,
    tmp_filled,
    pcp_filled,
    station_cells,
    output_dir,
    *,
    manifest=None,
    num_days_train=None,
    num_data_used=None,
    epochs=None,
    batch_size=None,
    train_split=None,
    swe_scaling_factor=None,
    early_stopping_patience=None,
    reduce_lr_patience=None,
    num_stations=None,
    save_model=None,
    model_format=None,
    model_filename=None,
):
    """
    TMP + PCP only (no SWE inputs). Forklift of ConvLSTM_TEMP_PCP.py.

    SWE still has to be loaded since it provides the y target, but the
    input tensor is sliced to channels 1:3 before fitting.

    Writes Actual_tmp_pcp.npy, model_output_tmp_pcp.npy and
    NS_stations_tmp_pcp.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    hp = _resolve_hparams(
        manifest,
        num_days_train=num_days_train,
        num_data_used=num_data_used,
        epochs=epochs,
        batch_size=batch_size,
        train_split=train_split,
        swe_scaling_factor=swe_scaling_factor,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        num_stations=num_stations,
        save_model=save_model,
        model_format=model_format,
        model_filename=model_filename,
    )

    # Cast to fp32 to halve windowed-dataset memory; Keras weights are fp32 anyway
    ds_swe = np.load(swe_filled).astype(np.float32, copy=False)
    ds_tmp = np.load(tmp_filled).astype(np.float32, copy=False)
    ds_pcp = np.load(pcp_filled).astype(np.float32, copy=False)

    ds1_swe = ds_swe[0 : hp.num_data_used, :, :]
    ds1_tmp = ds_tmp[0 : hp.num_data_used, :, :]
    ds1_pcp = ds_pcp[0 : hp.num_data_used, :, :]
    ds1_swe, ds1_tmp, ds1_pcp = _align_shapes(ds1_swe, ds1_tmp, ds1_pcp)
    ds1_swe = np.log10(1 + ds1_swe) / hp.swe_scaling_factor
    ds1 = np.stack((ds1_swe, ds1_tmp, ds1_pcp), axis=3)

    dataset = []
    for i in range(0, hp.num_data_used - hp.num_days_train):
        dataset.append(ds1[i : i + hp.num_days_train, :, :, :])
    dataset = np.array(dataset)

    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(hp.train_split * dataset.shape[0])]
    val_index = indexes[int(hp.train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    #Context: Splits each day stack into the input (x) and the answer (y). y is the
    #Context: last days snow water (the first map via :1). Note that right after
    #Context: calling this the code trims x down to the 2nd and 3rd maps (1:3), so this
    #Context: model only feeds on temperature and precipitation and never sees snow
    #Context: water as an input itself.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :, :]
        y = data[:, data.shape[1] - 1 : data.shape[1], :, :, :1]
        return x, y

    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)
    x_train = x_train[:, :, :, :, 1:3]
    x_val = x_val[:, :, :, :, 1:3]

    # this is the trickest part! for the many-to-one ConvLSTM, you have to
    # change the array shape of y_train
    y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
    y_val = np.transpose(y_val, [0, 2, 3, 1, 4])

    model = _build_convlstm((None, *x_train.shape[2:]))

    _fit_convlstm(model, x_train, y_train, x_val, y_val, hp)

    _save_model(
        model,
        output_dir,
        "model_tmp_pcp",
        hp.save_model,
        hp.model_format,
        hp.model_filename,
    )

    _evaluate_stations(model, x_val, y_val, station_cells, output_dir, hp, "tmp_pcp")

    _cleanup_keras_state()


#Context: A models settings (like how many layers or how fast it learns) are called
#Context: hyperparameters, and picking good ones by hand is hard. This function uses
#Context: a tool called Optuna to automaticly try lots of diffrent combinations and
#Context: see which ones train the best. It runs many short trials and keeps track
#Context: of the winner, then saves some charts showing how the search went.
def optimize_hyperparameters(swe_filled, output_dir, *, manifest=None, n_trials=None):
    """
    Optuna sweep over the SWE-only ConvLSTM.

    Forklift of optimization_hyperparameters.py. The search space is kept
    deliberately small so trials fit on a single GPU; the Keras session is
    cleared between trials.

    Writes optuna_optimization_history.png, optuna_param_importance.png,
    optuna_slice_plot.png and optuna_parallel_coordinate.png into
    ``output_dir``.
    """
    import tensorflow as tf
    from tensorflow.keras import backend as K
    import optuna
    import optuna.visualization.matplotlib as opt_viz
    import gc

    n_trials = _resolve(n_trials, manifest, "n_trials", 20)

    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Optional: Force CPU for testing if GPU is unstable
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Enable GPU memory growth (prevents tensor transfer errors)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Load SWE data
    ds = np.load(swe_filled)
    num_data_used = 7300
    ds1 = ds[0:num_data_used, :, :]

    # Dataset preparation function
    #Context: This builds the training data for a given window size (seq_length).
    #Context: It slides a window across the days to make many little stacks, splits
    #Context: them into a training pile and a checking pile, shrinks the numbers down
    #Context: so theyre easier to learn from, and then seperates each stack into the
    #Context: inputs (x) and the day to predict (y).
    #Context: It hands back x_train, y_train, x_val, y_val all ready for the model.
    def create_dataset(seq_length):
        dataset = []
        for i in range(0, num_data_used - seq_length):
            dataset.append(ds1[i : i + seq_length, :, :])
        dataset = np.array(dataset)
        dataset = np.expand_dims(dataset, axis=-1)

        # Train/val split
        indexes = np.arange(dataset.shape[0])
        train_idx = indexes[: int(0.8 * dataset.shape[0])]
        val_idx = indexes[int(0.8 * dataset.shape[0]) :]

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]

        # Log normalization
        train_dataset = np.log10(1 + train_dataset) / 3.5
        val_dataset = np.log10(1 + val_dataset) / 3.5

        # X, y creation
        #Context: Same splitter as in the other functions, just living inside here.
        #Context: x is every day except the last, y is only the last day which is
        #Context: the one we want the model to predict.
        #Context: Example: input [[1,2,3,4]] gives x=[[1,2,3]] and y=[[4]].
        def create_shifted_frames(data):
            x = data[:, 0 : data.shape[1] - 1, :, :]
            y = data[:, data.shape[1] - 1 : data.shape[1], :, :]
            return x, y

        x_train, y_train = create_shifted_frames(train_dataset)
        x_val, y_val = create_shifted_frames(val_dataset)

        # Reshape y for many-to-one ConvLSTM
        y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
        y_val = np.transpose(y_val, [0, 2, 3, 1, 4])

        return x_train, y_train, x_val, y_val

    # Build ConvLSTM model for a trial
    #Context: This assembles the actual neural network for one trial. Optuna gives
    #Context: it a "trial" object that suggests settings to try, like how many layers,
    #Context: how many pattern detectors to use, the size of the little window it
    #Context: scans with, and how big a step it takes when learning. It stacks the
    #Context: layers up, adds a final layer, compiles it and returns the ready model.
    def build_model(trial, input_shape):
        x_in = keras.Input(shape=input_shape)
        num_layers = trial.suggest_int("num_layers", 1, 2)  # limit to 2 for GPU safety
        x = x_in

        for i in range(num_layers):
            filters = trial.suggest_categorical(
                f"filters_l{i+1}", [16, 32]
            )  # smaller filters
            kernel_size = trial.suggest_categorical(f"kernel_l{i+1}", [(3, 3), (5, 5)])
            return_seq = True if i < num_layers - 1 else False
            x = layers.ConvLSTM2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                return_sequences=return_seq,
                activation="relu",
            )(x)
            x = layers.BatchNormalization()(x)

        x = layers.Conv2D(
            filters=1, kernel_size=(3, 3), padding="same", activation="sigmoid"
        )(x)

        lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        model = keras.models.Model(x_in, x)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.binary_crossentropy,
        )
        return model

    # Optuna objective function
    #Context: This is the function Optuna calls over and over, once per trial. It
    #Context: builds a dataset and a model using the suggested settings, trains it,
    #Context: and returns the smallest error it got on the checking pile (lower is
    #Context: better). Optuna uses that number to decide which settings worked well.
    #Context: If the GPU runs out of memory it just skips that trial by returning
    #Context: infinity.
    def objective(trial):
        # Suggest sequence length
        seq_length = trial.suggest_int(
            "seq_length", 3, 4
        )  # smaller seq_length for GPU safety
        x_train, y_train, x_val, y_val = create_dataset(seq_length)

        model = build_model(trial, input_shape=x_train.shape[1:])

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3
        )

        # Batch size as hyperparameter
        batch_size = trial.suggest_categorical("batch_size", [4, 8])

        try:
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                batch_size=batch_size,
                epochs=50,
                callbacks=[early_stopping, reduce_lr],
                verbose=0,
            )
        except tf.errors.ResourceExhaustedError:
            K.clear_session()
            gc.collect()
            return float("inf")  # skip trial if GPU runs out of memory

        val_loss = min(history.history["val_loss"])

        # Clear GPU memory
        K.clear_session()
        gc.collect()

        return val_loss

    # Run Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_trial.params)

    # Save figure helper
    #Context: A small helper that saves an Optuna chart to an image file. Optuna can
    #Context: hand back the plot in a few diffrent shapes (a single axes, an array
    #Context: of axes, or a whole figure), so this sorts out which one it got, digs
    #Context: out the underlying figure and writes it to disk as the given filename.
    def save_plot(ax, filename):
        # Optuna's plot_* helpers return Axes, an ndarray of Axes, or a
        # Figure depending on the call. Coerce all three to a Figure.
        # Case 1: numpy array of axes
        if isinstance(ax, np.ndarray):
            fig = ax.ravel()[0].get_figure()

        # Case 2: a single axes object
        elif hasattr(ax, "get_figure"):
            fig = ax.get_figure()

        # Case 3: a figure object
        else:
            fig = ax

        fig.set_size_inches(6, 4)
        fig.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 1. Optimization history
    ax1 = opt_viz.plot_optimization_history(study)
    save_plot(ax1, os.path.join(output_dir, "optuna_optimization_history.png"))

    # 2. Hyperparameter importance
    ax2 = opt_viz.plot_param_importances(study)
    save_plot(ax2, os.path.join(output_dir, "optuna_param_importance.png"))

    # 3. Slice plot
    ax3 = opt_viz.plot_slice(study)
    save_plot(ax3, os.path.join(output_dir, "optuna_slice_plot.png"))

    # 4. Parallel coordinate plot
    ax4 = opt_viz.plot_parallel_coordinate(study)
    save_plot(ax4, os.path.join(output_dir, "optuna_parallel_coordinate.png"))

    return study
