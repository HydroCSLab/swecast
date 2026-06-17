"""ConvLSTM training and hyperparameter tuning.

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
    station_cells: str


def _cleanup_keras_state():
    """Reset TF state between back-to-back train_* calls so we don't OOM."""
    # Without this the OOM killer eventually catches up with us when
    # several variants run in the same process.
    keras.backend.clear_session()
    gc.collect()


def _align_shapes(*arrays):
    """Crop (T, H, W) arrays to the smallest common (H, W).

    NSIDC and PRISM resolve the same bbox via different code paths
    (searchsorted vs. round-to-cell) so they can disagree by a row or
    column at the edges. Original scripts hardcoded h=142, w=310 and never
    hit this; bbox-driven extraction does, and np.stack barfs without it.
    """
    H = min(a.shape[-2] for a in arrays)
    W = min(a.shape[-1] for a in arrays)
    return tuple(a[..., :H, :W] for a in arrays)


def _save_model(model, output_dir, variant_default, save_model, model_format, model_filename):
    """Save ``model`` and return the path, or None if disabled.

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


def prepare_training_inputs(filled_stacks, stations_csv, output_dir, *, manifest=None,
                            cache_dir=None) -> TrainingInputs:
    """
    Prepare inputs required for ConvLSTM training and evaluation.

    This function bridges the Manifest-driven data acquisition and preprocessing
    workflow with the model training routines. It validates the presence of the
    filled SWE stack, prepares station metadata, filters stations to the SWE
    domain, and generates the station-to-grid-cell mapping used for station-based
    evaluation.

    Parameters
    ----------
    filled_stacks : dict
        Variable-to-file mapping returned by ``fill_stacks``. Must contain
        the key ``"SWE"`` corresponding to the filled SWE GeoTIFF stack.
        A sibling ``swe_filled.npy`` file is expected to exist alongside
        the GeoTIFF.
    stations_csv : str | Path
        CSV file containing station identifiers and coordinates, typically
        produced by ``stations.stations_to_csv``. The file must contain
        ``stationId``, ``latitude``, and ``longitude`` columns. Stations
        outside the SWE stack extent are excluded.
    output_dir : str | Path
        Directory where intermediate training inputs are written, including
        filtered station files and station-cell mappings.
    manifest : Manifest, optional
        SWECAST manifest used to resolve configuration values such as the
        study-area bounding box and cache directory.
    cache_dir : str | Path, optional
        Directory containing cached NSIDC NetCDF files used to obtain the
        latitude-longitude grid for station-cell identification. If not
        specified, defaults to ``manifest.cache_dir`` when available, or
        ``<swe_dir>/.cache``.

    Returns
    -------
    TrainingInputs
        Dataclass containing paths required by the training routines:

        * ``swe_filled`` -- path to ``swe_filled.npy``.
        * ``station_cells`` -- path to ``station_cells.npy``.

    Notes
    -----
    This function does not perform model training. It prepares the auxiliary
    inputs required by ``train_swe()``, ``train_swe_pcp()``,
    ``train_swe_tmp()``, and ``train_swe_tmp_pcp()``.
    """
    from pathlib import Path
    import pandas as pd
    import rasterio

    from .stations import identify_station_cells

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        (df["longitude"] >= bounds.left) & (df["longitude"] <= bounds.right)
        & (df["latitude"] >= bounds.bottom) & (df["latitude"] <= bounds.top)
    )
    df = df.loc[in_bbox, ["stationId", "latitude", "longitude"]]

    formatted_csv = output_dir / "stations_for_cells.csv"
    df.to_csv(formatted_csv, header=False, index=False)
    print(f"[swecast] Filtered {in_bbox.sum()} stations within SWE bbox -> {formatted_csv}")

    station_cells = output_dir / "station_cells.npy"
    bbox = getattr(manifest, "bbox", None) if manifest is not None else None
    identify_station_cells(formatted_csv, nc_path, output_path=str(station_cells), bbox=bbox)

    return TrainingInputs(
            swe_filled = swe_filled,
            station_cells = station_cells)


def train_swe(swe_filled, station_cells, output_dir, *, manifest=None,
              num_days_train=None, num_data_used=None, epochs=None,
              batch_size=None, train_split=None, log_norm_divisor=None,
              early_stopping_patience=None, reduce_lr_patience=None,
              num_stations=None, save_model=None, model_format=None,
              model_filename=None):
    """SWE-only ConvLSTM: previous N-1 days of SWE predict day N.

    Forklift of ConvLSTM_SWE_only.py. NS is reported against the 75 in-situ
    stations from the original California study.

    Hyperparameters fall back to ``manifest.<field>`` if the kwarg is None,
    then to script defaults if neither is set.

    Writes loss_curve.{png,pdf}, Actual_swe.npy, model_output_swe.npy,
    and NS_stations.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    num_days_train = _resolve(num_days_train, manifest, "num_days_train", 5)  # Number of previous time steps to be used forecast next day SWE 5 means it will forecast 5th day SWE using previous 4 days data
    num_data_used = _resolve(num_data_used, manifest, "num_data_used", 7300)  # 8 years of data 8*365
    epochs = _resolve(epochs, manifest, "epochs", 50)
    batch_size = _resolve(batch_size, manifest, "batch_size", 16)
    train_split = _resolve(train_split, manifest, "train_split", 0.8)
    log_norm_divisor = _resolve(log_norm_divisor, manifest, "log_norm_divisor", 3.5)
    es_patience = _resolve(early_stopping_patience, manifest, "early_stopping_patience", 10)
    rlr_patience = _resolve(reduce_lr_patience, manifest, "reduce_lr_patience", 5)
    num_stations = _resolve(num_stations, manifest, "num_stations", 75)
    save_model = _resolve(save_model, manifest, "save_model", True)
    model_format = _resolve(model_format, manifest, "model_format", "keras")
    model_filename = _resolve(model_filename, manifest, "model_filename", None)


    # read in SWE data (cast to fp32; Keras weights are fp32 anyway, and this
    # halves the memory footprint of the windowed dataset built below)
    ds = np.load(swe_filled).astype(np.float32, copy=False)
    ds1 = ds[0:num_data_used, :, :]


    # prepare for model input
    dataset = []
    for i in range(0, num_data_used - num_days_train):
        dataset.append(ds1[i : i + num_days_train, :, :])
    dataset = np.array(dataset)


    # split the dataset into training and validation parts
    dataset = np.expand_dims(dataset, axis=-1)
    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(train_split * dataset.shape[0])]
    val_index = indexes[int(train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # Normalize the data to the 0-1 range.The study used log normalization
    train_dataset = (
        np.log10(1 + train_dataset) / log_norm_divisor
    )  # +1 ensures that zero values do not cause issue and /3.5 scales values to approximately 0-1
    val_dataset = np.log10(1 + val_dataset) / log_norm_divisor


    # prepare for x and y for the model.`x` is frames 0 to n - 1, and `y` is frames n-1 to n.
    # For example, if data from 5 days were used, days 1 to 4 would be used as input features, and day 5 would be used as the target data.
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
    # y_train = y_train[:, -1, :, :]
    # y_val = y_val[:, -1, :, :]

    # y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
    # y_val = np.transpose(y_val, [0, 2, 3, 1, 4])


    # model design
    inp = layers.Input(shape=(None, *x_train.shape[2:]))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv2D` layer for the spatiotemporal outputs.
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

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
    )


    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=rlr_patience)

    start_time = time.time()

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
    )
    end_time = time.time()
    training_time = end_time - start_time
    print("Total training time:", training_time, "seconds")
    print("Best val_loss:", min(history.history["val_loss"]))

    plt.figure(figsize=(6, 4), dpi=300)   # high resolution for journals

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

    _save_model(model, output_dir, "model", save_model, model_format, model_filename)

    # following part aims at comparing 75 station SWE observations with
    # predictions from the above model over 580 days in validation data.

    # Predict frames for all x_val.

    y_val_prediction = model.predict(x_val)
    # print(y_val_prediction.shape)


    # read in coordinates of the 75 stations in our study region
    latlon = np.load(station_cells)
    latlon = latlon.astype(int)


    # extract observed swe for the 75 stations
    station_swe_origin = []
    y_val1 = np.squeeze(y_val)
    np.save(os.path.join(output_dir, "Actual_swe.npy"),(10**(y_val1*log_norm_divisor)-1))#yvalue use for model develop save to plot and visualize contrast with predicted
    for j in range(0, num_stations):
        station_swe_origin.append(y_val1[:, latlon[j, 0], latlon[j, 1]])

    station_swe_origin = np.array(station_swe_origin)


    # extract predicted swe for the 75 stations
    station_swe_predict = []
    y_val_prediction1 = np.squeeze(y_val_prediction)
    np.save(os.path.join(output_dir, "model_output_swe.npy"),(10**(y_val_prediction1*log_norm_divisor)-1)) #yvalue predicted
    for j in range(0, num_stations):
        station_swe_predict.append(y_val_prediction1[:, latlon[j, 0], latlon[j, 1]])
    station_swe_predict = np.array(station_swe_predict)
    # calculate NS for 75 stations over testing, using transformed data.

    for j in range(0, station_swe_predict.shape[1]):
        for i in range(0, num_stations):
            if station_swe_predict[i, j] < 0:
                station_swe_predict[i, j] = 0

    variances = np.var(station_swe_origin, axis=1)
    mse = ((station_swe_origin - station_swe_predict) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances


    """
    # data adjustment: change negative SWE predicts to be 0
    for j in range(0, 580):
        for i in range(0, 75):
            if station_swe_predict[i, j] < 0:
                station_swe_predict[i, j] = 0


    """

    # recover the SWE data to original scales, calculate NS for 75 stations

    station_swe_origin1 = []
    station_swe_predict1 = []
    # station_swe_origin1 = np.exp2(np.log2(10) * 3.5 * station_swe_origin) - 1

    # station_swe_predict1 = np.exp2(np.log2(10) * 3.5 * station_swe_predict) - 1


    station_swe_origin1 = 10 ** (log_norm_divisor * station_swe_origin) - 1

    station_swe_predict1 = 10 ** (log_norm_divisor * station_swe_predict) - 1

    variances = np.var(station_swe_origin1, axis=1)
    mse = ((station_swe_origin1 - station_swe_predict1) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances
    NS_stations = np.round(NS_stations,3)

    np.savetxt(os.path.join(output_dir, "NS_stations.csv"), NS_stations, delimiter=",")

    _cleanup_keras_state()


def train_swe_pcp(swe_filled, pcp_filled, station_cells, output_dir, *, manifest=None,
                  num_days_train=None, num_data_used=None,
                  epochs=None, batch_size=None,
                  train_split=None, log_norm_divisor=None,
                  early_stopping_patience=None, reduce_lr_patience=None,
                  num_stations=None,
                  save_model=None, model_format=None, model_filename=None):
    """SWE + PCP variant. Forklift of ConvLSTM_SWE_PCP.py.

    Writes Actual_swe_pcp.npy, model_output_swe_pcp.npy and
    NS_stations_swe_pcp.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    num_days_train = _resolve(num_days_train, manifest, "num_days_train", 5)  # Number of previous time steps to be used forecast next day SWE 5 means it will forecast 5th day SWE using previous 4 days data
    num_data_used = _resolve(num_data_used, manifest, "num_data_used", 7300)  # 8 years of data 8*365
    epochs = _resolve(epochs, manifest, "epochs", 50)
    batch_size = _resolve(batch_size, manifest, "batch_size", 16)
    train_split = _resolve(train_split, manifest, "train_split", 0.8)
    log_norm_divisor = _resolve(log_norm_divisor, manifest, "log_norm_divisor", 3.5)
    es_patience = _resolve(early_stopping_patience, manifest, "early_stopping_patience", 10)
    rlr_patience = _resolve(reduce_lr_patience, manifest, "reduce_lr_patience", 5)
    num_stations = _resolve(num_stations, manifest, "num_stations", 75)
    save_model = _resolve(save_model, manifest, "save_model", True)
    model_format = _resolve(model_format, manifest, "model_format", "keras")
    model_filename = _resolve(model_filename, manifest, "model_filename", None)


    # read in SWE data
    # Cast to fp32 to halve windowed-dataset memory; Keras weights are fp32 anyway
    ds_swe = np.load(swe_filled).astype(np.float32, copy=False)
    ds_pcp = np.load(pcp_filled).astype(np.float32, copy=False)

    ds1_swe = ds_swe[0:num_data_used, :, :]
    ds1_pcp = ds_pcp[0:num_data_used, :, :]
    ds1_swe, ds1_pcp = _align_shapes(ds1_swe, ds1_pcp)
    ds1_swe=np.log10(1 + ds1_swe) / log_norm_divisor
    ds1 = np.stack((ds1_swe,ds1_pcp),axis=3)

    # prepare for model input
    dataset = []
    for i in range(0, num_data_used - num_days_train):
        dataset.append(ds1[i : i + num_days_train, :, :, :])
    dataset = np.array(dataset)


    # split the dataset into training and validation parts
    #dataset = np.expand_dims(dataset, axis=-1)
    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(train_split * dataset.shape[0])]
    val_index = indexes[int(train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # Normalize the data to the 0-1 range.The study used log normalization
    #train_dataset = (
    #    np.log10(1 + train_dataset) / 3.5
    #)  # +1 ensures that zero values do not cause issue and /3.5 scales values to approximately 0-1
    #val_dataset = np.log10(1 + val_dataset) / 3.5


    # prepare for x and y for the model.`x` is frames 0 to n - 1, and `y` is frames n-1 to n.
    # For example, if data from 5 days were used, days 1 to 4 would be used as input features, and day 5 would be used as the target data.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :,:]
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
    # y_train = y_train[:, -1, :, :]
    # y_val = y_val[:, -1, :, :]

    # y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
    # y_val = np.transpose(y_val, [0, 2, 3, 1, 4])


    # model design
    inp = layers.Input(shape=(None, *x_train.shape[2:]))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv2D` layer for the spatiotemporal outputs.
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

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
    )


    # Define some callbacks to improve training. To get better results
    # these parameters may be modified
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=rlr_patience)

    # Fit the model to the training data.
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
    )

    _save_model(model, output_dir, "model_swe_pcp", save_model, model_format, model_filename)

    # following part aims at comparing 75 station SWE observations with
    # predictions from the above model over 580 days in validation data.

    # Predict frames for all x_val.

    y_val_prediction = model.predict(x_val)
    # print(y_val_prediction.shape)


    # read in coordinates of the 75 stations in our study region
    latlon = np.load(station_cells)
    latlon = latlon.astype(int)


    # extract observed swe for the 75 stations
    station_swe_origin = []
    y_val1 = np.squeeze(y_val)
    np.save(os.path.join(output_dir, "Actual_swe_pcp.npy"),(10**(y_val1*log_norm_divisor)-1))#yvalue use for model develop save to plot and visualize contrast with predicted
    for j in range(0, num_stations):
        station_swe_origin.append(y_val1[:, latlon[j, 0], latlon[j, 1]])

    station_swe_origin = np.array(station_swe_origin)


    # extract predicted swe for the 75 stations
    station_swe_predict = []
    y_val_prediction1 = np.squeeze(y_val_prediction)
    np.save(os.path.join(output_dir, "model_output_swe_pcp.npy"),(10**(y_val_prediction1*log_norm_divisor)-1)) #yvalue predicted
    for j in range(0, num_stations):
        station_swe_predict.append(y_val_prediction1[:, latlon[j, 0], latlon[j, 1]])
    station_swe_predict = np.array(station_swe_predict)
    # calculate NS for 75 stations over testing, using transformed data.

    for j in range(0, station_swe_predict.shape[1]):
        for i in range(0, num_stations):
            if station_swe_predict[i, j] < 0:
                station_swe_predict[i, j] = 0

    variances = np.var(station_swe_origin, axis=1)
    mse = ((station_swe_origin - station_swe_predict) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances


    # recover the SWE data to original scales, calculate NS for 75 stations

    station_swe_origin1 = 10 ** (log_norm_divisor * station_swe_origin) - 1
    station_swe_predict1 = 10 ** (log_norm_divisor * station_swe_predict) - 1

    variances = np.var(station_swe_origin1, axis=1)
    mse = ((station_swe_origin1 - station_swe_predict1) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances
    NS_stations = np.round(NS_stations,3)

    np.savetxt(os.path.join(output_dir, "NS_stations_swe_pcp.csv"), NS_stations, delimiter=",")

    _cleanup_keras_state()


def train_swe_tmp(swe_filled, tmp_filled, station_cells, output_dir, *, manifest=None,
                   num_days_train=None, num_data_used=None,
                   epochs=None, batch_size=None,
                   train_split=None, log_norm_divisor=None,
                   early_stopping_patience=None, reduce_lr_patience=None,
                   num_stations=None,
                   save_model=None, model_format=None, model_filename=None):
    """SWE + TMP variant. Forklift of ConvLSTM_SWE_TEMP.py.

    Writes Actual_swe_tmp.npy, model_output_swe_tmp.npy and
    NS_stations_swe_tmp.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    num_days_train = _resolve(num_days_train, manifest, "num_days_train", 5)
    num_data_used = _resolve(num_data_used, manifest, "num_data_used", 7300)
    epochs = _resolve(epochs, manifest, "epochs", 50)
    batch_size = _resolve(batch_size, manifest, "batch_size", 16)
    train_split = _resolve(train_split, manifest, "train_split", 0.8)
    log_norm_divisor = _resolve(log_norm_divisor, manifest, "log_norm_divisor", 3.5)
    es_patience = _resolve(early_stopping_patience, manifest, "early_stopping_patience", 10)
    rlr_patience = _resolve(reduce_lr_patience, manifest, "reduce_lr_patience", 5)
    num_stations = _resolve(num_stations, manifest, "num_stations", 75)
    save_model = _resolve(save_model, manifest, "save_model", True)
    model_format = _resolve(model_format, manifest, "model_format", "keras")
    model_filename = _resolve(model_filename, manifest, "model_filename", None)


    # read in SWE data (fp32 to halve windowed-dataset memory)
    ds_swe = np.load(swe_filled).astype(np.float32, copy=False)
    ds_tmp = np.load(tmp_filled).astype(np.float32, copy=False)

    ds1_swe = ds_swe[0:num_data_used, :, :]
    ds1_tmp = ds_tmp[0:num_data_used, :, :]
    ds1_swe, ds1_tmp = _align_shapes(ds1_swe, ds1_tmp)
    ds1_swe=np.log10(1 + ds1_swe) / log_norm_divisor
    ds1 = np.stack((ds1_swe,ds1_tmp),axis=3)

    # prepare for model input
    dataset = []
    for i in range(0, num_data_used - num_days_train):
        dataset.append(ds1[i : i + num_days_train, :, :, :])
    dataset = np.array(dataset)


    # split the dataset into training and validation parts
    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(train_split * dataset.shape[0])]
    val_index = indexes[int(train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]


    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :,:]
        y = data[:, data.shape[1] - 1 : data.shape[1], :, :, :1]
        return x, y


    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # this is the trickest part! for the many-to-one ConvLSTM, you have to
    # change the array shape of y_train
    y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
    y_val = np.transpose(y_val, [0, 2, 3, 1, 4])

    inp = layers.Input(shape=(None, *x_train.shape[2:]))

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

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=rlr_patience)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
    )

    _save_model(model, output_dir, "model_swe_tmp", save_model, model_format, model_filename)

    y_val_prediction = model.predict(x_val)

    # read in coordinates of the 75 stations in our study region
    latlon = np.load(station_cells)
    latlon = latlon.astype(int)

    station_swe_origin = []
    y_val1 = np.squeeze(y_val)
    np.save(os.path.join(output_dir, "Actual_swe_tmp.npy"),(10**(y_val1*log_norm_divisor)-1))
    for j in range(0, num_stations):
        station_swe_origin.append(y_val1[:, latlon[j, 0], latlon[j, 1]])

    station_swe_origin = np.array(station_swe_origin)

    station_swe_predict = []
    y_val_prediction1 = np.squeeze(y_val_prediction)
    np.save(os.path.join(output_dir, "model_output_swe_tmp.npy"),(10**(y_val_prediction1*log_norm_divisor)-1))
    for j in range(0, num_stations):
        station_swe_predict.append(y_val_prediction1[:, latlon[j, 0], latlon[j, 1]])
    station_swe_predict = np.array(station_swe_predict)

    for j in range(0, station_swe_predict.shape[1]):
        for i in range(0, num_stations):
            if station_swe_predict[i, j] < 0:
                station_swe_predict[i, j] = 0

    variances = np.var(station_swe_origin, axis=1)
    mse = ((station_swe_origin - station_swe_predict) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances

    station_swe_origin1 = 10 ** (log_norm_divisor * station_swe_origin) - 1
    station_swe_predict1 = 10 ** (log_norm_divisor * station_swe_predict) - 1

    variances = np.var(station_swe_origin1, axis=1)
    mse = ((station_swe_origin1 - station_swe_predict1) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances
    NS_stations = np.round(NS_stations,3)

    np.savetxt(os.path.join(output_dir, "NS_stations_swe_tmp.csv"), NS_stations, delimiter=",")

    _cleanup_keras_state()


def train_swe_tmp_pcp(swe_filled, tmp_filled, pcp_filled, station_cells, output_dir, *, manifest=None,
                       num_days_train=None, num_data_used=None,
                       epochs=None, batch_size=None,
                       train_split=None, log_norm_divisor=None,
                       early_stopping_patience=None, reduce_lr_patience=None,
                       num_stations=None,
                       save_model=None, model_format=None, model_filename=None):
    """SWE + TMP + PCP variant. Forklift of ConvLSTM_SWE_TEMP_PCP.py.

    Writes Actual_swe_tmp_pcp.npy, model_output_swe_tmp_pcp.npy and
    NS_stations_swe_tmp_pcp.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    num_days_train = _resolve(num_days_train, manifest, "num_days_train", 5)
    num_data_used = _resolve(num_data_used, manifest, "num_data_used", 7300)
    epochs = _resolve(epochs, manifest, "epochs", 50)
    batch_size = _resolve(batch_size, manifest, "batch_size", 16)
    train_split = _resolve(train_split, manifest, "train_split", 0.8)
    log_norm_divisor = _resolve(log_norm_divisor, manifest, "log_norm_divisor", 3.5)
    es_patience = _resolve(early_stopping_patience, manifest, "early_stopping_patience", 10)
    rlr_patience = _resolve(reduce_lr_patience, manifest, "reduce_lr_patience", 5)
    num_stations = _resolve(num_stations, manifest, "num_stations", 75)
    save_model = _resolve(save_model, manifest, "save_model", True)
    model_format = _resolve(model_format, manifest, "model_format", "keras")
    model_filename = _resolve(model_filename, manifest, "model_filename", None)

    # Cast to fp32 to halve windowed-dataset memory; Keras weights are fp32 anyway
    ds_swe = np.load(swe_filled).astype(np.float32, copy=False)
    ds_tmp = np.load(tmp_filled).astype(np.float32, copy=False)
    ds_pcp = np.load(pcp_filled).astype(np.float32, copy=False)

    ds1_swe = ds_swe[0:num_data_used, :, :]
    ds1_tmp = ds_tmp[0:num_data_used, :, :]
    ds1_pcp = ds_pcp[0:num_data_used, :, :]
    ds1_swe, ds1_tmp, ds1_pcp = _align_shapes(ds1_swe, ds1_tmp, ds1_pcp)
    ds1_swe=np.log10(1 + ds1_swe) / log_norm_divisor
    ds1 = np.stack((ds1_swe,ds1_tmp,ds1_pcp),axis=3)

    dataset = []
    for i in range(0, num_data_used - num_days_train):
        dataset.append(ds1[i : i + num_days_train, :, :, :])
    dataset = np.array(dataset)

    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(train_split * dataset.shape[0])]
    val_index = indexes[int(train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :,:]
        y = data[:, data.shape[1] - 1 : data.shape[1], :, :, :1]
        return x, y

    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # this is the trickest part! for the many-to-one ConvLSTM, you have to
    # change the array shape of y_train
    y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
    y_val = np.transpose(y_val, [0, 2, 3, 1, 4])

    inp = layers.Input(shape=(None, *x_train.shape[2:]))

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

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=rlr_patience)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
    )

    _save_model(model, output_dir, "model_swe_tmp_pcp", save_model, model_format, model_filename)

    y_val_prediction = model.predict(x_val)

    latlon = np.load(station_cells)
    latlon = latlon.astype(int)

    station_swe_origin = []
    y_val1 = np.squeeze(y_val)
    np.save(os.path.join(output_dir, "Actual_swe_tmp_pcp.npy"),(10**(y_val1*log_norm_divisor)-1))
    for j in range(0, num_stations):
        station_swe_origin.append(y_val1[:, latlon[j, 0], latlon[j, 1]])

    station_swe_origin = np.array(station_swe_origin)

    station_swe_predict = []
    y_val_prediction1 = np.squeeze(y_val_prediction)
    np.save(os.path.join(output_dir, "model_output_swe_tmp_pcp.npy"),(10**(y_val_prediction1*log_norm_divisor)-1))
    for j in range(0, num_stations):
        station_swe_predict.append(y_val_prediction1[:, latlon[j, 0], latlon[j, 1]])
    station_swe_predict = np.array(station_swe_predict)

    for j in range(0, station_swe_predict.shape[1]):
        for i in range(0, num_stations):
            if station_swe_predict[i, j] < 0:
                station_swe_predict[i, j] = 0

    variances = np.var(station_swe_origin, axis=1)
    mse = ((station_swe_origin - station_swe_predict) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances

    station_swe_origin1 = 10 ** (log_norm_divisor * station_swe_origin) - 1
    station_swe_predict1 = 10 ** (log_norm_divisor * station_swe_predict) - 1

    variances = np.var(station_swe_origin1, axis=1)
    mse = ((station_swe_origin1 - station_swe_predict1) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances
    NS_stations = np.round(NS_stations,3)

    np.savetxt(os.path.join(output_dir, "NS_stations_swe_tmp_pcp.csv"), NS_stations, delimiter=",")

    _cleanup_keras_state()


def train_tmp_pcp(swe_filled, tmp_filled, pcp_filled, station_cells, output_dir, *, manifest=None,
                   num_days_train=None, num_data_used=None,
                   epochs=None, batch_size=None,
                   train_split=None, log_norm_divisor=None,
                   early_stopping_patience=None, reduce_lr_patience=None,
                   num_stations=None,
                   save_model=None, model_format=None, model_filename=None):
    """TMP + PCP only (no SWE inputs). Forklift of ConvLSTM_TEMP_PCP.py.

    SWE still has to be loaded since it provides the y target, but the
    input tensor is sliced to channels 1:3 before fitting.

    Writes Actual_tmp_pcp.npy, model_output_tmp_pcp.npy and
    NS_stations_tmp_pcp.csv into ``output_dir``.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    num_days_train = _resolve(num_days_train, manifest, "num_days_train", 5)
    num_data_used = _resolve(num_data_used, manifest, "num_data_used", 7300)
    epochs = _resolve(epochs, manifest, "epochs", 50)
    batch_size = _resolve(batch_size, manifest, "batch_size", 16)
    train_split = _resolve(train_split, manifest, "train_split", 0.8)
    log_norm_divisor = _resolve(log_norm_divisor, manifest, "log_norm_divisor", 3.5)
    es_patience = _resolve(early_stopping_patience, manifest, "early_stopping_patience", 10)
    rlr_patience = _resolve(reduce_lr_patience, manifest, "reduce_lr_patience", 5)
    num_stations = _resolve(num_stations, manifest, "num_stations", 75)
    save_model = _resolve(save_model, manifest, "save_model", True)
    model_format = _resolve(model_format, manifest, "model_format", "keras")
    model_filename = _resolve(model_filename, manifest, "model_filename", None)

    # Cast to fp32 to halve windowed-dataset memory; Keras weights are fp32 anyway
    ds_swe = np.load(swe_filled).astype(np.float32, copy=False)
    ds_tmp = np.load(tmp_filled).astype(np.float32, copy=False)
    ds_pcp = np.load(pcp_filled).astype(np.float32, copy=False)

    ds1_swe = ds_swe[0:num_data_used, :, :]
    ds1_tmp = ds_tmp[0:num_data_used, :, :]
    ds1_pcp = ds_pcp[0:num_data_used, :, :]
    ds1_swe, ds1_tmp, ds1_pcp = _align_shapes(ds1_swe, ds1_tmp, ds1_pcp)
    ds1_swe=np.log10(1 + ds1_swe) / log_norm_divisor
    ds1 = np.stack((ds1_swe,ds1_tmp,ds1_pcp),axis=3)

    dataset = []
    for i in range(0, num_data_used - num_days_train):
        dataset.append(ds1[i : i + num_days_train, :, :, :])
    dataset = np.array(dataset)

    indexes = np.arange(dataset.shape[0])
    train_index = indexes[: int(train_split * dataset.shape[0])]
    val_index = indexes[int(train_split * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :,:]
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

    inp = layers.Input(shape=(None, *x_train.shape[2:]))

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

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=rlr_patience)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
    )

    _save_model(model, output_dir, "model_tmp_pcp", save_model, model_format, model_filename)

    y_val_prediction = model.predict(x_val)

    latlon = np.load(station_cells)
    latlon = latlon.astype(int)

    station_swe_origin = []
    y_val1 = np.squeeze(y_val)
    np.save(os.path.join(output_dir, "Actual_tmp_pcp.npy"),(10**(y_val1*log_norm_divisor)-1))
    for j in range(0, num_stations):
        station_swe_origin.append(y_val1[:, latlon[j, 0], latlon[j, 1]])

    station_swe_origin = np.array(station_swe_origin)

    station_swe_predict = []
    y_val_prediction1 = np.squeeze(y_val_prediction)
    np.save(os.path.join(output_dir, "model_output_tmp_pcp.npy"),(10**(y_val_prediction1*log_norm_divisor)-1))
    for j in range(0, num_stations):
        station_swe_predict.append(y_val_prediction1[:, latlon[j, 0], latlon[j, 1]])
    station_swe_predict = np.array(station_swe_predict)

    for j in range(0, station_swe_predict.shape[1]):
        for i in range(0, num_stations):
            if station_swe_predict[i, j] < 0:
                station_swe_predict[i, j] = 0

    variances = np.var(station_swe_origin, axis=1)
    mse = ((station_swe_origin - station_swe_predict) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances

    station_swe_origin1 = 10 ** (log_norm_divisor * station_swe_origin) - 1
    station_swe_predict1 = 10 ** (log_norm_divisor * station_swe_predict) - 1

    variances = np.var(station_swe_origin1, axis=1)
    mse = ((station_swe_origin1 - station_swe_predict1) ** 2).mean(axis=1)
    NS_stations = 1 - mse / variances
    NS_stations = np.round(NS_stations,3)

    np.savetxt(os.path.join(output_dir, "NS_stations_tmp_pcp.csv"), NS_stations, delimiter=",")

    _cleanup_keras_state()


def optimize_hyper_parameters(swe_filled, output_dir, *, manifest=None, n_trials=None):
    """Optuna sweep over the SWE-only ConvLSTM.

    Forklift of optimization_hyper_parameters.py. The search space is kept
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

    # -------------------------------
    # Optional: Force CPU for testing if GPU is unstable
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # -------------------------------

    # -------------------------------
    # Enable GPU memory growth (prevents tensor transfer errors)
    # -------------------------------
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # -------------------------------
    # Load SWE data
    # -------------------------------
    ds = np.load(swe_filled)
    num_data_used = 7300
    ds1 = ds[0:num_data_used, :, :]

    # -------------------------------
    # Dataset preparation function
    # -------------------------------
    def create_dataset(seq_length):
        dataset = []
        for i in range(0, num_data_used - seq_length):
            dataset.append(ds1[i: i + seq_length, :, :])
        dataset = np.array(dataset)
        dataset = np.expand_dims(dataset, axis=-1)

        # Train/val split
        indexes = np.arange(dataset.shape[0])
        train_idx = indexes[: int(0.8 * dataset.shape[0])]
        val_idx = indexes[int(0.8 * dataset.shape[0]):]

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]

        # Log normalization
        train_dataset = np.log10(1 + train_dataset) / 3.5
        val_dataset = np.log10(1 + val_dataset) / 3.5

        # X, y creation
        def create_shifted_frames(data):
            x = data[:, 0: data.shape[1]-1, :, :]
            y = data[:, data.shape[1]-1: data.shape[1], :, :]
            return x, y

        x_train, y_train = create_shifted_frames(train_dataset)
        x_val, y_val = create_shifted_frames(val_dataset)

        # Reshape y for many-to-one ConvLSTM
        y_train = np.transpose(y_train, [0, 2, 3, 1, 4])
        y_val = np.transpose(y_val, [0, 2, 3, 1, 4])

        return x_train, y_train, x_val, y_val

    # -------------------------------
    # Build ConvLSTM model for a trial
    # -------------------------------
    def build_model(trial, input_shape):
        x_in = keras.Input(shape=input_shape)
        num_layers = trial.suggest_int("num_layers", 1, 2)  # limit to 2 for GPU safety
        x = x_in

        for i in range(num_layers):
            filters = trial.suggest_categorical(f"filters_l{i+1}", [16, 32])  # smaller filters
            kernel_size = trial.suggest_categorical(f"kernel_l{i+1}", [(3,3), (5,5)])
            return_seq = True if i < num_layers - 1 else False
            x = layers.ConvLSTM2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                return_sequences=return_seq,
                activation="relu"
            )(x)
            x = layers.BatchNormalization()(x)

        x = layers.Conv2D(
            filters=1,
            kernel_size=(3,3),
            padding="same",
            activation="sigmoid"
        )(x)

        lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        model = keras.models.Model(x_in, x)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy"
        )
        return model

    # -------------------------------
    # Optuna objective function
    # -------------------------------
    def objective(trial):
        # Suggest sequence length
        seq_length = trial.suggest_int("seq_length", 3, 4)  # smaller seq_length for GPU safety
        x_train, y_train, x_val, y_val = create_dataset(seq_length)

        model = build_model(trial, input_shape=x_train.shape[1:])

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

        # Batch size as hyperparameter
        batch_size = trial.suggest_categorical("batch_size", [4, 8])

        try:
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                batch_size=batch_size,
                epochs=50,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
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

    # -------------------------------

    # Run Optuna study
    # -------------------------------
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_trial.params)

    # -------------------------------
    # Save figure helper
    # -------------------------------
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
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    # -------------------------------
    # 1. Optimization history
    # -------------------------------
    ax1 = opt_viz.plot_optimization_history(study)
    save_plot(ax1, os.path.join(output_dir, "optuna_optimization_history.png"))

    # -------------------------------
    # 2. Hyperparameter importance
    # -------------------------------
    ax2 = opt_viz.plot_param_importances(study)
    save_plot(ax2, os.path.join(output_dir, "optuna_param_importance.png"))

    # -------------------------------
    # 3. Slice plot
    # -------------------------------
    ax3 = opt_viz.plot_slice(study)
    save_plot(ax3, os.path.join(output_dir, "optuna_slice_plot.png"))

    # -------------------------------
    # 4. Parallel coordinate plot
    # -------------------------------
    ax4 = opt_viz.plot_parallel_coordinate(study)
    save_plot(ax4, os.path.join(output_dir, "optuna_parallel_coordinate.png"))

    return study
