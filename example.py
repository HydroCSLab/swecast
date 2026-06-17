from datetime import date
from swecast import (
    Manifest,
    prepare_training_inputs,
    train_swe,
    train_swe_pcp,
    train_swe_tmp,
    train_swe_tmp_pcp,
)

# defin a manifest for the data we want to use.
manifest = Manifest(
    start=date(2001, 1, 1),
    end=date(2021, 1, 1),
    bbox=(-121.9, 36.08, -109, 41.98),
    num_data_used=5000,
)

inputs = prepare_training_inputs(
    output_dir="./output",
    manifest=manifest,
)

# 1. SWE
train_swe(
    swe_filled=inputs.swe_filled,
    station_cells=inputs.station_cells,
    output_dir="./output/forecast_swe",
    manifest=manifest,
)

# 2. SWE + PCP (2 channels)
train_swe_pcp(
    swe_filled=inputs.swe_filled,
    pcp_filled=inputs.pcp_filled,
    station_cells=inputs.station_cells,
    output_dir="./output/forecast_swe_pcp",
    manifest=manifest,
    batch_size=8,
)

# 3. SWE + TMP (2 channels)
train_swe_tmp(
    swe_filled=inputs.swe_filled,
    tmp_filled=inputs.tmp_filled,
    station_cells=inputs.station_cells,
    output_dir="./output/forecast_swe_tmp",
    manifest=manifest,
    batch_size=8,
)

# 4. SWE + TMP + PCP (3 channels)
train_swe_tmp_pcp(
    swe_filled=inputs.swe_filled,
    tmp_filled=inputs.tmp_filled,
    pcp_filled=inputs.pcp_filled,
    station_cells=inputs.station_cells,
    output_dir="./output/forecast_swe_tmp_pcp",
    manifest=manifest,
    batch_size=4,
)
