import swecast
from datetime import date
from swecast import Manifest, build_stacks, build_swe_stacks
from swecast import fetch_stations, stations_to_csv, fill_stacks, fill_npy
from swecast import (
    prepare_training_inputs,
    train_swe,
    train_swe_pcp,
    train_swe_tmp,
    train_swe_tmp_pcp,
)

#defin a manifest for the data we want to use.
manifest = Manifest(
    start=date(2001, 1, 1),
    end=date(2021, 1, 1),
    bbox=(-121.9, 36.08, -109, 41.98),
    num_data_used=5000, 
)



#build stacks
outputs = build_stacks(manifest, output_dir="./output")
#swe stacks
swe_outputs = build_swe_stacks(manifest, output_dir="./output")


# Gap-fill SWE stacks (also fills sibling swe.npy -> swe_filled.npy)
filled = fill_stacks(swe_outputs)

# Gap-fill the PRISM .npy stacks too, since the multi-channel ConvLSTM
# variants expect pcp_filled.npy and tmp_filled.npy.
pcp_filled = fill_npy("./output/pcp.npy")  # -> ./output/pcp_filled.npy
tmp_filled = fill_npy("./output/tmp.npy")  # -> ./output/tmp_filled.npy

inputs = prepare_training_inputs(
    filled_stacks=filled,
    stations_csv="./output/.cache/swe_stations.csv",
    output_dir="./output/forecast",
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
    swe_filled=swe_filled,
    pcp_filled=pcp_filled,
    station_cells=station_cells,
    output_dir="./output/forecast_swe_pcp",
    manifest=manifest,
    batch_size=8,
)

# 3. SWE + TEMP (2 channels)
train_swe_tmp(
    swe_filled="./output/swe_filled.npy",
    tmp_filled="./output/tmp_filled.npy",
    station_cells=station_cells,
    output_dir="./output/forecast_swe_tmp",
    manifest=manifest,
    batch_size=8,
)

# 4. SWE + TEMP + PCP (3 channels)
train_swe_tmp_pcp(
    swe_filled="./output/swe_filled.npy",
    tmp_filled="./output/tmp_filled.npy",
    pcp_filled="./output/pcp_filled.npy",
    station_cells=station_cells,
    output_dir="./output/forecast_swe_tmp_pcp",
    manifest=manifest,
    batch_size=4,
)
