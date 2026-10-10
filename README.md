# Land Cover Classification with Machine Learning

Random Forest / Gradient Boosted classification on Sentinel-2 imagery, trained on
LandCoverNet labels, applied to a new area, with a full accuracy assessment and
a comparison to NLCD.

## Important: Radiant MLHub has been retired

The original project brief points to Radiant MLHub. That platform was shut down at
the end of October 2023 — all its datasets, including LandCoverNet, were migrated to
**Source Cooperative** (https://source.coop), a project also run by Radiant Earth.

- Old (dead): `mlhub.earth` API + Python client + API token
- New (works): anonymous S3-compatible access via `https://data.source.coop`, no
  account or token required

This project uses the new access path throughout.

## Pipeline

```
00_run_pipeline.py             -> MASTER SCRIPT: runs any/all of the 5 steps below,
                                   in order, from one command
01_download_conus_chips.py    -> pull a random sample of LandCoverNet NA chips
                                  whose footprint falls inside the continental US
02_build_training_table.py    -> turn chip GeoTIFFs into a flat pixel feature table
                                  (bands + NDVI/NDWI/NDBI), split by chip/tile
03_train_model.py             -> train RandomForest / HistGradientBoosting,
                                  save the fitted model + feature importances
04_apply_to_new_area.py       -> pull fresh Sentinel-2 imagery from Earth Search
                                  STAC for a held-out AOI and classify it
05_accuracy_assessment.py     -> confusion matrix, per-class metrics, and a
                                  NLCD comparison for the same AOI
```

Every script is heavily commented line-by-line — open them directly to see what
each step is doing and why, not just what it's called.

### Running the whole pipeline with one command

`00_run_pipeline.py` calls scripts 01-05 in order as separate command-line
processes (via Python's `subprocess` module), passing along whatever
parameters you give it. It doesn't reimplement any of their logic — it's
purely an orchestrator, so each script still works fine run on its own too.

```bash
cd src

# Run everything start to finish:
python 00_run_pipeline.py --n-chips 15 --seed 42 \
    --bbox -122.55 37.60 -122.35 37.80

# Just check the bucket layout, same as step 1's own --discover:
python 00_run_pipeline.py --discover

# Already have chips downloaded + a training table built? Skip straight to
# retraining, reclassifying, and reassessing:
python 00_run_pipeline.py --steps 3,4,5 --bbox -122.55 37.60 -122.35 37.80

# See exactly what WOULD run, without running anything (good for checking
# your flags before committing to a long download):
python 00_run_pipeline.py --n-chips 15 --bbox -122.55 37.60 -122.35 37.80 --dry-run

# Use a different model algorithm and more download concurrency:
python 00_run_pipeline.py --n-chips 20 --max-workers 24 --model-type gbm \
    --bbox -122.55 37.60 -122.35 37.80
```

Run `python 00_run_pipeline.py --help` to see every parameter grouped by
which step it belongs to (pipeline control, shared file paths, then each
step's own options). `--bbox` is the only one that's conditionally
required — only if step 4 is included in `--steps` (the default `--steps`
value runs all 5, so it's effectively always required unless you narrow
`--steps` to exclude step 4).

Shared file paths (`--raw-dir`, `--training-table`, `--model-path`,
`--classified-path`) are each used as BOTH one step's output and the next
step's input, so you never need to type matching paths twice — change
`--model-path` once and both step 3 (which writes it) and step 4 (which
reads it) automatically agree.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Notes on the "download CONUS chips" step

LandCoverNet North America (`landcovernet_na`) covers Canada, the US, and Mexico —
there's no separate "US-only" collection. `01_download_conus_chips.py` handles the
filtering itself:

1. Lists every chip folder in the `landcovernet_na` product (anonymous S3 list).
2. Parses the MGRS tile ID out of each folder name (e.g. `15SWD`) and does a cheap
   coarse filter on UTM zone / latitude band to throw out anything obviously in
   Canada, Mexico, or the open ocean.
3. For the surviving candidates, converts the MGRS tile's centroid to lat/lon and
   keeps only chips whose tile falls inside the official CONUS bounding box.
4. Randomly samples N chips from that precise CONUS set and downloads the full
   label + Sentinel-2 source files for just those chips.

This means step 1 never downloads the whole 1,561-chip NA collection — it only
pulls full data for the chips you actually asked for.

Run with `--discover` first to print the real bucket/folder layout without
downloading anything. Confirmed real layout, per chip:

```
landcovernet_na/data/v1.0/2018/<TILE>/<CHIP_ID>/
    <TILE>_<CHIP_ID>_2018_LC_10m.tif        <- label raster ("LC" = land cover)
    <TILE>_<CHIP_ID>_labeling_dates.csv     <- which dates the annual label was derived from
    S2/<TILE>_<CHIP_ID>_<DATE>/*_B02_10m.tif, ...   <- Sentinel-2, what we train on
    L8/<TILE>_<CHIP_ID>_<DATE>/*_B02_10m.tif, ...   <- Landsat-8 (unused)
    S1/<TILE>_<CHIP_ID>_<DATE>/*_VH_10m.tif, ...    <- Sentinel-1 (unused)
```

Two things worth knowing about this layout:

1. It's four levels deeper than LandCoverNet's older Radiant MLHub archive
   layout, and every band filename has a `_10m` resolution suffix.
2. **Landsat-8 uses the exact same `B01`–`B07` band-naming convention as
   Sentinel-2.** `02_build_training_table.py`'s `find_band_files()`
   deliberately searches only inside each chip's `S2/` subfolder --
   searching the whole chip folder would silently mix Landsat-8 and
   Sentinel-2 reflectance values together under the same band name.

`01_download_conus_chips.py` and `02_build_training_table.py` are written
against this confirmed layout.

## NLCD comparison caveat

NLCD only covers the continental US, uses a different (16-class, Anderson-derived)
legend, and is generated mostly from Landsat rather than Sentinel-2. Section 5 maps
LandCoverNet's 7 classes to the closest NLCD classes and reports agreement, but
treat this as a methodology comparison, not a strict accuracy score against ground
truth — see the writeup template in `05_accuracy_assessment.py`'s output.
