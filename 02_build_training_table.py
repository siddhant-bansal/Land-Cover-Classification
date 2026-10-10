"""
Turn downloaded LandCoverNet chips into a flat per-pixel feature table.

Expects the layout produced by 01_download_conus_chips.py, confirmed via
`--discover` to look like:

    data/raw/<TILE>_<CHIP_ID>/<TILE>_<CHIP_ID>_2018_LC_10m.tif   -> label raster
    data/raw/<TILE>_<CHIP_ID>/<TILE>_<CHIP_ID>_labeling_dates.csv -> (unused here)
    data/raw/<TILE>_<CHIP_ID>/S2/<TILE>_<CHIP_ID>_<DATE>/*_B02_10m.tif, ...  -> Sentinel-2
    data/raw/<TILE>_<CHIP_ID>/L8/<TILE>_<CHIP_ID>_<DATE>/*_B02_10m.tif, ...  -> Landsat-8 (unused here)
    data/raw/<TILE>_<CHIP_ID>/S1/<TILE>_<CHIP_ID>_<DATE>/*_VH_10m.tif, ...   -> Sentinel-1 (unused here)

We only use the Sentinel-2 (S2/) imagery as features here -- note that
Landsat-8 (L8/) uses the EXACT SAME "B01".."B07" band-naming convention, so
`find_band_files()` deliberately scopes its search to the S2/ subfolder only
to avoid silently mixing the two sensors together. If you want to also use
Landsat-8 or Sentinel-1 bands as extra features later, you'd add a similarly
scoped loader for L8/ or S1/ rather than widening this one.

For each chip we take the per-band *median across all available dates* as the
feature value for that pixel (a standard, cloud-robust way to collapse a
Sentinel-2 time series into a single classification-ready composite), then
compute NDVI / NDWI / NDBI on top of the composite.

Output: data/training_table.parquet, with columns:
    chip_id, row, col, <band columns>, ndvi, ndwi, ndbi, label, consensus
"""

# --- Standard library imports -----------------------------------------------
import argparse   # command-line flag parsing
import re         # regex, used to spot Sentinel-2 band codes in filenames
from pathlib import Path  # filesystem path handling

# --- Third-party imports -----------------------------------------------------
import numpy as np       # array math (medians, reshaping, NaN-safe division)
import pandas as pd       # building/saving the tabular output
import rasterio           # reading GeoTIFF raster files
from tqdm import tqdm     # progress bar while looping over chips

import sys
# Make config.py (in the same folder) importable regardless of the current
# working directory this script is run from.
sys.path.append(str(Path(__file__).parent))
from config import S2_BANDS, LANDCOVERNET_CLASSES


def find_band_files(chip_dir: Path):
    """
    Return {band_name: [Path, ...]} for every Sentinel-2 date raster found
    under this chip's folder.

    IMPORTANT: confirmed via --discover, each chip folder contains SIBLING
    subfolders for multiple sensors:
        <chip_dir>/S2/<TILE>_<CHIP>_<DATE>/..._B02_10m.tif   (Sentinel-2, what we want)
        <chip_dir>/L8/<TILE>_<CHIP>_<DATE>/..._B02_10m.tif   (Landsat-8 -- ALSO uses B01-B07!)
        <chip_dir>/S1/<TILE>_<CHIP>_<DATE>/..._VH_10m.tif    (Sentinel-1, VV/VH polarization, not band codes)

    Landsat-8's bands are named with the exact same "B01".."B07" codes as
    Sentinel-2's lower bands. If we naively rglob() the WHOLE chip folder,
    we'd silently mix Landsat-8 reflectance values in with Sentinel-2
    reflectance values under the same "B02" key -- they're physically
    different measurements, so this would corrupt the composites without
    raising any error. We avoid that by only searching inside the S2/
    subfolder specifically.
    """
    # Start with an empty list for every band we care about, so bands that
    # simply don't have any files still show up as band_files[band] == [].
    band_files = {b: [] for b in S2_BANDS}

    s2_dir = chip_dir / "S2"
    if not s2_dir.exists():
        # No Sentinel-2 imagery for this chip at all -- return empty lists
        # rather than accidentally falling back to searching L8/S1.
        return band_files

    # rglob("*.tif") recursively searches every subfolder of s2_dir (i.e.
    # every dated observation folder) for files ending in .tif.
    for tif in s2_dir.rglob("*.tif"):
        # Look for a Sentinel-2 band code, optionally followed by a
        # resolution suffix like "_10m", right before ".tif" in the
        # filename -- e.g. "..._B06_10m.tif" -> matches "B06".
        # Band codes covered: B02-B09, B11, B12, and the special case B8A.
        m = re.search(r"_(B0?[2-9]|B1[12]|B8A)(?:_\d+m)?\.tif$", tif.name, re.IGNORECASE)
        if not m:
            # Shouldn't normally happen inside S2/, but skip gracefully if
            # some other file type shows up here.
            continue

        # The regex already captures the band code fully zero-padded (e.g.
        # "B06", not "B6"), since "B0?[2-9]" matches the literal "0" when
        # present. No extra normalization needed -- just uppercase it in
        # case the file used lowercase band codes.
        band = m.group(1).upper()

        if band in band_files:
            band_files[band].append(tif)

    return band_files


def load_label(chip_dir: Path):
    """
    Read the label raster for a chip. Confirmed via --discover: the label
    file lives directly in the chip folder, named
        <TILE>_<CHIP_ID>_2018_LC_10m.tif
    e.g. "10SGJ_00_2018_LC_10m.tif" ("LC" = land cover). There's also a
    "<TILE>_<CHIP_ID>_labeling_dates.csv" alongside it (which dates were
    used to derive the annual label) -- we don't need that file for
    training, just the label raster itself.

    Per LandCoverNet's documentation this GeoTIFF may have a second band
    holding the per-pixel consensus/agreement score; we read it if present,
    and fall back to NaN if the file only has one band.
    """
    # Primary, confirmed pattern: "*_2018_LC_10m.tif" directly in chip_dir.
    candidates = list(chip_dir.glob("*_LC_*.tif"))
    # Fallbacks, in case a different LandCoverNet region/version names things
    # slightly differently (e.g. no "_10m" suffix, or a generic "labels.tif").
    candidates += [chip_dir / "labels.tif"]
    candidates += list(chip_dir.glob("*/labels.tif"))
    candidates += list(chip_dir.rglob("*label*.tif"))

    label_path = next((p for p in candidates if p.exists()), None)
    if label_path is None:
        print(f"    !! no label raster found under {chip_dir} -- run "
              f"`python 01_download_conus_chips.py --discover` and inspect "
              f"this chip's full file listing to find the right filename, "
              f"then update load_label() accordingly.")
        return None, None

    with rasterio.open(label_path) as src:
        label = src.read(1)  # band 1: integer class codes, one per pixel
        # Some chip folders might only have a single-band label file; guard
        # against that by only reading band 2 if it actually exists.
        consensus = src.read(2) if src.count > 1 else np.full_like(label, np.nan, dtype="float32")

    return label, consensus


def median_composite(band_files, band_name):
    """
    Given the list of file paths for one band across multiple observation
    dates, read them all in and return the per-pixel MEDIAN value across
    time. Using the median (rather than, say, the mean or a single date)
    is a simple way to suppress the effect of clouds/shadows on a handful
    of dates without having to run explicit cloud masking.
    """
    paths = band_files.get(band_name, [])
    if not paths:
        return None  # this band simply wasn't found for this chip

    stack = []
    for p in paths:
        with rasterio.open(p) as src:
            # Read band 1 of each single-band Sentinel-2 file; cast to
            # float32 so later NaN-based math (division, medians) works
            # cleanly even if the source data is an integer dtype.
            stack.append(src.read(1).astype("float32"))

    # np.stack turns the list of 2D (H, W) arrays into one 3D (T, H, W) array,
    # where T is the number of dates; nanmedian collapses the T axis, giving
    # back a single (H, W) composite that ignores any NaN values.
    return np.nanmedian(np.stack(stack, axis=0), axis=0)


def build_chip_table(chip_dir: Path):
    """
    Build one pandas DataFrame of per-pixel features + label for a single
    chip folder, or return None if the chip is missing required data.
    """
    label, consensus = load_label(chip_dir)
    if label is None:
        # No usable label raster -- this chip can't be used for training.
        return None

    band_files = find_band_files(chip_dir)

    # Build a median composite for every band that has at least one file.
    composites = {}
    for band in S2_BANDS:
        comp = median_composite(band_files, band)
        if comp is None:
            continue
        composites[band] = comp

    if not composites:
        # Found a label but no usable imagery bands -- also unusable.
        return None

    # Sanity check: every band composite must be the same pixel dimensions
    # as the label raster, otherwise our row/col indices won't line up.
    shape = label.shape
    for band, arr in composites.items():
        if arr.shape != shape:
            raise ValueError(
                f"{chip_dir.name}: {band} shape {arr.shape} != label shape {shape}"
            )

    # np.indices(shape) generates two 2D arrays of the same shape as `label`:
    # one holding each pixel's row index, one holding each pixel's column
    # index. We flatten (.ravel()) everything into 1D so each row of the
    # final table corresponds to exactly one pixel.
    rows, cols = np.indices(shape)

    data = {
        "chip_id": chip_dir.name,   # which chip this pixel came from (used
                                     # later for chip-grouped train/test splits)
        "row": rows.ravel(),
        "col": cols.ravel(),
        "label": label.ravel(),
        "consensus": consensus.ravel(),
    }
    # Add one flattened column per spectral band.
    for band, arr in composites.items():
        data[band] = arr.ravel()

    df = pd.DataFrame(data)

    # LandCoverNet uses 0 (and possibly other codes outside 1-7) to mark
    # unlabeled/nodata pixels. Drop anything that isn't one of the 7 real
    # classes so we don't accidentally train the model to predict "nodata".
    df = df[df["label"].isin(LANDCOVERNET_CLASSES.keys())].copy()

    def safe_index(a, b):
        # Standard normalized-difference formula: (a - b) / (a + b).
        # The tiny +1e-6 in the denominator avoids a divide-by-zero error
        # for pixels where both bands happen to be exactly 0.
        return (a - b) / (a + b + 1e-6)

    # Only compute each spectral index if its required input bands actually
    # made it into `composites` (some chips might be missing a band).
    if {"B08", "B04"} <= set(composites):
        # NDVI (Normalized Difference Vegetation Index): near-infrared vs red.
        # High values = healthy/dense vegetation.
        df["ndvi"] = safe_index(df["B08"], df["B04"])
    if {"B03", "B08"} <= set(composites):
        # NDWI (Normalized Difference Water Index): green vs near-infrared.
        # High values = open water.
        df["ndwi"] = safe_index(df["B03"], df["B08"])
    if {"B11", "B08"} <= set(composites):
        # NDBI (Normalized Difference Built-up Index): shortwave-infrared vs
        # near-infrared. High values = built-up/impervious surfaces.
        df["ndbi"] = safe_index(df["B11"], df["B08"])

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("../data/raw"),
                         help="Folder containing downloaded chip subfolders")
    parser.add_argument("--out", type=Path, default=Path("../data/training_table.parquet"),
                         help="Where to save the resulting feature table")
    parser.add_argument("--sample-per-class", type=int, default=20000,
                         help="Max pixels per class per chip, to keep the table manageable "
                              "(each chip is 256x256=65,536 pixels, which adds up fast)")
    args = parser.parse_args()

    # Every immediate subdirectory of --raw-dir is assumed to be one chip's
    # downloaded folder.
    chip_dirs = [d for d in args.raw_dir.iterdir() if d.is_dir()]
    if not chip_dirs:
        print(f"No chip folders found in {args.raw_dir}. Run 01_download_conus_chips.py first.")
        return

    tables = []  # will hold one DataFrame per usable chip

    # tqdm wraps the iterable to show a live progress bar in the terminal.
    for chip_dir in tqdm(chip_dirs, desc="Building per-chip tables"):
        df = build_chip_table(chip_dir)
        if df is None or df.empty:
            print(f"  skipped {chip_dir.name} (missing labels or bands)")
            continue

        if args.sample_per_class:
            # Downsample each class within this chip to at most
            # `sample_per_class` pixels, so common classes (like
            # "cultivated_vegetation") don't completely dominate the table
            # at the expense of rarer classes (like "permanent_snow_ice").
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.sample_per_class), random_state=42))
            )

        tables.append(df)

    if not tables:
        print("No usable chips found -- nothing to write.")
        return

    # Stack every chip's table into one big DataFrame.
    full = pd.concat(tables, ignore_index=True)

    # Make sure the output folder exists, then write as Parquet (a compact,
    # typed, columnar format -- much faster to reload than CSV for this
    # many rows).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out, index=False)

    print(f"\nWrote {len(full):,} pixel rows from {len(tables)} chips to {args.out}")
    # Quick class-balance sanity check, printed as human-readable class names
    # rather than raw integer codes.
    print(full["label"].map(LANDCOVERNET_CLASSES).value_counts())


if __name__ == "__main__":
    main()
