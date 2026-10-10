"""
Apply the trained model to a brand-new area, using fresh Sentinel-2 imagery
pulled from the Earth Search STAC API (no download of the whole scene needed --
rasterio reads windows directly off the cloud-optimized GeoTIFFs).

Example
-------
    python 04_apply_to_new_area.py \\
        --bbox -122.55 37.60 -122.35 37.80 \\
        --date-range 2024-06-01/2024-09-30 \\
        --out ../data/classified_new_area.tif
"""

# --- Standard library imports -----------------------------------------------
import argparse
from pathlib import Path

# --- Third-party imports -----------------------------------------------------
import joblib   # loads the {"model": ..., "feature_cols": ...} bundle from step 03
import numpy as np
import rasterio
from rasterio.windows import from_bounds     # convert a bbox into a pixel window
from rasterio.warp import transform_bounds    # reproject a bbox between CRSs
from pystac_client import Client              # queries STAC (SpatioTemporal Asset Catalog) APIs

import sys
sys.path.append(str(Path(__file__).parent))
from config import EARTH_SEARCH_STAC_URL, S2_BANDS, S2_BAND_TO_EARTH_SEARCH_ASSET


def find_best_scene(bbox, date_range, max_cloud=20):
    """
    Query Earth Search for Sentinel-2 L2A scenes covering `bbox` within
    `date_range`, and return the single least-cloudy matching scene.
    """
    # Client.open() connects to the STAC API root and reads its capabilities.
    catalog = Client.open(EARTH_SEARCH_STAC_URL)

    # .search() builds a query against the catalog:
    #   collections   -- restrict to Sentinel-2 Level-2A (surface reflectance)
    #   bbox           -- only scenes that intersect this [minlon,minlat,maxlon,maxlat]
    #   datetime       -- only scenes within this date range (STAC's ISO 8601 "start/end")
    #   query          -- extra filter on the eo:cloud_cover property
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    # .items() lazily fetches matching STAC Items (one per scene); list(...)
    # forces it to actually pull them all down now.
    items = list(search.items())
    if not items:
        raise SystemExit(
            f"No Sentinel-2 scenes found for bbox={bbox}, date_range={date_range}, "
            f"cloud_cover<{max_cloud}. Try widening the date range or cloud threshold."
        )

    # Sort ascending by cloud cover percentage (missing values treated as
    # 100%, i.e. worst) and take the clearest scene.
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    best = items[0]
    print(f"Using scene {best.id} (cloud cover "
          f"{best.properties.get('eo:cloud_cover', 'unknown')}%, "
          f"date {best.properties.get('datetime')})")
    return best


def read_bands_for_bbox(item, bbox, bands=S2_BANDS):
    """
    Read just the pixel window covering `bbox` (given in WGS84 lon/lat) from
    each requested band asset of a STAC item -- NOT the whole scene. Since
    Sentinel-2 assets on AWS are Cloud-Optimized GeoTIFFs (COGs), rasterio
    can fetch only the needed bytes over HTTP range requests.

    IMPORTANT: Earth Search's `sentinel-2-l2a` collection does NOT name its
    assets "B02", "B03", etc. -- it uses common/human-readable names instead
    ("blue", "green", "red", "nir", "swir16", ...). We look each band code up
    in S2_BAND_TO_EARTH_SEARCH_ASSET (config.py) to get the real asset key,
    but keep using the "B02"-style codes as OUR OWN dict keys throughout this
    function and in build_feature_stack() below, so the rest of the pipeline
    (and the feature_cols list saved alongside the trained model) doesn't
    need to know anything about Earth Search's particular naming scheme.
    """
    arrays = {}
    # We'll lock in the transform/CRS/shape from the FIRST band we
    # successfully read, and treat that as the reference grid; any later
    # band with a different resolution (e.g. 20m bands vs this scene's first
    # 10m band) gets resampled onto that same grid.
    ref_transform = ref_crs = ref_shape = None

    for band in bands:
        asset_key = S2_BAND_TO_EARTH_SEARCH_ASSET.get(band)
        if asset_key is None:
            print(f"  (no known Earth Search asset name for {band}, skipping)")
            continue

        asset = item.assets.get(asset_key)
        if asset is None:
            print(f"  (asset '{asset_key}' for {band} not found on this item, skipping)")
            continue

        with rasterio.open(asset.href) as src:
            # Reproject the WGS84 bbox into whatever CRS this specific band
            # raster uses (Sentinel-2 tiles use local UTM zones, not WGS84).
            warped_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
            # Convert that reprojected bbox into a pixel row/col window
            # relative to this raster's affine transform.
            window = from_bounds(*warped_bbox, transform=src.transform)

            # Read only that window (not the whole file); boundless=True
            # pads with fill_value if the requested window slightly overhangs
            # the raster's actual extent, instead of raising an error.
            arr = src.read(1, window=window, out_dtype="float32",
                           boundless=True, fill_value=0)

            if ref_transform is None:
                # First band read: establish this as the reference grid that
                # every subsequent band will be aligned to.
                ref_transform = src.window_transform(window)
                ref_crs = src.crs
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                # This band has a different pixel grid than the reference
                # (e.g. it's a native 20m band being compared against a 10m
                # reference) -- resample it onto the reference grid so every
                # band ends up as arrays of identical shape/alignment.
                from rasterio.warp import reproject, Resampling
                resampled = np.empty(ref_shape, dtype="float32")
                reproject(
                    source=arr,
                    destination=resampled,
                    src_transform=src.window_transform(window),
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,  # smooth interpolation, fine for continuous reflectance values
                )
                arr = resampled

            arrays[band] = arr

    return arrays, ref_transform, ref_crs, ref_shape


def build_feature_stack(arrays, feature_cols):
    """
    Given the raw band arrays read above, compute the same derived indices
    (NDVI/NDWI/NDBI) used during training, then stack everything into a
    single (H, W, n_features) array in the exact column order the model
    expects (`feature_cols`, saved alongside the model in step 03).
    """
    def idx(a, b):
        # Same normalized-difference formula used in 02_build_training_table.py
        return (a - b) / (a + b + 1e-6)

    derived = {}
    if {"B08", "B04"} <= set(arrays):
        derived["ndvi"] = idx(arrays["B08"], arrays["B04"])
    if {"B03", "B08"} <= set(arrays):
        derived["ndwi"] = idx(arrays["B03"], arrays["B08"])
    if {"B11", "B08"} <= set(arrays):
        derived["ndbi"] = idx(arrays["B11"], arrays["B08"])

    # Merge raw bands and derived indices into one lookup dict.
    all_layers = {**arrays, **derived}

    # Make sure every feature the trained model expects is actually available;
    # fail loudly and clearly if something is missing rather than silently
    # producing wrong predictions.
    missing = [f for f in feature_cols if f not in all_layers]
    if missing:
        raise SystemExit(f"Missing required features for this model: {missing}")

    # np.stack with axis=-1 turns a list of (H, W) arrays into one (H, W,
    # n_features) array, in the SAME order as feature_cols -- which must
    # match the column order used during training.
    stack = np.stack([all_layers[f] for f in feature_cols], axis=-1)
    return stack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=float, nargs=4, required=True,
                         metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                         help="Bounding box of the new area, in WGS84 lon/lat")
    parser.add_argument("--date-range", type=str, default="2024-06-01/2024-09-30",
                         help="STAC datetime range, ISO 8601 'start/end'")
    parser.add_argument("--max-cloud", type=float, default=20,
                         help="Maximum acceptable scene-level cloud cover percentage")
    parser.add_argument("--model", type=Path, default=Path("../data/model.joblib"),
                         help="Path to the model saved by 03_train_model.py")
    parser.add_argument("--out", type=Path, default=Path("../data/classified_new_area.tif"),
                         help="Where to save the classified GeoTIFF")
    args = parser.parse_args()

    # Load the {"model": fitted_classifier, "feature_cols": [...]} bundle
    # saved by 03_train_model.py.
    bundle = joblib.load(args.model)
    clf, feature_cols = bundle["model"], bundle["feature_cols"]

    # Find the best (least cloudy) Sentinel-2 scene for this area/date range,
    # then read just the bbox window from each needed band.
    item = find_best_scene(args.bbox, args.date_range, args.max_cloud)
    arrays, transform, crs, shape = read_bands_for_bbox(item, args.bbox)

    # Build the (H, W, n_features) feature stack in the exact column order
    # the trained model expects.
    stack = build_feature_stack(arrays, feature_cols)

    h, w, n_features = stack.shape
    # Flatten from (H, W, n_features) to (H*W, n_features) -- scikit-learn's
    # .predict() expects a 2D (n_samples, n_features) array, treating each
    # pixel as one "sample".
    flat = stack.reshape(-1, n_features)

    # Some pixels might have NaN/Inf values (e.g. from the boundless=True
    # padding, or division edge cases in the NDVI/NDWI/NDBI formulas).
    # np.isfinite(...).all(axis=1) marks a pixel "valid" only if every one
    # of its features is a finite number.
    valid = np.isfinite(flat).all(axis=1)

    # Pre-fill predictions with 0 (our "nodata" class code), then only
    # predict on the valid pixels -- this avoids feeding NaNs into the model
    # and keeps invalid pixels clearly marked as nodata in the output.
    preds = np.zeros(flat.shape[0], dtype="uint8")
    preds[valid] = clf.predict(flat[valid])
    # Reshape the flat prediction array back into a 2D (H, W) image.
    classified = preds.reshape(h, w)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Write the classified array out as a single-band GeoTIFF, using the same
    # CRS/transform we tracked while reading the input bands so the output
    # lines up correctly on a map.
    with rasterio.open(
        args.out, "w",
        driver="GTiff",
        height=h, width=w, count=1,
        dtype="uint8", crs=crs, transform=transform,
        compress="lzw",   # lossless compression, keeps file size down
        nodata=0,          # marks class code 0 as "no data" for GIS software
    ) as dst:
        dst.write(classified, 1)

    print(f"\nWrote classified map to {args.out} ({h}x{w} px)")


if __name__ == "__main__":
    main()
