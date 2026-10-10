"""
Shared constants for the land cover classification pipeline.

Every other script in this project does `from config import ...` so that
bucket names, bounding boxes, and class definitions only live in one place.
"""

# ============================================================================
# SOURCE COOPERATIVE ACCESS
# ============================================================================
# Radiant MLHub (the platform named in the original project brief) was retired
# at the end of October 2023. Its datasets, including LandCoverNet, were
# migrated to Source Cooperative -- a separate project also run by Radiant
# Earth. Source Cooperative exposes data through a plain S3-compatible HTTP
# endpoint with anonymous (no login, no API key) access, which is simpler
# than the old Radiant MLHub API + Python client + token setup ever was.

# The base URL of the Source Cooperative "data proxy" -- this stands in for
# a normal AWS S3 endpoint (e.g. https://s3.amazonaws.com) when we create a
# boto3 client later.
SOURCE_COOP_ENDPOINT = "https://data.source.coop"

# In Source Cooperative's URL scheme, https://source.coop/<account>/<product>
# maps to bucket=<account>, key prefix=<product>/... in the S3 API. Radiant
# Earth publishes LandCoverNet under the "radiantearth" account.
LANDCOVERNET_BUCKET = "radiantearth"

# Within that bucket, the North America region of LandCoverNet lives under
# this prefix (LandCoverNet is split into six regional products: af, as, au,
# eu, na, sa -- we only need "na" since we're filtering to the continental US).
LANDCOVERNET_NA_PREFIX = "landcovernet/landcovernet_na/"

# The actual per-chip imagery/labels live several levels deeper than the
# product root, confirmed via `01_download_conus_chips.py --discover`:
#   landcovernet_na/data/v1.0/2018/<TILE>/<CHIP_ID>/S2/<TILE>_<CHIP_ID>_<DATE>/*.tif
# "2018" is the only year LandCoverNet v1.0 covers (its annual labels are
# derived from a full year of 2018 imagery), so it's hardcoded here rather
# than being a variable -- if Radiant Earth ever adds another year, this is
# the one line you'd need to change.
LANDCOVERNET_NA_DATA_PREFIX = LANDCOVERNET_NA_PREFIX + "data/v1.0/2018/"

# ============================================================================
# CONTINENTAL US (CONUS) BOUNDING BOX
# ============================================================================
# These are the official extreme points of the continental United States.
# "Continental US" (CONUS) deliberately EXCLUDES Alaska and Hawaii -- if we
# just filtered on "is this in the USA" we'd also pull in Alaskan tundra
# chips, which would skew the training data toward land cover types that
# don't represent the "lower 48."
CONUS_BBOX = {
    "min_lon": -125.0,       # West edge: roughly the Pacific coast of WA/OR/CA
    "max_lon": -66.93457,    # East edge: roughly the Atlantic coast of Maine
    "min_lat": 24.396308,    # South edge: the Florida Keys
    "max_lat": 49.384358,    # North edge: the WA/Canada border
}

# ---------------------------------------------------------------------------
# Cheap pre-filter, used BEFORE we do the more precise lat/lon check above.
# MGRS (Military Grid Reference System) tile IDs start with a UTM zone number
# and a latitude band letter, e.g. "15SWD" = zone 15, band S, 100km square WD.
# We can rule out a lot of non-CONUS tiles just from the zone/band prefix,
# without doing any coordinate math, which keeps the download script fast.
# ---------------------------------------------------------------------------
CONUS_UTM_ZONES = set(range(10, 20))  # Zone 10 (US West Coast) through 19 (Maine)

# MGRS latitude bands are ~8 degrees tall, lettered C (near -80 lat) up to X
# (near +84 lat), skipping I and O to avoid confusion with 1 and 0.
# Band R covers ~24-32N, S covers ~32-40N, T covers ~40-48N -- together these
# span the full CONUS latitude range (24.4N to 49.4N) with a little margin.
CONUS_MGRS_BANDS = {"R", "S", "T"}

# ============================================================================
# LANDCOVERNET CLASS TAXONOMY
# ============================================================================
# LandCoverNet labels each pixel with one of 7 integer class codes. This
# dict lets us translate those integers into human-readable names anywhere
# in the pipeline (confusion matrix labels, printed reports, etc.).
LANDCOVERNET_CLASSES = {
    1: "water",
    2: "natural_bare_ground",
    3: "artificial_bare_ground",
    4: "woody_vegetation",
    5: "cultivated_vegetation",
    6: "semi_natural_vegetation",
    7: "permanent_snow_ice",
}

# ---------------------------------------------------------------------------
# Rough mapping from LandCoverNet's 7 classes to NLCD-style groupings, used
# only in the final writeup/comparison step (05_accuracy_assessment.py).
# NLCD has 16 classes with different semantics (e.g. it splits "developed"
# into 4 intensity levels), so this is a many-to-one, APPROXIMATE crosswalk
# meant to make the two products roughly comparable -- not an authoritative
# translation table.
# ---------------------------------------------------------------------------
LANDCOVERNET_TO_NLCD_GROUP = {
    "water": "Water",
    "natural_bare_ground": "Barren",
    "artificial_bare_ground": "Developed",
    "woody_vegetation": "Forest",
    "cultivated_vegetation": "Cultivated",
    "semi_natural_vegetation": "Grassland/Shrubland",
    "permanent_snow_ice": "Snow/Ice",
}

# ============================================================================
# SENTINEL-2 BAND SETUP
# ============================================================================
# The Sentinel-2 L2A (surface reflectance, atmospherically corrected) bands
# we'll use as classifier features. We include all 10m and 20m bands that
# carry useful spectral information for land cover (we skip the 60m
# atmospheric bands B01/B09/B10, which are meant for atmospheric correction,
# not surface classification).
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Human-readable names for each band, purely for printouts/debugging.
S2_BAND_NAMES = {
    "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
    "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir_narrow",
    "B11": "swir1", "B12": "swir2",
}

# ---------------------------------------------------------------------------
# Earth Search (element84's STAC API, used in 04_apply_to_new_area.py) does
# NOT expose Sentinel-2 assets under keys like "B02" -- confirmed by testing,
# item.assets uses common/human names instead: "blue", "green", "red", etc.
# (This is different from the raw LandCoverNet band filenames, which DO use
# "B02"-style codes -- so this mapping is only needed on the Earth Search
# side, not in 02_build_training_table.py.)
# This dict translates our internal S2_BANDS codes into the actual asset
# keys Earth Search uses, so 04_apply_to_new_area.py can look up
# `item.assets[S2_BAND_TO_EARTH_SEARCH_ASSET["B02"]]` and get back the right
# asset instead of a silent `None`.
# ---------------------------------------------------------------------------
S2_BAND_TO_EARTH_SEARCH_ASSET = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",       # nir = the standard 10m near-infrared band (B08)
    "B8A": "nir08",      # nir08 = the narrower 20m near-infrared band (B8A)
    "B11": "swir16",
    "B12": "swir22",
}

# Public, free-to-query STAC (SpatioTemporal Asset Catalog) API that serves
# up-to-date Sentinel-2 scenes hosted on AWS. Used in 04_apply_to_new_area.py
# to pull fresh imagery for an area outside the training set.
EARTH_SEARCH_STAC_URL = "https://earth-search.aws.element84.com/v1"
