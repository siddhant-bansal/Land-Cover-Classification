"""
Download a random sample of LandCoverNet North America chips whose footprint
falls inside the continental US (CONUS).

LandCoverNet is hosted on Source Cooperative (the successor to the now-retired
Radiant MLHub), and is accessible anonymously via an S3-compatible endpoint --
no account, API key, or `radiant_mlhub` client needed.

Usage
-----
    # First, just look at how the bucket is actually laid out (no downloads):
    python 01_download_conus_chips.py --discover

    # Then pull e.g. 15 random CONUS chips:
    python 01_download_conus_chips.py --n-chips 15 --seed 42 --dest ../data/raw

    # Each chip has 600+ small files, so downloads within a chip happen in
    # parallel via a thread pool (default 16 at a time); tune with:
    python 01_download_conus_chips.py --n-chips 15 --max-workers 24

Bucket layout
-------------
Confirmed via --discover, the real layout is nested several levels deeper
than LandCoverNet's older Radiant MLHub archive format:

    landcovernet_na/data/v1.0/2018/<TILE>/<CHIP_ID>/S2/<TILE>_<CHIP_ID>_<DATE>/*.tif

where <TILE> is an MGRS 100km grid square ID (e.g. "10SGJ") and <CHIP_ID> is
a two-digit index (e.g. "00"). So the "chip folder" we actually want to
download as one unit is:

    landcovernet_na/data/v1.0/2018/<TILE>/<CHIP_ID>/

which -- alongside the S2/ (Sentinel-2) subfolder -- may also contain
sibling subfolders/files for Sentinel-1, Landsat-8, and the labels raster.
This script downloads that *entire* chip folder (everything under it,
whatever it contains), rather than guessing at individual filenames.

Notes
-----
LandCoverNet tile IDs are MGRS 100km grid squares. We use each tile's
centroid as a cheap, good-enough proxy for "is this chip in CONUS" -- a chip
is a 256x256 px (2.56km x 2.56km) sample inside a 100km x 100km MGRS tile,
so this can very occasionally include a chip that's actually just across the
Canada/Mexico border from a US tile. Good enough for building a training
set; not a survey-grade spatial filter.
"""

# --- Standard library imports -----------------------------------------------
import argparse   # parses command-line flags like --n-chips, --seed, --dest
import re         # regular expressions, used to pull MGRS tile IDs out of folder names
import random     # random.sample() for picking a random subset of chips
import sys        # sys.path manipulation (below) and sys.exit-style errors
from pathlib import Path  # object-oriented filesystem paths (nicer than raw strings)

# concurrent.futures is Python's standard library toolkit for running work in
# parallel. We use ThreadPoolExecutor specifically -- see the big explanatory
# comment above download_chip() below for what that actually means and why
# it helps here.
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Third-party imports -----------------------------------------------------
import boto3                        # AWS SDK for Python; we use its S3 client
from botocore import UNSIGNED       # marks requests as "anonymous" (no AWS credentials)
from botocore.config import Config  # lets us pass signature_version=UNSIGNED to boto3

# The `mgrs` package converts between MGRS grid references (like "10SGJ") and
# WGS84 latitude/longitude. We wrap the import in try/except so that if it's
# missing, the user gets a clear "pip install mgrs" message instead of a
# confusing traceback.
try:
    import mgrs
except ImportError:
    print("Missing dependency: pip install mgrs", file=sys.stderr)
    raise

# Make sure Python can find config.py, which lives in the same folder as this
# script (src/). This line adds that folder to the module search path so the
# `from config import ...` below works no matter where you run this script from.
sys.path.append(str(Path(__file__).parent))
from config import (
    SOURCE_COOP_ENDPOINT,          # base URL for Source Cooperative's S3-compatible API
    LANDCOVERNET_BUCKET,           # "radiantearth" -- the account/bucket name
    LANDCOVERNET_NA_PREFIX,        # "landcovernet/landcovernet_na/" -- product path
    LANDCOVERNET_NA_DATA_PREFIX,   # "...na/data/v1.0/2018/" -- actual tile/chip root
    CONUS_BBOX,                    # dict with min/max lat/lon for the continental US
    CONUS_UTM_ZONES,               # set of UTM zone numbers that overlap CONUS
    CONUS_MGRS_BANDS,              # set of MGRS latitude-band letters that overlap CONUS
)

# ---------------------------------------------------------------------------
# Regex that matches an MGRS 100km-grid-square tile ID embedded anywhere in a
# string, e.g. ".../2018/10SGJ/00/..." -> "10SGJ".
#   (\d{1,2})              -- UTM zone number: 1 or 2 digits (01-60)
#   ([C-HJ-NP-X])          -- latitude band letter (skips I and O, and A/B/Y/Z
#                             which are reserved for polar regions)
#   ([A-HJ-NP-Z]{2})       -- two-letter 100km grid square ID (also skips I/O)
# The (?<!...) and (?!...) lookaround groups make sure we don't match in the
# middle of a longer alphanumeric run (e.g. accidentally matching inside a
# date string like "20180209" or a filename hash).
# ---------------------------------------------------------------------------
MGRS_TILE_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})([C-HJ-NP-X])([A-HJ-NP-Z]{2})(?![A-Za-z0-9])")


def get_client():
    """
    Build a boto3 S3 client pointed at Source Cooperative's data proxy instead
    of real AWS S3, configured for ANONYMOUS access (no AWS account needed).
    """
    return boto3.client(
        "s3",
        # Redirect all S3 API calls to Source Cooperative's endpoint instead
        # of the default AWS S3 endpoint.
        endpoint_url=SOURCE_COOP_ENDPOINT,
        # UNSIGNED tells boto3 not to try to sign requests with AWS
        # credentials -- Source Cooperative allows public, unauthenticated
        # reads, so signing would just cause errors (or be unnecessary).
        config=Config(signature_version=UNSIGNED),
    )


def list_common_prefixes(client, prefix, bucket=LANDCOVERNET_BUCKET):
    """
    List every immediate "folder" (S3 common prefix) directly under `prefix`.

    S3 doesn't really have folders -- object keys are just long strings like
    "landcovernet_na/data/v1.0/2018/10SGJ/00/S2/.../file.tif". Passing
    Delimiter="/" makes the API group keys by everything up to the next "/"
    after `prefix`, which simulates one level of folder listing without
    having to fetch every single object key up front.
    """
    # A paginator automatically handles S3's page-by-page responses (S3 caps
    # each response at 1000 keys), so we don't have to manually track
    # "NextContinuationToken" ourselves.
    paginator = client.get_paginator("list_objects_v2")

    prefixes = []  # will hold every discovered "folder path" string
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        # "CommonPrefixes" is where S3 puts the folder-like groupings when
        # a Delimiter is specified; it's a list of dicts like {"Prefix": "..."}
        for cp in page.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])
    return prefixes


def discover(client, sample_size=10):
    """
    Print the real bucket layout so you can sanity-check naming before
    committing to a full download. Run with `--discover` to call this.

    Descends: product root -> data/v1.0/2018/ -> one tile -> one chip,
    printing every level so you can see exactly how deep things are nested
    and what filenames look like at the bottom.
    """
    print(f"--- Tiles under {LANDCOVERNET_NA_DATA_PREFIX} ---")
    tile_prefixes = list_common_prefixes(client, LANDCOVERNET_NA_DATA_PREFIX)
    print(f"Found {len(tile_prefixes)} tile folders.\n")
    for p in tile_prefixes[:sample_size]:
        print(" ", p)

    if not tile_prefixes:
        return tile_prefixes

    first_tile = tile_prefixes[0]
    print(f"\n--- Chip folders under {first_tile} ---")
    chip_prefixes = list_common_prefixes(client, first_tile)
    for p in chip_prefixes[:sample_size]:
        print(" ", p)

    if not chip_prefixes:
        return tile_prefixes

    first_chip = chip_prefixes[0]
    print(f"\n--- FULL recursive contents of {first_chip} ---")
    # No Delimiter here -- this returns every object key nested at any depth
    # under first_chip, which is exactly what we need to see where the
    # labels file lives relative to the S2/S1/L8 imagery subfolders.
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=LANDCOVERNET_BUCKET, Prefix=first_chip):
        for obj in page.get("Contents", []):
            print("   -", obj["Key"], f"({obj['Size']} bytes)")

    return tile_prefixes


def mgrs_tile_from_key(key: str):
    """
    Given an S3 key/folder name, extract the embedded MGRS tile ID (e.g.
    "10SGJ") if present, or return None if no MGRS-shaped substring is found.
    """
    m = MGRS_TILE_RE.search(key)
    if not m:
        return None
    zone, band, square = m.groups()
    # int(zone) strips any leading zero (e.g. "05" -> "5") so tile IDs are
    # formatted consistently no matter how they appeared in the original key.
    return f"{int(zone)}{band}{square}"


def tile_centroid_latlon(tile_id: str):
    """
    Approximate the centroid of an MGRS 100km grid square using its zone/band/
    square designator, by asking for the point at (50km, 50km) within that
    square -- i.e. dead center, since each square is 100km x 100km.
    """
    m = mgrs.MGRS()

    # A "full precision" MGRS string is: zone+band+square, followed by a
    # 5-digit easting offset and a 5-digit northing offset (each ranging from
    # 00000 to 99999, representing meters within the 100km square). We hard
    # code "50000" + "50000" to always ask for the centroid of the square,
    # regardless of which specific chip within the tile we're looking at.
    full = f"{tile_id}5000050000"
    try:
        lat, lon = m.toLatLon(full)
        return lat, lon
    except Exception:
        # toLatLon can raise on malformed or out-of-range MGRS strings (e.g.
        # zones near the poles have irregular grid square rules); treat any
        # failure as "couldn't determine location" rather than crashing.
        return None


def coarse_zone_band_ok(tile_id: str):
    """
    Fast pre-filter: check just the UTM zone number + latitude band letter
    (the first 3-4 characters of the tile ID) against the CONUS-overlapping
    zones/bands defined in config.py. This lets us skip the more expensive
    mgrs.toLatLon() call entirely for tiles that are obviously not in CONUS
    (e.g. anything in Europe, Africa, or far northern Canada).
    """
    match = re.match(r"(\d{1,2})([A-Z])", tile_id)
    if not match:
        return False
    zone, band = int(match.group(1)), match.group(2)
    return zone in CONUS_UTM_ZONES and band in CONUS_MGRS_BANDS


def in_conus_bbox(lat, lon):
    """Check whether a (lat, lon) point falls inside the CONUS bounding box."""
    return (
        CONUS_BBOX["min_lat"] <= lat <= CONUS_BBOX["max_lat"]
        and CONUS_BBOX["min_lon"] <= lon <= CONUS_BBOX["max_lon"]
    )


def find_conus_chips(client):
    """
    Full pipeline: list every tile, keep only tiles inside CONUS, then list
    the chip (00, 01, 02, ...) subfolders within each surviving tile.

    Filtering at the TILE level first (before listing chips) means we only
    pay for a second round of S3 list calls on tiles that already passed the
    CONUS check -- much cheaper than listing every chip in North America
    up front.

    Returns a list of dicts: {"prefix": ..., "tile_id": ..., "chip_id": ...}
    """
    tile_prefixes = list_common_prefixes(client, LANDCOVERNET_NA_DATA_PREFIX)
    print(f"Found {len(tile_prefixes)} tile folders in landcovernet_na.")

    conus_tiles = []  # (tile_prefix, tile_id) pairs that pass the CONUS filter
    for tile_prefix in tile_prefixes:
        tile_id = mgrs_tile_from_key(tile_prefix)
        if tile_id is None:
            # Couldn't parse an MGRS tile out of this folder name; skip it
            # rather than guessing.
            continue

        if not coarse_zone_band_ok(tile_id):
            # Cheap rejection: this tile's zone/band can't possibly be in
            # CONUS, so don't even bother computing its exact centroid.
            continue

        latlon = tile_centroid_latlon(tile_id)
        if latlon and in_conus_bbox(*latlon):
            conus_tiles.append((tile_prefix, tile_id))

    print(f"{len(conus_tiles)} tiles fall inside the CONUS bbox: "
          f"{sorted(t for _, t in conus_tiles)}")

    # Now, for each CONUS-passing tile, list its chip (00, 01, ...) subfolders.
    chips = []
    for tile_prefix, tile_id in conus_tiles:
        chip_prefixes = list_common_prefixes(client, tile_prefix)
        for chip_prefix in chip_prefixes:
            # The chip ID is just the last path segment, e.g.
            # ".../10SGJ/00/" -> "00". Splitting on "/" and dropping the
            # trailing empty string (since the prefix ends in "/") gets us
            # that last segment cleanly.
            chip_id = chip_prefix.rstrip("/").split("/")[-1]
            chips.append({"prefix": chip_prefix, "tile_id": tile_id, "chip_id": chip_id})

    print(f"{len(chips)} chip folders total across those tiles.")
    return chips


def download_chip(client, chip, dest_dir, bucket=LANDCOVERNET_BUCKET, max_workers=16):
    """
    Download every object nested under one chip's S3 prefix (imagery,
    labels, whatever else lives there) into a local folder named
    "<tile_id>_<chip_id>" directly under dest_dir -- e.g. "10SGJ_00" --
    which is the flat, one-folder-per-chip layout that
    02_build_training_table.py expects.

    ------------------------------------------------------------------------
    WHY THREADS, AND WHAT IS A "ThreadPoolExecutor"?
    ------------------------------------------------------------------------
    A single chip folder can contain 600+ small files (Sentinel-2, Landsat-8,
    and Sentinel-1 bands across dozens of observation dates). Downloading
    them ONE AT A TIME, waiting for each to fully finish before starting the
    next, means your program spends almost all of its time just sitting
    idle waiting for the network -- the actual CPU work of writing a ~130KB
    file to disk takes almost no time at all compared to the round-trip
    delay of "ask the server for this file, wait for the reply."

    That waiting time is exactly what threads are good for. A "thread" is a
    separate, independent line of execution that can run concurrently with
    others inside the same program. If we kick off, say, 16 downloads on 16
    different threads at once, they can all be "waiting on the network"
    AT THE SAME TIME instead of one after another -- so instead of the total
    time being (600 files x latency-per-file), it's closer to
    (600 files / 16 threads) x latency-per-file. Roughly a 16x speedup for
    this kind of "mostly waiting on I/O" workload.

    (Side note for anyone who's heard of Python's "GIL" -- the Global
    Interpreter Lock, which normally prevents two Python threads from
    executing Python bytecode at the exact same instant: that restriction
    doesn't really hurt us here, because while a thread is blocked waiting
    on a network response, it releases the GIL and lets other threads run.
    Threading is a great fit for "I/O-bound" work like this; it would NOT
    help much for "CPU-bound" work like heavy number-crunching, where you'd
    want multiprocessing instead.)

    A `ThreadPoolExecutor` is just a manager that maintains a fixed-size
    "pool" of worker threads and hands out queued tasks to whichever worker
    is free next, so you don't have to manually create/track/clean up
    threads yourself. `max_workers=16` below means "keep at most 16
    downloads in flight at once" -- a reasonable balance between speed and
    not hammering the server with too many simultaneous connections.

    (boto3's S3 client is documented as safe to share across multiple
    threads, so we can reuse the single `client` passed in rather than
    creating a new one per thread.)
    ------------------------------------------------------------------------
    """
    chip_prefix = chip["prefix"]
    local_chip_dir = dest_dir / f"{chip['tile_id']}_{chip['chip_id']}"

    # --- Step 1: figure out WHAT to download (fast, single-threaded) --------
    # Listing which files exist is cheap (a handful of API calls), so we do
    # that part up front, normally, before we even think about threads.
    # This builds a plain Python list of (remote_key, local_path) pairs --
    # nothing has actually been downloaded yet at this point.
    paginator = client.get_paginator("list_objects_v2")
    download_jobs = []  # each item: (key, local_path)

    for page in paginator.paginate(Bucket=bucket, Prefix=chip_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]  # full remote path, e.g. ".../10SGJ/00/S2/.../B06_10m.tif"

            # Strip the chip's own prefix so the remaining path is relative
            # to the chip folder itself, e.g. "S2/10SGJ_00_20180209/..._B06_10m.tif"
            # -- this preserves the S2/<date>/ nesting locally, just rooted
            # at our flat "10SGJ_00" folder instead of the long remote path.
            rel_path = key[len(chip_prefix):]
            local_path = local_chip_dir / rel_path
            download_jobs.append((key, local_path))

    # --- Step 2: do the actual downloading, in parallel ----------------------
    def _download_one(job):
        """
        This is the function each worker thread will actually run. It
        downloads exactly ONE file. We define it as a small nested function
        so it has easy access to `client` and `bucket` from the enclosing
        scope, without needing to pass them through as extra arguments.

        IMPORTANT: this function must be safe to run on MANY DIFFERENT
        THREADS AT THE SAME TIME, for different files. It is here, because:
          - each call only touches its own local `key`/`local_path` variables
            (no shared mutable state being written to from multiple threads)
          - `local_path.parent.mkdir(parents=True, exist_ok=True)` is safe to
            call redundantly from multiple threads for the same folder --
            exist_ok=True means it won't raise an error if another thread
            already created that folder a moment earlier
          - boto3 clients are documented as thread-safe for exactly this
            kind of concurrent use
        If any of those weren't true, running this on multiple threads could
        cause race conditions (two threads corrupting shared state by writing
        to it at the same time) -- so keep that in mind if you ever modify
        this function to, say, append results to a shared list instead of
        returning them.
        """
        key, local_path = job
        # Create any missing parent directories before writing into them.
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # download_file streams the object straight to disk (rather than
        # loading the whole thing into memory first).
        client.download_file(bucket, key, str(local_path))
        return local_path

    downloaded = []
    failed = []

    # The "with ... as executor:" block below is a context manager: it
    # creates the thread pool, and GUARANTEES that when we leave the block
    # (normally, or because of an error), the executor waits for all
    # submitted work to finish and cleans up its worker threads properly.
    # You don't have to manually call executor.shutdown() -- the "with"
    # statement does that for you automatically.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.submit(fn, arg) hands one task off to the pool and
        # returns IMMEDIATELY with a "Future" object -- a placeholder that
        # represents "the result of this task, whenever it's done." At this
        # point, none of the downloads have necessarily started yet, and
        # none have necessarily finished; submit() just enqueues the work.
        #
        # We build a dict mapping {future: job} so that later, when a
        # future completes, we can still look up which (key, local_path)
        # it was for -- handy for error messages.
        future_to_job = {
            executor.submit(_download_one, job): job
            for job in download_jobs
        }

        # as_completed(...) yields each Future as soon as IT finishes,
        # in whatever order they happen to complete in (NOT necessarily
        # the order we submitted them) -- e.g. if worker thread #3 finishes
        # its file before worker thread #1 does, we'll see #3's future
        # first here. This lets us start processing results as they arrive
        # instead of waiting for every single one before touching any of
        # them.
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                # future.result() gives us the return value of
                # _download_one() for this job -- or RE-RAISES whatever
                # exception happened inside that thread, right here in the
                # main thread, so we can catch it with a normal try/except
                # instead of it silently disappearing inside a worker thread.
                local_path = future.result()
                downloaded.append(local_path)
            except Exception as exc:
                key, _ = job
                print(f"    !! failed to download {key}: {exc}")
                failed.append(job)

    if failed:
        print(f"    ({len(failed)} of {len(download_jobs)} files in this chip failed to download)")

    return downloaded


def main():
    # --- Define the command-line interface ----------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-chips", type=int, default=15,
                         help="Number of random CONUS chips to download")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed, so the same --n-chips value always "
                              "picks the same chips (reproducibility)")
    parser.add_argument("--dest", type=Path, default=Path("../data/raw"),
                         help="Local folder to download chips into")
    parser.add_argument("--max-workers", type=int, default=16,
                         help="How many files to download at once (per chip) using a "
                              "thread pool. Higher = faster, up to a point, but too "
                              "high can overwhelm your connection or the server -- "
                              "16 is a reasonable default.")
    parser.add_argument("--discover", action="store_true",
                         help="Just print the bucket layout, don't download anything")
    args = parser.parse_args()

    # Build the anonymous S3 client once and reuse it for every request below.
    client = get_client()

    if args.discover:
        # --discover mode: print what's actually in the bucket, then exit
        # immediately without downloading anything. Use this first!
        discover(client)
        return

    # Find every (tile, chip) combination whose tile falls inside CONUS.
    conus_chips = find_conus_chips(client)
    if not conus_chips:
        print("No CONUS chips found -- run with --discover first to check the "
              "bucket layout, the naming convention may have changed.")
        return

    # Seed Python's random module so `--seed 42` always produces the exact
    # same sample of chips, making the download (and everything downstream)
    # reproducible.
    random.seed(args.seed)
    # random.sample() picks unique items without replacement; min(...) guards
    # against asking for more chips than actually exist.
    sample = random.sample(conus_chips, min(args.n_chips, len(conus_chips)))

    # Make sure the destination folder exists before we start writing into it.
    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(sample)} chips to {args.dest.resolve()} "
          f"(up to {args.max_workers} files at a time per chip) ...")
    for chip in sample:
        local_name = f"{chip['tile_id']}_{chip['chip_id']}"
        print(f"  {chip['prefix']}  ->  {args.dest / local_name}")
        download_chip(client, chip, args.dest, max_workers=args.max_workers)

    print("\nDone. Downloaded chips:")
    for chip in sample:
        print(" ", f"{chip['tile_id']}_{chip['chip_id']}")


# Standard Python idiom: only run main() if this file is executed directly
# (e.g. `python 01_download_conus_chips.py`), not if it's imported as a
# module from somewhere else.
if __name__ == "__main__":
    main()
