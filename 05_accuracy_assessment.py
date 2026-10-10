"""
Accuracy assessment:
  1. Confusion matrix + per-class metrics on the held-out LandCoverNet test chips
     (from 03_train_model.py's test_predictions.parquet)
  2. A qualitative comparison to NLCD for the new area classified in
     04_apply_to_new_area.py

NLCD download
-------------
Grab the NLCD tile covering your new-area bbox from MRLC:
    https://www.mrlc.gov/viewer/  (or the bulk downloads at https://www.mrlc.gov/data)
Reproject/clip it to match your classified GeoTIFF's grid with e.g.:
    gdalwarp -t_srs EPSG:XXXX -te <minx> <miny> <maxx> <maxy> -tr 10 10 \\
        nlcd_input.tif nlcd_clipped.tif
"""

# --- Standard library imports -----------------------------------------------
import argparse
from pathlib import Path

# --- Third-party imports -----------------------------------------------------
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import sys
sys.path.append(str(Path(__file__).parent))
from config import LANDCOVERNET_CLASSES, LANDCOVERNET_TO_NLCD_GROUP

# ---------------------------------------------------------------------------
# NLCD's own numeric class codes, mapped to the same coarse group names used
# in LANDCOVERNET_TO_NLCD_GROUP (config.py), so the two products can be
# compared on equal footing. NLCD 2019/2021-era codes shown here; adjust if
# you're using a different NLCD version with different codes.
#   1x = water/ice, 2x = developed, 3x = barren, 4x = forest,
#   5x/7x = shrub/grassland, 8x = agriculture, 9x = wetlands
# ---------------------------------------------------------------------------
NLCD_GROUPS = {
    11: "Water", 12: "Snow/Ice",
    21: "Developed", 22: "Developed", 23: "Developed", 24: "Developed",
    31: "Barren",
    41: "Forest", 42: "Forest", 43: "Forest",
    51: "Grassland/Shrubland", 52: "Grassland/Shrubland",
    71: "Grassland/Shrubland", 72: "Grassland/Shrubland", 73: "Grassland/Shrubland", 74: "Grassland/Shrubland",
    81: "Cultivated", 82: "Cultivated",
    90: "Forest", 95: "Grassland/Shrubland",
}


def landcovernet_confusion_matrix(pred_path: Path, out_dir: Path):
    """
    Build and save a confusion matrix + classification report comparing the
    model's predictions to ground truth on the held-out LandCoverNet test
    chips (produced by 03_train_model.py).
    """
    df = pd.read_parquet(pred_path)

    # Fix the label ordering (1-7) so the confusion matrix rows/columns are
    # always in the same, predictable order, regardless of which classes
    # happen to appear in this particular test set.
    labels = sorted(LANDCOVERNET_CLASSES.keys())
    names = [LANDCOVERNET_CLASSES[l] for l in labels]

    # confusion_matrix[i, j] = number of pixels whose TRUE class is labels[i]
    # and PREDICTED class is labels[j]. Perfect predictions would show all
    # the "weight" on the diagonal.
    cm = confusion_matrix(df["label"], df["prediction"], labels=labels)

    # ConfusionMatrixDisplay handles drawing the matrix as a labeled,
    # color-shaded grid via matplotlib.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved confusion matrix figure to {fig_path}")

    # Text report: precision/recall/F1/support per class, plus overall
    # accuracy -- both printed to the console and saved to a .txt file.
    report = classification_report(df["label"], df["prediction"], labels=labels, target_names=names)
    print(report)

    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)
    print(f"Saved text report to {report_path}")


def compare_to_nlcd(classified_path: Path, nlcd_path: Path, out_dir: Path):
    """
    Compare our classified GeoTIFF for a new area against an NLCD tile
    covering the same area, at the coarse "group" level (both products'
    classes get mapped into ~7 shared categories) since their native
    legends don't match one-to-one.
    """
    with rasterio.open(classified_path) as src:
        classified = src.read(1)  # our model's predicted class per pixel
    with rasterio.open(nlcd_path) as src:
        nlcd = src.read(1)         # NLCD's class code per pixel

    if classified.shape != nlcd.shape:
        # The two rasters MUST be pixel-aligned (same shape, same grid) for
        # a per-pixel comparison to make sense -- fail with clear
        # instructions rather than silently comparing misaligned pixels.
        raise SystemExit(
            f"Shape mismatch: classified={classified.shape}, nlcd={nlcd.shape}. "
            "Reproject/clip the NLCD tile to match your classified grid first "
            "(see the gdalwarp command in this script's docstring)."
        )

    # np.vectorize turns a plain Python function into one that can be applied
    # element-wise across a whole numpy array. Here we translate:
    #   our integer class code -> LandCoverNet class name -> shared group name
    lcn_group = np.vectorize(lambda v: LANDCOVERNET_TO_NLCD_GROUP.get(
        LANDCOVERNET_CLASSES.get(int(v), None), "unknown"))(classified)
    # ...and separately: NLCD's integer class code -> shared group name
    nlcd_group = np.vectorize(lambda v: NLCD_GROUPS.get(int(v), "unknown"))(nlcd)

    # Only compare pixels where BOTH products produced a recognized class
    # (excludes nodata pixels and any class codes not in our lookup tables).
    valid = (lcn_group != "unknown") & (nlcd_group != "unknown")
    # Fraction of valid pixels where the two products agree on the group.
    agreement = (lcn_group[valid] == nlcd_group[valid]).mean()

    print(f"\nCrosswalked group-level agreement with NLCD: {agreement:.1%}")
    print("(This compares 7 LandCoverNet classes grouped into ~7 coarse categories "
          "against NLCD's 16 classes grouped the same way -- a methodology "
          "comparison, not a strict accuracy score, since the two products use "
          "different sensors, years, and legends.)")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "nlcd_comparison.txt"
    summary_path.write_text(
        f"Group-level agreement with NLCD: {agreement:.1%}\n"
        f"Valid comparison pixels: {valid.sum():,} / {valid.size:,}\n"
    )
    print(f"Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-predictions", type=Path,
                         default=Path("../data/test_predictions.parquet"),
                         help="Test predictions saved by 03_train_model.py")
    parser.add_argument("--classified", type=Path,
                         default=Path("../data/classified_new_area.tif"),
                         help="Classified GeoTIFF from 04_apply_to_new_area.py")
    parser.add_argument("--nlcd", type=Path, default=None,
                         help="Path to an NLCD GeoTIFF clipped/reprojected to match --classified")
    parser.add_argument("--out-dir", type=Path, default=Path("../data/assessment"),
                         help="Where to save figures and text reports")
    args = parser.parse_args()

    # Part 1: confusion matrix on held-out LandCoverNet chips (always runs
    # if the test predictions file exists).
    if args.test_predictions.exists():
        landcovernet_confusion_matrix(args.test_predictions, args.out_dir)
    else:
        print(f"No test predictions found at {args.test_predictions}, skipping confusion matrix.")

    # Part 2: NLCD comparison (only runs if the user passed --nlcd, since it
    # requires a manually downloaded/clipped NLCD tile).
    if args.nlcd:
        compare_to_nlcd(args.classified, args.nlcd, args.out_dir)
    else:
        print("\nNo --nlcd path given, skipping NLCD comparison. Download a tile from "
              "https://www.mrlc.gov/data covering your new area and pass it in.")


if __name__ == "__main__":
    main()
