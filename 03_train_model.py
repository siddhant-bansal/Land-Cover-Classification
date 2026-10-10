"""
Train a Random Forest (or HistGradientBoosting) classifier on the pixel
feature table built by 02_build_training_table.py.

Key detail: we split by *chip*, not by pixel. Randomly splitting individual
pixels lets neighboring pixels from the same chip leak into both train and
test, which inflates accuracy because nearby pixels are spatially
autocorrelated. GroupShuffleSplit on chip_id avoids that.
"""

# --- Standard library imports -----------------------------------------------
import argparse   # command-line flag parsing
import json        # saving metrics as a machine-readable .json file
from pathlib import Path

# --- Third-party imports -----------------------------------------------------
import joblib      # efficient serialization for scikit-learn models
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit  # chip-aware train/test split
from sklearn.metrics import classification_report

import sys
sys.path.append(str(Path(__file__).parent))
from config import LANDCOVERNET_CLASSES

# The full set of possible feature columns this script knows how to use.
# 02_build_training_table.py may not have produced every one of these (e.g.
# if a band was missing for some chips), so we intersect this list with
# whatever columns actually exist in the loaded table before training.
FEATURE_COLUMNS_BASE = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
                         "ndvi", "ndwi", "ndbi"]


def main():
    # --- Command-line interface ----------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("../data/training_table.parquet"),
                         help="Path to the per-pixel feature table from step 02")
    parser.add_argument("--model", choices=["rf", "gbm"], default="rf",
                         help="rf = Random Forest, gbm = Histogram Gradient Boosting")
    parser.add_argument("--test-size", type=float, default=0.25,
                         help="Fraction of CHIPS (not pixels) held out for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("../data/model.joblib"),
                         help="Where to save the trained model")
    args = parser.parse_args()

    # Load the flat pixel table produced by 02_build_training_table.py.
    df = pd.read_parquet(args.table)

    # Only use feature columns that actually exist in this table (guards
    # against a chip/band being missing upstream).
    feature_cols = [c for c in FEATURE_COLUMNS_BASE if c in df.columns]
    if not feature_cols:
        raise SystemExit("No feature columns found in the training table.")

    # Drop any pixel rows with missing values in any feature or the label --
    # scikit-learn's classifiers here can't handle NaNs.
    df = df.dropna(subset=feature_cols + ["label"])

    # X = feature matrix (n_pixels, n_features); y = integer class per pixel;
    # groups = which chip each pixel came from (used by GroupShuffleSplit
    # below to keep whole chips together during the split).
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["chip_id"].values

    # GroupShuffleSplit splits GROUPS (here: chips) into train/test, then
    # returns the pixel-level indices belonging to each group. This means
    # every pixel from a given chip ends up entirely in train OR entirely in
    # test -- never split across both, which would otherwise let the model
    # "cheat" by learning local textures from a chip it also gets tested on.
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    # next(...) grabs the single split we asked for (n_splits=1) as a
    # (train_indices, test_indices) tuple of row positions into X/y/groups.
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train pixels: {len(X_train):,} from {len(set(groups[train_idx]))} chips")
    print(f"Test pixels:  {len(X_test):,} from {len(set(groups[test_idx]))} chips")

    # --- Pick and configure the classifier ------------------------------------
    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,       # number of decision trees in the forest
            max_depth=None,          # let trees grow until leaves are pure
                                      # (or hit min_samples_split), a common
                                      # default for RF since bagging/averaging
                                      # controls overfitting on its own
            class_weight="balanced_subsample",
            # ^ re-weights each tree's bootstrap sample so rare classes
            #   (e.g. snow/ice) don't get drowned out by common ones
            #   (e.g. cultivated vegetation)
            n_jobs=-1,               # use all available CPU cores
            random_state=args.seed,  # reproducibility
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=300,            # number of boosting rounds (trees added sequentially)
            random_state=args.seed,
        )

    # --- Train and evaluate ----------------------------------------------------
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Build a lookup of {class_id: human_readable_name} sorted by class id,
    # for labeling the classification report / confusion matrix nicely.
    class_names = [LANDCOVERNET_CLASSES[c] for c in sorted(LANDCOVERNET_CLASSES)]

    # Only report on classes that actually appear in the test set predictions
    # or ground truth (avoids sklearn warnings about classes with 0 support).
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [LANDCOVERNET_CLASSES[c] for c in present_labels]

    # classification_report computes precision/recall/F1 per class, plus
    # overall accuracy; output_dict=True also gives us a dict version we can
    # save to JSON for later use (e.g. in a writeup).
    report = classification_report(y_test, y_pred, labels=present_labels,
                                    target_names=present_names, output_dict=True)
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=present_names))

    # Both RandomForest and HistGradientBoosting expose feature_importances_
    # after fitting (a rough measure of how much each feature contributed to
    # the model's splits) -- print them sorted from most to least important.
    if hasattr(clf, "feature_importances_"):
        importances = dict(zip(feature_cols, clf.feature_importances_.tolist()))
        print("\nFeature importances:")
        for feat, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
            print(f"  {feat:10s} {imp:.4f}")
    else:
        importances = None

    # --- Save everything needed to reuse this model later ----------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Save both the fitted model AND the exact list/order of feature columns
    # it expects, so 04_apply_to_new_area.py can build a matching feature
    # stack without guessing.
    joblib.dump({"model": clf, "feature_cols": feature_cols}, args.out)
    print(f"\nSaved model to {args.out}")

    # Save the classification report + feature importances as JSON, for easy
    # inclusion in a project writeup without re-running training.
    metrics_path = args.out.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"classification_report": report, "feature_importances": importances}, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Keep the held-out test set + predictions around (chip_id, row, col,
    # true label, predicted label) so 05_accuracy_assessment.py can build a
    # confusion matrix without re-running the model.
    test_df = df.iloc[test_idx][["chip_id", "row", "col", "label"]].copy()
    test_df["prediction"] = y_pred
    test_out = args.out.with_name("test_predictions.parquet")
    test_df.to_parquet(test_out, index=False)
    print(f"Saved test predictions to {test_out}")


if __name__ == "__main__":
    main()
