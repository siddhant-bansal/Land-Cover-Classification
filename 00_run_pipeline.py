"""
Master pipeline script -- runs all 5 project steps in order, from one command.

This script does NOT reimplement any of the pipeline logic itself. Instead,
it calls each of the five numbered scripts (01 through 05) one after
another, as separate command-line programs, passing along whichever
parameters you give it. Think of it as a conductor: it doesn't play any
instruments itself, it just tells each section when to come in and what
to play.

WHY SUBPROCESSES, NOT IMPORTS?
------------------------------------------------------------------------
An alternative design would be to `import` each of the 01-05 scripts as
Python modules and call their functions directly, all inside one running
Python process. We deliberately DON'T do that here, for a few reasons that
are worth understanding if you're new to this pattern:

  1. Each of the 01-05 scripts is written to be run "standalone" (they each
     have their own `if __name__ == "__main__": main()` block and their own
     argparse setup). Re-using them as standalone COMMAND-LINE PROGRAMS,
     via Python's `subprocess` module, means this master script doesn't need
     to know anything about their internals -- it just needs to know what
     command-line flags each one accepts. If someone edits one of those
     scripts later, this master script keeps working without changes, as
     long as the flags stay the same.
  2. Running each step as its OWN separate process means each one gets a
     clean memory space. Step 2 (building a half-million-row pixel table)
     doesn't leave any memory lying around for Step 3 to trip over, because
     Step 2's entire process exits and is cleaned up by the operating
     system before Step 3 even starts.
  3. If one step crashes, it crashes ITS OWN process, not this master
     script's process -- so we can catch that cleanly (via the process's
     "return code") and decide whether to stop the whole pipeline or keep
     going, without a crash in, say, scikit-learn bringing down our
     orchestration logic too.

Usage examples
--------------
    # Run the entire pipeline start to finish:
    python 00_run_pipeline.py --n-chips 15 --seed 42 \\
        --bbox -122.55 37.60 -122.35 37.80

    # Just look at the bucket layout first, without downloading or running
    # anything else (equivalent to step 1's own --discover flag):
    python 00_run_pipeline.py --discover

    # Skip the download+table-building steps because you already have them,
    # and just retrain + re-apply + re-assess:
    python 00_run_pipeline.py --steps 3,4,5 --bbox -122.55 37.60 -122.35 37.80

    # See exactly what commands WOULD run, without actually running them:
    python 00_run_pipeline.py --n-chips 15 --bbox -122.55 37.60 -122.35 37.80 --dry-run
"""

# --- Standard library imports -----------------------------------------------
import argparse     # command-line flag parsing
import subprocess   # lets us launch and manage separate OS-level processes
import sys           # sys.executable (below) and sys.exit for clean error exits
from pathlib import Path


def build_step_commands(args, script_dir: Path):
    """
    Translate the master script's parsed arguments into the exact list of
    command-line invocations for each of the 5 pipeline steps.

    Returns a list of (step_number, description, command) tuples, where
    `command` is a list of strings suitable for subprocess.run(), e.g.:
        [sys.executable, "01_download_conus_chips.py", "--n-chips", "15", ...]

    Shared file paths (the training table, the trained model, etc.) are
    computed ONCE here and reused as both "where step N writes its output"
    and "where step N+1 reads its input from" -- this guarantees the steps
    actually hand off to each other correctly, instead of relying on you to
    separately type matching paths for a producer flag and a consumer flag.
    """
    # `python` here is whichever Python interpreter you want each step to
    # run under. We default to `sys.executable` -- the exact same
    # interpreter currently running THIS master script -- so that if you
    # launched this script from inside a virtual environment, each step
    # automatically runs inside that same virtual environment too, without
    # you having to specify it twice. `--python` lets you override that if
    # you ever need a different interpreter.
    python = args.python

    # These three paths get reused across step boundaries, so we compute
    # them once here rather than letting each step guess independently.
    raw_dir = args.raw_dir
    training_table = args.training_table
    model_path = args.model_path
    # 03_train_model.py always saves its held-out test predictions as a
    # file named "test_predictions.parquet" sitting right next to whatever
    # --out path you gave it (see that script's `args.out.with_name(...)`
    # line) -- we replicate that exact same naming rule here so Step 5 knows
    # where to find it without us having to invent a brand new flag just to
    # pass that path along.
    test_predictions_path = model_path.with_name("test_predictions.parquet")
    classified_path = args.classified_path

    commands = []

    # ------------------------------------------------------------------ #
    # Step 1: download a random sample of CONUS chips from LandCoverNet
    # ------------------------------------------------------------------ #
    step1_cmd = [
        python, str(script_dir / "01_download_conus_chips.py"),
        "--n-chips", str(args.n_chips),
        "--seed", str(args.seed),
        "--dest", str(raw_dir),
        "--max-workers", str(args.max_workers),
    ]
    if args.discover:
        # --discover is special: it tells step 1 to just print the bucket
        # layout and exit, without downloading anything. We still build the
        # rest of the commands list below (in case --discover was combined
        # with --steps for some reason), but main() special-cases this flag
        # to run ONLY step 1 and then stop -- see the big comment in main().
        step1_cmd.append("--discover")
    commands.append((1, "Download CONUS chips from LandCoverNet", step1_cmd))

    # ------------------------------------------------------------------ #
    # Step 2: build the flat per-pixel training table from downloaded chips
    # ------------------------------------------------------------------ #
    step2_cmd = [
        python, str(script_dir / "02_build_training_table.py"),
        "--raw-dir", str(raw_dir),
        "--out", str(training_table),
        "--sample-per-class", str(args.sample_per_class),
    ]
    commands.append((2, "Build per-pixel training table", step2_cmd))

    # ------------------------------------------------------------------ #
    # Step 3: train the Random Forest / Gradient Boosting classifier
    # ------------------------------------------------------------------ #
    step3_cmd = [
        python, str(script_dir / "03_train_model.py"),
        "--table", str(training_table),
        "--model", args.model_type,       # "rf" or "gbm" -- the ALGORITHM choice
        "--test-size", str(args.test_size),
        "--seed", str(args.train_seed),
        "--out", str(model_path),
    ]
    commands.append((3, f"Train {args.model_type.upper()} classifier", step3_cmd))

    # ------------------------------------------------------------------ #
    # Step 4: apply the trained model to a new area via fresh Sentinel-2
    # ------------------------------------------------------------------ #
    # We build this command's list even when --bbox wasn't supplied, because
    # build_step_commands() always constructs all 5 steps up front (so
    # --dry-run can show you the whole pipeline, and so main() can freely
    # pick a SUBSET of steps to actually run via --steps). main() already
    # refuses to proceed (via parser.error) if step 4 is requested without
    # --bbox -- but if step 4 ISN'T requested, args.bbox can legitimately
    # still be None here, so we only add the --bbox flag when we actually
    # have a value, rather than unconditionally indexing into it.
    step4_cmd = [
        python, str(script_dir / "04_apply_to_new_area.py"),
    ]
    if args.bbox is not None:
        # argparse's nargs=4 flag expects 4 separate string arguments, not
        # one combined string -- so we pass each bbox number individually.
        step4_cmd += ["--bbox", str(args.bbox[0]), str(args.bbox[1]),
                      str(args.bbox[2]), str(args.bbox[3])]
    step4_cmd += [
        "--date-range", args.date_range,
        "--max-cloud", str(args.max_cloud),
        "--model", str(model_path),       # the trained model FILE this time, not the algorithm name
        "--out", str(classified_path),
    ]
    commands.append((4, "Classify new area from fresh Sentinel-2 imagery", step4_cmd))

    # ------------------------------------------------------------------ #
    # Step 5: confusion matrix + NLCD comparison
    # ------------------------------------------------------------------ #
    step5_cmd = [
        python, str(script_dir / "05_accuracy_assessment.py"),
        "--test-predictions", str(test_predictions_path),
        "--classified", str(classified_path),
        "--out-dir", str(args.assessment_dir),
    ]
    if args.nlcd is not None:
        # --nlcd is optional both here and in 05_accuracy_assessment.py
        # itself -- only pass it along if the user actually gave us one.
        step5_cmd += ["--nlcd", str(args.nlcd)]
    commands.append((5, "Accuracy assessment (confusion matrix + NLCD comparison)", step5_cmd))

    return commands


def run_command(step_number, description, command, dry_run=False):
    """
    Actually launch one step as a subprocess, stream its output live, and
    report whether it succeeded.

    Returns True if the step succeeded, False if it failed.
    """
    print(f"\n{'=' * 70}")
    print(f"STEP {step_number}: {description}")
    print(f"{'=' * 70}")
    # Printing the exact command lets you copy-paste it and re-run just
    # that one step by hand later, which is handy for debugging.
    print("Command:", " ".join(command))

    if dry_run:
        # --dry-run mode: show what WOULD happen, but don't actually launch
        # anything. Useful for sanity-checking your flags before committing
        # to a multi-hour download/training run.
        print("(--dry-run: not actually executing)")
        return True

    # subprocess.run() launches `command` as a brand new OS process, waits
    # for it to finish, and gives us back a "CompletedProcess" object. By
    # NOT passing capture_output=True, we let the child process's normal
    # print() output flow straight through to our own terminal in real
    # time, exactly as if you'd typed the command yourself -- this is what
    # lets you see each step's progress bars/printouts live instead of
    # everything appearing in one big dump only after it's done.
    #
    # check=False (the default) means subprocess.run() will NOT raise an
    # exception just because the child process exited with an error code --
    # instead, it just records that exit code on the result object, and we
    # decide for ourselves (right below) what to do about it.
    result = subprocess.run(command, check=False)

    # By convention, a process that finished successfully exits with
    # return code 0; any non-zero code signals some kind of failure.
    if result.returncode != 0:
        print(f"\n!! Step {step_number} ({description}) failed "
              f"with exit code {result.returncode}.")
        return False

    print(f"\nStep {step_number} completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,  # preserves the
                                                                 # usage examples'
                                                                 # formatting above
    )

    # ------------------------------------------------------------------ #
    # Pipeline control
    # ------------------------------------------------------------------ #
    control = parser.add_argument_group("pipeline control")
    control.add_argument(
        "--steps", type=str, default="1,2,3,4,5",
        help="Comma-separated list of step numbers to run, in order, e.g. "
             "'3,4,5' to skip downloading/table-building and just retrain. "
             "Default: all 5 steps.",
    )
    control.add_argument(
        "--discover", action="store_true",
        help="Just run step 1 in --discover mode (print the LandCoverNet "
             "bucket layout) and exit immediately -- skips every other step "
             "regardless of --steps.",
    )
    control.add_argument(
        "--dry-run", action="store_true",
        help="Print each step's command without actually running it.",
    )
    control.add_argument(
        "--continue-on-error", action="store_true",
        help="By default, the pipeline stops at the first step that fails. "
             "Pass this to keep going with later steps anyway (rarely what "
             "you want, since later steps usually depend on earlier ones' "
             "output -- mainly useful if you're re-running a subset of "
             "steps you know already succeeded).",
    )
    control.add_argument(
        "--python", type=str, default=sys.executable,
        help="Path to the Python interpreter used to run each step. "
             "Defaults to whichever interpreter is running this script.",
    )

    # ------------------------------------------------------------------ #
    # Shared paths (reused across step boundaries -- see build_step_commands)
    # ------------------------------------------------------------------ #
    paths = parser.add_argument_group("shared file paths")
    paths.add_argument(
        "--raw-dir", type=Path, default=Path("../data/raw"),
        help="Where step 1 downloads chips to, and where step 2 reads them from.",
    )
    paths.add_argument(
        "--training-table", type=Path, default=Path("../data/training_table.parquet"),
        help="Where step 2 saves the pixel feature table, and where step 3 reads it from.",
    )
    paths.add_argument(
        "--model-path", type=Path, default=Path("../data/model.joblib"),
        help="Where step 3 saves the trained model, and where step 4 reads it from.",
    )
    paths.add_argument(
        "--classified-path", type=Path, default=Path("../data/classified_new_area.tif"),
        help="Where step 4 saves the classified GeoTIFF, and where step 5 reads it from.",
    )
    paths.add_argument(
        "--assessment-dir", type=Path, default=Path("../data/assessment"),
        help="Where step 5 saves its confusion matrix figure and text reports.",
    )

    # ------------------------------------------------------------------ #
    # Step 1 parameters (01_download_conus_chips.py)
    # ------------------------------------------------------------------ #
    step1 = parser.add_argument_group("step 1: download CONUS chips")
    step1.add_argument("--n-chips", type=int, default=15,
                        help="Number of random CONUS chips to download.")
    step1.add_argument("--seed", type=int, default=42,
                        help="Random seed for picking which chips to download.")
    step1.add_argument("--max-workers", type=int, default=16,
                        help="How many files to download at once (per chip) via a thread pool.")

    # ------------------------------------------------------------------ #
    # Step 2 parameters (02_build_training_table.py)
    # ------------------------------------------------------------------ #
    step2 = parser.add_argument_group("step 2: build training table")
    step2.add_argument("--sample-per-class", type=int, default=20000,
                        help="Max pixels per class per chip, to keep the table manageable.")

    # ------------------------------------------------------------------ #
    # Step 3 parameters (03_train_model.py)
    # ------------------------------------------------------------------ #
    step3 = parser.add_argument_group("step 3: train model")
    step3.add_argument("--model-type", choices=["rf", "gbm"], default="rf",
                        help="rf = Random Forest, gbm = Histogram Gradient Boosting. "
                             "(Note: this is the ALGORITHM choice -- not to be confused "
                             "with --model-path above, which is a FILE location.)")
    step3.add_argument("--test-size", type=float, default=0.25,
                        help="Fraction of CHIPS (not pixels) held out for testing.")
    step3.add_argument("--train-seed", type=int, default=42,
                        help="Random seed for the chip-grouped train/test split.")

    # ------------------------------------------------------------------ #
    # Step 4 parameters (04_apply_to_new_area.py)
    # ------------------------------------------------------------------ #
    step4 = parser.add_argument_group("step 4: apply to new area")
    step4.add_argument(
        "--bbox", type=float, nargs=4, default=None,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box of the new area to classify, in WGS84 lon/lat. "
             "REQUIRED if step 4 is included in --steps.",
    )
    step4.add_argument("--date-range", type=str, default="2024-06-01/2024-09-30",
                        help="STAC datetime range, ISO 8601 'start/end', for the Sentinel-2 search.")
    step4.add_argument("--max-cloud", type=float, default=20,
                        help="Maximum acceptable scene-level cloud cover percentage.")

    # ------------------------------------------------------------------ #
    # Step 5 parameters (05_accuracy_assessment.py)
    # ------------------------------------------------------------------ #
    step5 = parser.add_argument_group("step 5: accuracy assessment")
    step5.add_argument(
        "--nlcd", type=Path, default=None,
        help="Optional: path to an NLCD GeoTIFF clipped/reprojected to match "
             "--classified-path, for the NLCD comparison. If omitted, that "
             "comparison is skipped.",
    )

    args = parser.parse_args()

    # --- Validate / finish setting up the parsed arguments -------------------
    # Turn "3,4,5" into the Python list [3, 4, 5]. We do this by hand
    # instead of using argparse's built-in list parsing so that the
    # --steps flag can stay a simple, easy-to-type comma-separated string
    # on the command line (e.g. `--steps 3,4,5`) rather than requiring
    # awkward repeated flags or quoting.
    try:
        requested_steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    except ValueError:
        parser.error(f"--steps must be a comma-separated list of integers, got: {args.steps!r}")

    for s in requested_steps:
        if s not in (1, 2, 3, 4, 5):
            parser.error(f"--steps contains invalid step number {s} (must be 1-5)")

    # --bbox is only truly required if step 4 is actually going to run --
    # we can't make it a plain `required=True` argparse argument, because
    # that would force you to supply it even when running e.g. `--steps 1,2`
    # (just downloading and building the table), where it's irrelevant.
    if 4 in requested_steps and not args.discover and args.bbox is None:
        parser.error("--bbox is required when step 4 is included in --steps "
                     "(e.g. --bbox -122.55 37.60 -122.35 37.80)")

    # All five numbered scripts live in the same folder as this master
    # script -- Path(__file__).parent resolves that folder regardless of
    # what directory you happen to run this command from.
    script_dir = Path(__file__).parent

    # --- Handle --discover as a special, pipeline-bypassing case --------------
    if args.discover:
        commands = build_step_commands(args, script_dir)
        step_number, description, command = commands[0]  # step 1's command
        run_command(step_number, description, command, dry_run=args.dry_run)
        return

    # --- Build every step's command up front, then run only the requested ones
    all_commands = build_step_commands(args, script_dir)
    commands_to_run = [c for c in all_commands if c[0] in requested_steps]

    print(f"Running steps: {requested_steps}")

    for step_number, description, command in commands_to_run:
        success = run_command(step_number, description, command, dry_run=args.dry_run)
        if not success and not args.continue_on_error:
            print(f"\nStopping pipeline after step {step_number} failed. "
                  f"(Pass --continue-on-error to keep going anyway.)")
            sys.exit(1)

    print(f"\n{'=' * 70}")
    print("Pipeline finished.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
