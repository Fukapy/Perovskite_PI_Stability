"""
Full Pipeline Runner
====================
Executes all analysis steps in the correct order.

Steps
-----
  1  data_curation    Load and curate the Perovskite Database
  2  vectorization    Build CSR feature matrices (all encoding variants)
  3  reanalysis       5-fold CV + FULL model training (Random Forest)
  4  interpretation   Feature importance at Lv1 / Lv2 / Lv3
  5  shap_analysis    SHAP values and bar chart
  6  DB_k_plot        Apparent rate constant k — 3D response surface
                      (stability mode only; skipped for PCE)

Usage
-----
  # Run all steps (stability target, default):
  python main.py

  # Run all steps with PCE as target:
  python main.py --target PCE

  # Force re-run of all steps (ignore existing outputs):
  python main.py --force

  # Manually skip specific steps:
  python main.py --skip 5 6

  # Run only specific steps:
  python main.py --steps 4 5

  # Combine options:
  python main.py --target PCE --skip 6 --force

Notes
-----
- By default, steps whose key outputs already exist are
  automatically skipped.  Use --force to override this.
- Steps 1-2 are target-independent (feature matrices are shared).
- Step 6 (DB_k_plot) is stability-specific and is automatically
  skipped when --target PCE is set.
"""

import argparse
import importlib
import os
import sys
import time
import traceback
from datetime import timedelta

import pipeline_config

# ---------------------------------------------------------------------------
# Pipeline definition
# Each entry: (step_number, module_name, description, stability_only)
# ---------------------------------------------------------------------------
STEPS = [
    (1, "data_curation",   "Data curation  — load, filter, curate dataset",    False),
    (2, "vectorization",   "Vectorization  — build all CSR feature matrices",   False),
    (3, "reanalysis",      "Reanalysis     — 5-fold CV + FULL model training",  False),
    (4, "interpretation",  "Interpretation — feature importance Lv1 / Lv2 / Lv3", False),
    (5, "shap_analysis",   "SHAP analysis  — mean SHAP values and bar chart",   False),
    (6, "DB_k_plot",       "DB k plot      — apparent rate constant surface",    True),
]


# ---------------------------------------------------------------------------
# Checkpoint files for auto-skip detection
# ---------------------------------------------------------------------------
def _checkpoint_files(step_num: int, target_mode: str) -> list[str]:
    """Return a list of key output files for the given step.

    If ALL listed files exist, the step is considered complete and can
    be auto-skipped.  Returns an empty list if no checkpoint is defined
    (step will always run).
    """
    tag = pipeline_config.target_file_tag()  # "lnTS80m" or "PCE"

    if step_num == 1:
        return ["outputs/curated/Perovskite_5800data_addln.csv"]

    if step_num == 2:
        return [
            "outputs/csr/all_sp2_dummy_zero_csr.npz",
            "outputs/csr/all_sp2_oliynyk_zero_csr.npz",
        ]

    if step_num == 3:
        return [
            "outputs/model/cv5/CV_summary_clean_table.csv",
            f"outputs/model/cv5/rf_model_{tag}_all_sp2_dummy_zero_fold5.pkl",
        ]

    if step_num == 4:
        return [
            f"outputs/model/cv5/fti_{tag}_all_sp2_dummy_zero_cv5mean.csv",
        ]

    if step_num == 5:
        return [
            f"outputs/model/shap/shap_mean_bar_{tag}_all_sp2_dummy_zero_FULL.png",
        ]

    if step_num == 6:
        return ["outputs/figures/DB_k_response_surface.png"]

    return []


def _step_is_complete(step_num: int, target_mode: str) -> bool:
    """Check whether ALL checkpoint files for the step exist."""
    files = _checkpoint_files(step_num, target_mode)
    if not files:
        return False
    return all(os.path.exists(f) for f in files)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _header(step_num: int, description: str) -> None:
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  STEP {step_num}  |  {description}")
    print(f"{bar}")


def _fmt_elapsed(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def run_steps(step_numbers: list[int], target_mode: str,
              force: bool = False) -> None:
    """Import each requested module and call its main() in order."""
    results: dict[int, str] = {}
    total_start = time.perf_counter()

    for step_num, module_name, description, stability_only in STEPS:
        if step_num not in step_numbers:
            results[step_num] = "skipped (--skip)"
            continue

        # Skip stability-only steps when running PCE
        if stability_only and target_mode == "PCE":
            results[step_num] = "skipped (PCE mode)"
            print(f"\n  Step {step_num} skipped — stability-only step "
                  f"(not applicable for --target PCE)")
            continue

        # Auto-skip if outputs already exist
        if not force and _step_is_complete(step_num, target_mode):
            results[step_num] = "skipped (outputs exist)"
            print(f"\n  Step {step_num} skipped — outputs already exist. "
                  f"Use --force to re-run.")
            continue

        _header(step_num, description)
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(module_name)
            mod.main()
            elapsed = time.perf_counter() - t0
            results[step_num] = f"OK  ({_fmt_elapsed(elapsed)})"
        except Exception:
            elapsed = time.perf_counter() - t0
            results[step_num] = f"FAILED  ({_fmt_elapsed(elapsed)})"
            print("\n[ERROR] Step failed with the following traceback:")
            traceback.print_exc()
            print("\nPipeline halted at step", step_num)
            print("Fix the error and re-run with --steps",
                  " ".join(str(s) for s in step_numbers if s >= step_num))
            _print_summary(results, total_start, target_mode)
            sys.exit(1)

    _print_summary(results, total_start, target_mode)


def _print_summary(results: dict[int, str], total_start: float,
                   target_mode: str) -> None:
    total_elapsed = time.perf_counter() - total_start
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  PIPELINE SUMMARY  (target: {target_mode})")
    print(bar)
    for step_num, module_name, description, _ in STEPS:
        status = results.get(step_num, "not reached")
        label  = description.split("—")[0].strip()
        print(f"  Step {step_num}  {label:<20s}  {status}")
    print(f"{bar}")
    print(f"  Total elapsed: {_fmt_elapsed(total_elapsed)}")
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    all_nums = [s[0] for s in STEPS]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--target", "-y",
        choices=["stability", "PCE"],
        default="stability",
        help="Regression target: 'stability' for ln(TS80m) (default), "
             "'PCE' for JV_default_PCE.",
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-run of all steps, ignoring existing outputs.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--steps",
        nargs="+",
        type=int,
        metavar="N",
        help="Run only these step numbers (e.g. --steps 4 5).",
    )
    group.add_argument(
        "--skip",
        nargs="+",
        type=int,
        metavar="N",
        help="Skip these step numbers (e.g. --skip 5 6).",
    )

    args = parser.parse_args()

    if args.steps is not None:
        invalid = [n for n in args.steps if n not in all_nums]
        if invalid:
            parser.error(f"Invalid step numbers: {invalid}. Valid: {all_nums}")
        args.run = sorted(set(args.steps))
    elif args.skip is not None:
        invalid = [n for n in args.skip if n not in all_nums]
        if invalid:
            parser.error(f"Invalid step numbers: {invalid}. Valid: {all_nums}")
        args.run = sorted(set(all_nums) - set(args.skip))
    else:
        args.run = all_nums

    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # Set global target mode BEFORE importing analysis modules
    pipeline_config.TARGET_MODE = args.target

    print(f"\nPerovskite Database Analysis Pipeline")
    print(f"  Target : {args.target} ({pipeline_config.target_column()})")
    print(f"  Steps  : {args.run}")
    if args.force:
        print(f"  Force  : ON (re-run all steps)")
    else:
        print(f"  Auto-skip : ON (existing outputs will be skipped)")

    run_steps(args.run, args.target, force=args.force)
