#!/usr/bin/env python3
"""
start_all_pna_exp.py — orchestrate all PNA training experiments.

Each (experiment, batch_size) pair is run only when its expected output CSV is
absent.  Use --force to unconditionally rerun for one or more batch sizes.

Experiments
-----------
  simple  : bs 512, 1024, 2048, 4096
  spike   : bs 512, 4096
  utils   : bs 512, 1024, 2048, 4096
  carbon  : bs 512, 1024, 2048, 4096

Epoch scaling (mirrors the shell scripts)
-----------------------------------------
  bs=512  → 15 epochs (reference)
  Others  → round(15 × bs / 512), floor 1, cap 30

Output-file check
-----------------
  Before launching a run the script globs for the expected output CSV in the
  pna_result/ subdirectory.  Any worker-count suffix is accepted (wk0, wk2 …).

Usage
-----
  # Run everything not yet done:
  python scripts/start_all_pna_exp.py

  # Force bs=512 to rerun across all experiments:
  python scripts/start_all_pna_exp.py --force 512

  # Force multiple batch sizes:
  python scripts/start_all_pna_exp.py --force 512 1024
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, List

# ---------------------------------------------------------------------------
# Paths (this file lives in scripts/; repo root is one level up)
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.resolve()
REPO_DIR    = SCRIPTS_DIR.parent.resolve()
RESULT_DIR  = REPO_DIR / "pna_result"


# ---------------------------------------------------------------------------
# Epoch formula — matches (15 * BS + 256) / 512 with floor 1, cap 30
# ---------------------------------------------------------------------------

def _epochs(bs: int) -> int:
    """Return the number of training epochs for the given batch size."""
    e = (15 * bs + 256) // 512
    return max(1, min(30, e))


# ---------------------------------------------------------------------------
# Output-file existence helpers
# ---------------------------------------------------------------------------

def _glob_exists(subdir: Path, pattern: str, exclude_suffix: str = "") -> bool:
    """Return True if subdir contains at least one file matching pattern."""
    if not subdir.is_dir():
        return False
    matches = list(subdir.glob(pattern))
    if exclude_suffix:
        matches = [p for p in matches if not p.name.endswith(exclude_suffix)]
    return bool(matches)


def _simple_done(bs: int) -> bool:
    # pna_simple_bs{N}_wk*.csv — exclude the _agg.csv companion files
    return _glob_exists(
        RESULT_DIR / "simple",
        f"pna_simple_bs{bs}_wk*.csv",
        exclude_suffix="_agg.csv",
    )


def _spike_done(bs: int) -> bool:
    return _glob_exists(
        RESULT_DIR / "spike",
        f"pna_spike_bs{bs}_wk*_gc_on.csv",
    )


def _utils_done(bs: int) -> bool:
    return _glob_exists(
        RESULT_DIR / "utils",
        f"pna_utils_bs{bs}_wk*_steps.csv",
    )


def _carbon_done(bs: int) -> bool:
    return _glob_exists(
        RESULT_DIR / "carbon",
        f"pna_carbon_bs{bs}_wk*_steps.csv",
    )


# ---------------------------------------------------------------------------
# Experiment catalogue
# ---------------------------------------------------------------------------

EXPERIMENTS: List[dict] = [
    {
        "name":        "simple",
        "script":      "start-pna-simple.sh",
        "batch_sizes": [512, 1024, 2048, 4096],
        "done_fn":     _simple_done,
    },
    {
        "name":        "spike",
        "script":      "start-pna-spike.sh",
        "batch_sizes": [512, 4096],
        "done_fn":     _spike_done,
    },
    {
        "name":        "utils",
        "script":      "start-pna-utils.sh",
        "batch_sizes": [512, 1024, 2048, 4096],
        "done_fn":     _utils_done,
    },
    {
        "name":        "carbon",
        "script":      "start-pna-carbon.sh",
        "batch_sizes": [512, 1024, 2048, 4096],
        "done_fn":     _carbon_done,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run(script: str, bs: int) -> None:
    cmd = ["bash", str(SCRIPTS_DIR / script), "-bs", str(bs)]
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"    [warn] {script} -bs {bs} exited with code {result.returncode}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all PNA experiments, skipping any whose output CSV already exists.\n"
            "Use --force BS [BS ...] to rerun specific batch sizes unconditionally."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        metavar="BS",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Batch size(s) to rerun unconditionally, e.g. --force 512 or "
            "--force 512 1024.  Affects all experiments that include these sizes."
        ),
    )
    args = parser.parse_args()

    forced: set[int] = set(args.force)
    if forced:
        print(f"Forced batch sizes: {sorted(forced)}\n")

    for exp in EXPERIMENTS:
        print(f"{'=' * 52}")
        print(f"  {exp['name'].upper()}  —  bs {exp['batch_sizes']}")
        print(f"{'=' * 52}")
        for bs in exp["batch_sizes"]:
            epochs = _epochs(bs)
            if bs not in forced and exp["done_fn"](bs):
                print(f"  [skip] bs={bs:>4}  ({epochs:>2} epochs) — output already exists")
                continue
            reason = "forced" if bs in forced else "output missing"
            print(f"  [run ] bs={bs:>4}  ({epochs:>2} epochs) — {reason}")
            _run(exp["script"], bs)
        print()

    print("All experiments complete.")


if __name__ == "__main__":
    main()
