#!/usr/bin/env python3
"""
plot_raw.py — plot raw (minimal) per-step timing from PNABaseStats.

One plot per batch size: total step execution time bar chart.

Output
------
  pna_result/base/plots/pna_base_bs{N}_wk2_total_time.png

Usage
-----
    python pna_result/plot_raw.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEP_COLOR = "#888888"


def _parse_bs(stem: str) -> str:
    m = re.search(r"bs(\d+)", stem)
    return m.group(1) if m else "?"


def discover(directory: Path) -> List[Path]:
    d = directory / "base"
    if not d.is_dir():
        return []
    return sorted(d.glob("pna_base_bs*_wk2.csv"))


def plot_total_time(df: pd.DataFrame, out_path: Path, batch_size: str) -> None:
    steps   = df["step_idx"].to_numpy()
    step_ms = df["step_ms"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.suptitle(
        f"Raw: Per Step Execution Time (batch size {batch_size})",
        fontsize=15, fontweight="bold",
    )

    ax.bar(steps, step_ms, width=1.0, color=STEP_COLOR, alpha=0.80)

    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel("Time (ms)", fontsize=14)

    # Fewer, larger x-ticks
    x_max = int(steps[-1]) if len(steps) else 100
    x_step = max(1, round(x_max / 6 / 10) * 10)  # ~6 ticks, rounded to nearest 10
    ax.set_xticks(np.arange(0, x_max + 1, x_step))
    ax.tick_params(axis="both", labelsize=13)

    # Fewer y-ticks
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main() -> None:
    script_dir = Path(__file__).parent.resolve()
    csvs = discover(script_dir)

    if not csvs:
        print(f"No pna_base_bs*_wk2.csv found in {script_dir / 'base'}.\n"
              "Run start-pna-base.sh first.")
        sys.exit(0)

    print(f"Auto-discovered {len(csvs)} base CSV(s)")

    out_dir = script_dir / "base" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in csvs:
        bs = _parse_bs(path.stem)
        print(f"\nProcessing: {path.name}  (bs={bs})")

        df = pd.read_csv(path)
        if "step_ms" not in df.columns or "step_idx" not in df.columns:
            print("  [skip] missing columns")
            continue

        plot_total_time(df, out_dir / f"{path.stem}_total_time.png", bs)

    print("\nDone.")


if __name__ == "__main__":
    main()
