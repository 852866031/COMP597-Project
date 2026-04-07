#!/usr/bin/env python3
"""
plot_simple.py — visualise outputs produced by PNATrainerSimpleStats.

Usage
-----
# Auto-discover simple/result.csv relative to this script's directory:
    python pna_result/plot_simple.py

# Point at a specific steps CSV:
    python pna_result/plot_simple.py --steps pna_result/simple/result.csv

Output
------
Plots are saved to  pna_result/simple/plots/

Four plots are produced:
  1. total_time.png  — total step execution time per step (bar chart)
  2. breakdown.png   — stacked area execution time breakdown per step
  3. loss.png        — loss per step
  4. pancake.png     — donut chart of mean time share per substep
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

SUBSTEP_COLS   = ["forward_ms", "backward_ms", "optimizer_ms"]
SUBSTEP_LABELS = ["Forward", "Backward", "Optimizer"]
SUBSTEP_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
STEP_COLOR     = "#888888"


def _parse_meta(stem: str) -> dict:
    meta = {"batch_size": "?"}
    m = re.search(r"bs(\d+)", stem)
    if m:
        meta["batch_size"] = m.group(1)
    return meta


def _subtitle(meta: dict) -> str:
    return f"batch size {meta['batch_size']}"


def _find_agg_csv(steps_path: Path) -> Optional[Path]:
    """simple/pna_simple_bs<N>.csv  ->  simple/pna_simple_bs<N>_agg.csv"""
    candidate = steps_path.parent / (steps_path.stem + "_agg.csv")
    return candidate if candidate.exists() else None


def _out_dir(steps_path: Path) -> Path:
    """simple/plots/ (file is already inside the simple/ subdir)"""
    d = steps_path.parent / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Plot 1 — total execution time per step
# ---------------------------------------------------------------------------

def plot_total_time(df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    steps   = df["step_idx"].to_numpy()
    step_ms = df["step_ms"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA — Total Step Execution Time\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.bar(steps, step_ms, width=1.0, color=STEP_COLOR, alpha=0.80)

    ax.set_xlabel("Step")
    ax.set_ylabel("Time (ms)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  1. total_time        -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — execution time breakdown per step (stacked area)
# ---------------------------------------------------------------------------

def plot_breakdown(df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    steps   = df["step_idx"].to_numpy()
    bottoms = np.zeros(len(df))

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA — Execution Time Breakdown per Step\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    for col, label, color in zip(SUBSTEP_COLS, SUBSTEP_LABELS, SUBSTEP_COLORS):
        vals = df[col].to_numpy()
        ax.fill_between(steps, bottoms, bottoms + vals,
                        alpha=0.80, color=color, label=label)
        bottoms += vals

    # Overlay raw total step time so spikes that exceed the stack are visible
    ax.plot(steps, df["step_ms"].to_numpy(), color=STEP_COLOR,
            linewidth=0.7, alpha=0.45, label="Total step (raw)")

    ax.legend(fontsize=9, loc="upper right")

    ax.set_xlabel("Step")
    ax.set_ylabel("Time (ms)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  2. breakdown         -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — loss per step
# ---------------------------------------------------------------------------

def plot_loss(df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    loss = df["loss"].dropna()
    if loss.empty:
        print("  3. [skip] no loss values found")
        return

    steps = df.loc[loss.index, "step_idx"]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA — Loss per Step\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.plot(steps, loss, color="#4C72B0", linewidth=0.7, alpha=0.7)

    ax.set_xlabel("Step")
    ax.set_ylabel("L1 Loss")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  3. loss              -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — aggregate step time pancake (donut) chart
# ---------------------------------------------------------------------------

def plot_pancake(df: pd.DataFrame, agg_df: Optional[pd.DataFrame],
                 out_path: Path, meta: dict) -> None:
    """Donut chart showing the average proportion of step time per substep.

    Tries to pull mean values from the agg CSV first; falls back to computing
    them from the steps data directly so the plot works even without an agg CSV.
    """
    # Resolve mean values
    if agg_df is not None:
        row = agg_df.set_index("metric")["mean"]
        try:
            fwd_mean  = float(row["forward_ms"])
            bwd_mean  = float(row["backward_ms"])
            opt_mean  = float(row["optimizer_ms"])
            step_mean = float(row["step_ms"])
        except KeyError:
            agg_df = None  # fall through to manual calculation

    if agg_df is None:
        fwd_mean  = float(df["forward_ms"].mean())
        bwd_mean  = float(df["backward_ms"].mean())
        opt_mean  = float(df["optimizer_ms"].mean())
        step_mean = float(df["step_ms"].mean())

    # Other time (overhead between substeps not captured individually)
    other_mean = max(0.0, step_mean - fwd_mean - bwd_mean - opt_mean)

    values = [fwd_mean, bwd_mean, opt_mean, other_mean]
    labels = SUBSTEP_LABELS + ["Other"]
    colors = SUBSTEP_COLORS + ["#d3d3d3"]

    # Drop zero slices
    nonzero = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
    values, labels, colors = zip(*nonzero)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(
        f"PNA — Average Step Time Breakdown\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5),
    )

    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    # Centre annotation: mean total step time
    ax.text(0, 0, f"mean step\n{step_mean:.1f} ms",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#333333")

    # Legend with absolute ms values
    legend_labels = [f"{l}  ({v:.1f} ms)" for l, v in zip(labels, values)]
    ax.legend(wedges, legend_labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9,
              frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  4. pancake           -> {out_path}")


# ---------------------------------------------------------------------------
# Discovery + orchestration
# ---------------------------------------------------------------------------

def discover_steps_csvs(directory: Path) -> List[Path]:
    """Look for simple/pna_simple_bs512_wk*.csv relative to directory."""
    simple_dir = directory / "simple"
    if not simple_dir.is_dir():
        return []
    return sorted(
        p for p in simple_dir.glob("pna_simple_bs512_wk*.csv")
        if not p.name.endswith("_agg.csv")
    )


def process_file(steps_path: Path) -> None:
    print(f"\nProcessing: {steps_path}")
    df = pd.read_csv(steps_path)
    required = {"step_idx", "step_ms", "forward_ms", "backward_ms", "optimizer_ms"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [skip] missing columns: {missing}")
        return

    meta = _parse_meta(steps_path.stem)
    out  = _out_dir(steps_path)
    stem = steps_path.stem

    # Agg CSV (optional — used by pancake plot)
    agg_path = _find_agg_csv(steps_path)
    agg_df   = pd.read_csv(agg_path) if agg_path else None
    if agg_df is None:
        print("  [info] no matching agg CSV — pancake will use step means")

    plot_total_time(df,         out / f"{stem}_total_time.png", meta)
    plot_breakdown( df,         out / f"{stem}_breakdown.png",  meta)
    plot_loss(      df,         out / f"{stem}_loss.png",       meta)
    plot_pancake(   df, agg_df, out / f"{stem}_pancake.png",    meta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PNATrainerSimpleStats CSV output."
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        metavar="CSV",
        help=(
            "Path(s) to simple/result.csv file(s). "
            "If omitted, auto-discovers simple/result.csv in the script's directory."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    if args.steps:
        csv_files = [Path(p).resolve() for p in args.steps]
    else:
        csv_files = discover_steps_csvs(script_dir)
        if not csv_files:
            print(
                f"No simple/pna_simple_bs512_wk*.csv found in {script_dir / 'simple'}.\n"
                "Run with --steps /path/to/pna_simple_bs512_wk<N>.csv to specify files explicitly."
            )
            sys.exit(0)
        print(f"Auto-discovered {len(csv_files)} steps CSV(s)")

    for path in csv_files:
        if not path.exists():
            print(f"[warn] file not found, skipping: {path}")
            continue
        process_file(path)

    print("\nDone.")


if __name__ == "__main__":
    main()
