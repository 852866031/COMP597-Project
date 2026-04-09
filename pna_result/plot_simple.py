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


def _out_dir(steps_path: Path, batch_size: str = "") -> Path:
    """simple/plots/bs{N}/ — one subdirectory per batch size."""
    subdir = f"bs{batch_size}" if batch_size and batch_size != "?" else steps_path.stem
    d = steps_path.parent / "plots" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _epoch_starts(df: pd.DataFrame) -> List[int]:
    """Return step_idx values where the epoch number changes (skip first row)."""
    if "epoch" not in df.columns:
        return []
    changed = df["epoch"].diff().fillna(0) != 0
    changed.iloc[0] = False
    return df.loc[changed, "step_idx"].tolist()


def _draw_epoch_vlines(ax, epoch_step_ids: List[int]) -> None:
    first = True
    for step_id in epoch_step_ids:
        ax.axvline(x=step_id, color="#E63946", linestyle=":",
                   linewidth=1.2, alpha=0.85,
                   label="Epoch start" if first else None)
        first = False


# ---------------------------------------------------------------------------
# Batch-shape overlay (secondary y-axis)
# ---------------------------------------------------------------------------

BATCH_SHAPE_COLS = ["batch_num_graphs", "batch_num_nodes", "batch_num_edges"]
BATCH_SHAPE_COLORS = {"batch_num_graphs": "#FB8500",
                      "batch_num_nodes":  "#8338EC",
                      "batch_num_edges":  "#E63946"}
BATCH_SHAPE_LABELS = {"batch_num_graphs": "# graphs",
                      "batch_num_nodes":  "# nodes",
                      "batch_num_edges":  "# edges"}


def _has_batch_shape(df: pd.DataFrame) -> bool:
    """Return True if at least one batch-shape column has data."""
    return any(
        col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
        for col in BATCH_SHAPE_COLS
    )


def _overlay_batch_shape(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Overlay batch_num_graphs / num_nodes / num_edges on a secondary y-axis.

    Only columns that are present and non-empty are drawn.  A twin axis is
    created so the batch-shape scale does not interfere with the time scale.
    """
    steps = df["step_idx"].to_numpy()
    plotted = []
    for col in BATCH_SHAPE_COLS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(vals).all():
            continue
        plotted.append((col, vals))

    if not plotted:
        return

    ax2 = ax.twinx()
    for col, vals in plotted:
        ax2.plot(steps, vals,
                 color=BATCH_SHAPE_COLORS[col],
                 linewidth=1.2, alpha=0.75, linestyle="--",
                 label=BATCH_SHAPE_LABELS[col])

    ax2.set_ylabel("Batch shape", fontsize=9)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.legend(fontsize=8, loc="lower right")


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
# Plot 1b — total execution time + batch shape overlay (bs4096 only)
# ---------------------------------------------------------------------------

def plot_total_time_batch_shape(df: pd.DataFrame, out_path: Path,
                                meta: dict) -> None:
    """Total step time bar chart with batch-shape lines on a secondary axis.

    Only produced when batch-shape columns are present in *df*.
    """
    if not _has_batch_shape(df):
        print("  1b.[skip] batch shape columns not found")
        return

    steps   = df["step_idx"].to_numpy()
    step_ms = df["step_ms"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA — Total Step Execution Time + Batch Shape\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.bar(steps, step_ms, width=1.0, color=STEP_COLOR, alpha=0.80,
           label="Step time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Time (ms)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    _overlay_batch_shape(ax, df)

    # Merge legends from both axes
    handles, labels = ax.get_legend_handles_labels()
    ax2 = [c for c in fig.get_axes() if c is not ax]
    if ax2:
        h2, l2 = ax2[0].get_legend_handles_labels()
        handles += h2
        labels  += l2
    ax.legend(handles, labels, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  1b.total_time_batch  -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — execution time breakdown per step (stacked area)
# ---------------------------------------------------------------------------

def plot_breakdown(df: pd.DataFrame, out_path: Path, meta: dict,
                   epoch_step_ids: Optional[List[int]] = None) -> None:
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

    if epoch_step_ids:
        _draw_epoch_vlines(ax, epoch_step_ids)

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
    """Find all pna_simple_bs*_wk2.csv files across all batch sizes."""
    simple_dir = directory / "simple"
    if not simple_dir.is_dir():
        return []
    return sorted(
        p for p in simple_dir.glob("pna_simple_bs*_wk2.csv")
        if not p.name.endswith("_agg.csv")
    )


def _bs_representative_csvs(all_csvs: List[Path]) -> dict:
    """Return one CSV path per batch size (first wk match), keyed by int bs."""
    result: dict = {}
    for path in all_csvs:
        m = re.search(r"bs(\d+)", path.name)
        if m:
            bs = int(m.group(1))
            if bs not in result:
                result[bs] = path
    return result


# ---------------------------------------------------------------------------
# Batch-size comparison — mean epoch latency
# ---------------------------------------------------------------------------

def plot_epoch_latency_vs_bs(bs_csv_map: dict, out_path: Path) -> None:
    """Bar chart: mean epoch latency (s) vs batch size.

    Each bar represents the mean of per-epoch total step time
    (sum of step_ms for all steps within the epoch, averaged over epochs).
    """
    batch_sizes = sorted(bs_csv_map.keys())
    avgs_s: List[Optional[float]] = []

    for bs in batch_sizes:
        df = pd.read_csv(bs_csv_map[bs])
        df["step_ms"] = pd.to_numeric(df.get("step_ms", pd.Series(dtype=float)),
                                      errors="coerce")
        if "epoch" not in df.columns or df["step_ms"].isna().all():
            avgs_s.append(float("nan"))
            continue
        epoch_totals = df.groupby("epoch")["step_ms"].sum()
        avgs_s.append(float(epoch_totals.mean()) / 1000.0)  # ms → s

    colors = ["#4C72B0", "#DD8452", "#55A868", "#8338EC",
              "#E76F51", "#2A9D8F", "#E63946", "#888888"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "PNA — Mean Epoch Latency vs Batch Size\n"
        "sum of step_ms per epoch, averaged across epochs",
        fontsize=13, fontweight="bold",
    )

    x    = np.arange(len(batch_sizes))
    bars = ax.bar(x, avgs_s, width=0.55,
                  color=colors[:len(batch_sizes)], alpha=0.85, zorder=3)

    valid = [v for v in avgs_s if v is not None and not np.isnan(v)]
    y_max = max(valid) if valid else 1.0

    for bar, val in zip(bars, avgs_s):
        if val is None or np.isnan(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.01,
            f"{val:.1f}s",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"bs{bs}" for bs in batch_sizes], fontsize=10)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Mean Epoch Latency (s)")
    ax.set_ylim(0, y_max * 1.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  bs epoch latency  -> {out_path}")


def process_file(steps_path: Path) -> None:
    print(f"\nProcessing: {steps_path}")
    df = pd.read_csv(steps_path)
    required = {"step_idx", "step_ms", "forward_ms", "backward_ms", "optimizer_ms"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [skip] missing columns: {missing}")
        return

    meta = _parse_meta(steps_path.stem)
    out  = _out_dir(steps_path, meta["batch_size"])
    stem = steps_path.stem

    epoch_step_ids = _epoch_starts(df)

    # Agg CSV (optional — used by pancake plot)
    agg_path = _find_agg_csv(steps_path)
    agg_df   = pd.read_csv(agg_path) if agg_path else None
    if agg_df is None:
        print("  [info] no matching agg CSV — pancake will use step means")

    plot_total_time(df,                    out / f"{stem}_total_time.png", meta)
    if meta.get("batch_size") == "4096":
        plot_total_time_batch_shape(df,    out / f"{stem}_total_time_batch_shape.png", meta)
    plot_breakdown( df,                    out / f"{stem}_breakdown.png",  meta, epoch_step_ids)
    plot_loss(      df,                    out / f"{stem}_loss.png",       meta)
    plot_pancake(   df, agg_df,            out / f"{stem}_pancake.png",    meta)


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
                f"No simple/pna_simple_bs*_wk2.csv found in {script_dir / 'simple'}.\n"
                "Run start-pna-simple.sh first."
            )
            sys.exit(0)
        print(f"Auto-discovered {len(csv_files)} steps CSV(s)")

    for path in csv_files:
        if not path.exists():
            print(f"[warn] file not found, skipping: {path}")
            continue
        process_file(path)

    # Batch-size comparison plot (requires ≥2 batch sizes)
    bs_map = _bs_representative_csvs(csv_files)
    if len(bs_map) >= 2:
        print("\nGenerating batch-size comparison:")
        plot_epoch_latency_vs_bs(bs_map, script_dir / "plots" / "bs_epoch_latency.png")
    else:
        print("\n[info] Only one batch size found; skipping epoch-latency comparison.")

    print("\nDone.")


if __name__ == "__main__":
    main()
