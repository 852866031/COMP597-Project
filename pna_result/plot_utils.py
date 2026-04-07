#!/usr/bin/env python3
"""
plot_utils.py — visualise outputs produced by PNAUtilsStats.

Usage
-----
# Auto-discover utils/pna_utils_bs*_steps.csv in the same directory:
    python pna_result/plot_utils.py

# Point at a specific steps CSV:
    python pna_result/plot_utils.py --steps pna_result/utils/pna_utils_bs256_steps.csv

Output
------
Plots are saved to  pna_result/utils/plots/

Up to six plots are produced per CSV:
  1. total_time.png    — total step execution time per step (bar chart)
  2. breakdown.png     — stacked area execution time breakdown per step
  3. pancake.png       — donut chart of mean time share per substep
  4. util_gpu.png      — GPU utilisation (%) sampled over steps
  5. util_cpu.png      — CPU utilisation (%) sampled over steps
  6. util_ram.png      — RAM used (GB) sampled over steps

Plots 4-6 are skipped when the respective column has no sampled data.
Epoch boundaries are annotated with red dashed vertical lines on util plots.
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

EPOCH_LINE_COLOR = "#E63946"   # red for epoch boundary markers

# Colours for sampled_phase dots on util plots
PHASE_COLORS = {
    "forward":  "#4C72B0",   # blue   — sampled during forward pass
    "backward": "#DD8452",   # orange — sampled during backward pass
    "":         "#888888",   # grey   — no phase info
}


def _parse_meta(stem: str) -> dict:
    meta = {"batch_size": "?"}
    m = re.search(r"bs(\d+)", stem)
    if m:
        meta["batch_size"] = m.group(1)
    return meta


def _subtitle(meta: dict) -> str:
    return f"batch size {meta['batch_size']}"


def _find_agg_csv(steps_path: Path) -> Optional[Path]:
    """pna_utils_bs256_steps.csv  →  pna_utils_bs256_steps_agg.csv"""
    candidate = steps_path.parent / (steps_path.stem + "_agg.csv")
    return candidate if candidate.exists() else None


def _out_dir(steps_path: Path) -> Path:
    """utils/plots/ (file is already inside the utils/ subdir)"""
    d = steps_path.parent / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _epoch_starts(df: pd.DataFrame) -> List[int]:
    """Return step_idx values that begin a new epoch (all except the first).

    Requires an ``epoch`` column; returns an empty list if absent.
    """
    if "epoch" not in df.columns:
        return []
    changed = df["epoch"].diff().fillna(0) != 0
    changed.iloc[0] = False
    return df.loc[changed, "step_idx"].tolist()


# ---------------------------------------------------------------------------
# Plot 1 — total execution time per step (bar chart)
# ---------------------------------------------------------------------------

def plot_total_time(df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    steps   = df["step_idx"].to_numpy()
    step_ms = df["step_ms"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Utils — Total Step Execution Time\n{_subtitle(meta)}",
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
    print(f"  1. total_time  -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — execution time breakdown per step (stacked area)
# ---------------------------------------------------------------------------

def plot_breakdown(df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    steps   = df["step_idx"].to_numpy()
    bottoms = np.zeros(len(df))

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Utils — Execution Time Breakdown per Step\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    for col, label, color in zip(SUBSTEP_COLS, SUBSTEP_LABELS, SUBSTEP_COLORS):
        vals = df[col].to_numpy()
        ax.fill_between(steps, bottoms, bottoms + vals,
                        alpha=0.80, color=color, label=label)
        bottoms += vals

    ax.plot(steps, df["step_ms"].to_numpy(), color=STEP_COLOR,
            linewidth=0.7, alpha=0.45, label="Total step (raw)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Time (ms)")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  2. breakdown   -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — aggregate step time pancake (donut) chart
# ---------------------------------------------------------------------------

def plot_pancake(df: pd.DataFrame, agg_df: Optional[pd.DataFrame],
                 out_path: Path, meta: dict) -> None:
    if agg_df is not None:
        row = agg_df.set_index("metric")["mean"]
        try:
            fwd_mean  = float(row["forward_ms"])
            bwd_mean  = float(row["backward_ms"])
            opt_mean  = float(row["optimizer_ms"])
            step_mean = float(row["step_ms"])
        except KeyError:
            agg_df = None

    if agg_df is None:
        fwd_mean  = float(df["forward_ms"].mean())
        bwd_mean  = float(df["backward_ms"].mean())
        opt_mean  = float(df["optimizer_ms"].mean())
        step_mean = float(df["step_ms"].mean())

    other_mean = max(0.0, step_mean - fwd_mean - bwd_mean - opt_mean)

    values = [fwd_mean, bwd_mean, opt_mean, other_mean]
    labels = SUBSTEP_LABELS + ["Other"]
    colors = SUBSTEP_COLORS + ["#d3d3d3"]

    nonzero = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
    values, labels, colors = zip(*nonzero)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(
        f"PNA Utils — Average Step Time Breakdown\n{_subtitle(meta)}",
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

    ax.text(0, 0, f"mean step\n{step_mean:.1f} ms",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#333333")

    legend_labels = [f"{l}  ({v:.1f} ms)" for l, v in zip(labels, values)]
    ax.legend(wedges, legend_labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  3. pancake     -> {out_path}")


# ---------------------------------------------------------------------------
# Plots 4-6 — one hardware utilisation plot each
# ---------------------------------------------------------------------------

def _draw_epoch_vlines(ax, epoch_step_ids: List[int]) -> None:
    """Draw red dashed vertical lines at the start of each new epoch."""
    first = True
    for step_id in epoch_step_ids:
        ax.axvline(x=step_id, color=EPOCH_LINE_COLOR,
                   linestyle=":", linewidth=1.2, alpha=0.85,
                   label="Epoch start" if first else None)
        first = False


def plot_util_gpu(df: pd.DataFrame, out_path: Path, meta: dict,
                  epoch_step_ids: List[int]) -> None:
    """GPU utilisation plot — forward and backward as independent series."""
    fwd_col = "fwd_gpu_util"
    bwd_col = "bwd_gpu_util"

    fwd_series = pd.to_numeric(df[fwd_col], errors="coerce") if fwd_col in df.columns else pd.Series(dtype=float)
    bwd_series = pd.to_numeric(df[bwd_col], errors="coerce") if bwd_col in df.columns else pd.Series(dtype=float)

    fwd_mask = fwd_series.notna()
    bwd_mask = bwd_series.notna()

    if fwd_mask.sum() == 0 and bwd_mask.sum() == 0:
        print(f"  4. util_gpu            -> [skip] no sampled data in fwd_gpu_util / bwd_gpu_util")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Utils — GPU Utilisation\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    for series, mask, col_key, label in [
        (fwd_series, fwd_mask, "forward",  "Forward"),
        (bwd_series, bwd_mask, "backward", "Backward"),
    ]:
        if mask.sum() == 0:
            continue
        color = PHASE_COLORS[col_key]
        steps = df.loc[mask, "step_idx"].to_numpy()
        vals  = series[mask].to_numpy()
        ax.plot(steps, vals, color=color, linewidth=0.7, alpha=0.45, zorder=1)
        ax.scatter(steps, vals, color=color, s=22, zorder=3, label=label)
        avg = float(vals.mean())
        ax.axhline(y=avg, color=color, linestyle="--", linewidth=1.4,
                   alpha=0.85, zorder=2, label=f"{label} avg ({avg:.1f}%)")

    _draw_epoch_vlines(ax, epoch_step_ids)

    ax.set_xlabel("Step")
    ax.set_ylabel("GPU Utilisation (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  4. util_gpu            -> {out_path}")


def plot_util_cpu(df: pd.DataFrame, out_path: Path, meta: dict,
                  epoch_step_ids: List[int]) -> None:
    """CPU utilisation plot — combines forward and backward samples."""
    # Merge fwd and bwd CPU samples: prefer fwd when both exist for a step,
    # fall back to bwd, giving a single value per sampled step.
    fwd = pd.to_numeric(df.get("fwd_cpu_util", pd.Series(dtype=float)), errors="coerce")
    bwd = pd.to_numeric(df.get("bwd_cpu_util", pd.Series(dtype=float)), errors="coerce")
    series = fwd.combine_first(bwd)
    mask   = series.notna()
    if mask.sum() == 0:
        print(f"  5. util_cpu            -> [skip] no sampled data for cpu_util")
        return

    sampled_steps = df.loc[mask, "step_idx"].to_numpy()
    sampled_vals  = series[mask].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Utils — CPU Utilisation\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.plot(sampled_steps, sampled_vals, color="#457B9D", linewidth=0.8, alpha=0.45)
    ax.scatter(sampled_steps, sampled_vals, color="#457B9D", s=20, zorder=3)

    _draw_epoch_vlines(ax, epoch_step_ids)
    if epoch_step_ids:
        ax.legend(fontsize=9, loc="upper right")

    ax.set_xlabel("Step")
    ax.set_ylabel("CPU Utilisation (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  5. util_cpu            -> {out_path}")


def plot_util_ram(df: pd.DataFrame, out_path: Path, meta: dict,
                  epoch_step_ids: List[int]) -> None:
    """RAM utilisation plot — y-axis in GB, starting from 0."""
    fwd = pd.to_numeric(df.get("fwd_ram_used_mb", pd.Series(dtype=float)), errors="coerce")
    bwd = pd.to_numeric(df.get("bwd_ram_used_mb", pd.Series(dtype=float)), errors="coerce")
    series = fwd.combine_first(bwd)
    mask   = series.notna()
    if mask.sum() == 0:
        print(f"  6. util_ram            -> [skip] no sampled data for ram_used_mb")
        return

    sampled_steps = df.loc[mask, "step_idx"].to_numpy()
    sampled_gb    = series[mask].to_numpy() / 1024.0   # MiB → GB

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Utils — RAM Utilisation\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.plot(sampled_steps, sampled_gb, color="#2A9D8F", linewidth=0.8, alpha=0.45)
    ax.scatter(sampled_steps, sampled_gb, color="#2A9D8F", s=20, zorder=3)

    _draw_epoch_vlines(ax, epoch_step_ids)
    if epoch_step_ids:
        ax.legend(fontsize=9, loc="upper right")

    y_upper = sampled_gb.max() * 1.15   # 15% headroom above peak

    ax.set_xlabel("Step")
    ax.set_ylabel("RAM Used (GB)")
    ax.set_ylim(0, y_upper)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  6. util_ram            -> {out_path}")


# ---------------------------------------------------------------------------
# Discovery + orchestration
# ---------------------------------------------------------------------------

def discover_steps_csvs(directory: Path) -> List[Path]:
    """Find pna_utils step CSV files in utils/ subdir."""
    utils_dir = directory / "utils"
    if not utils_dir.is_dir():
        return []
    return sorted(utils_dir.glob("pna_utils_bs*_steps.csv"))


def process_file(steps_path: Path) -> None:
    print(f"\nProcessing: {steps_path.name}")

    df = pd.read_csv(steps_path)
    required = {"step_idx", "step_ms", "forward_ms", "backward_ms", "optimizer_ms"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [skip] missing columns: {missing}")
        return

    meta = _parse_meta(steps_path.stem)
    out  = _out_dir(steps_path)
    stem = steps_path.stem

    agg_path = _find_agg_csv(steps_path)
    agg_df   = pd.read_csv(agg_path) if agg_path else None
    if agg_df is None:
        print("  [info] no matching agg CSV — pancake will use step means")

    epoch_starts = _epoch_starts(df)

    plot_total_time(df,           out / f"{stem}_total_time.png",  meta)
    plot_breakdown( df,           out / f"{stem}_breakdown.png",   meta)
    plot_pancake(   df, agg_df,   out / f"{stem}_pancake.png",     meta)
    plot_util_gpu(  df,           out / f"{stem}_util_gpu.png",    meta, epoch_starts)
    plot_util_cpu(  df,           out / f"{stem}_util_cpu.png",    meta, epoch_starts)
    plot_util_ram(  df,           out / f"{stem}_util_ram.png",    meta, epoch_starts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PNAUtilsStats CSV output."
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        metavar="CSV",
        help=(
            "Path(s) to pna_utils_bs*_steps.csv file(s). "
            "If omitted, auto-discovers all matching files in utils/."
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
                f"No pna_utils_bs*_steps.csv files found in {script_dir / 'utils'}.\n"
                "Run start-pna-utils.sh first."
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
