#!/usr/bin/env python3
"""
plot_batchsize.py — compare PNA metrics across batch sizes 64, 256, and 1024.

All three batch sizes must be present; the script exits with an error listing
every missing file rather than silently skipping.

Sources required (9 files total — 3 per batch size):
  pna_result/utils/pna_utils_bs{N}_steps.csv
      Step-level hardware utilisation samples from PNAUtilsStats.

  pna_result/carbon/pna_carbon_bs{N}_step-steps.csv
      Per-step energy data from PNACarbonStats (CodeCarbon task rows).

  pna_result/carbon/pna_carbon_bs{N}_substep-substeps.csv
      Per-substep energy data from PNACarbonStats (forward / backward /
      optimizer).  Only sampled steps appear here (interval-gate policy);
      the per-epoch sums therefore reflect measured samples rather than
      the full epoch.

Output — pna_result/plots/
------
  1. bs_gpu_util.png        — avg GPU utilisation (%) across all epochs, per batch size
  2. bs_cpu_util.png        — avg CPU utilisation (%) across all epochs, per batch size (per-process)
  3. bs_energy_total.png    — mean per-epoch total energy (mWh) averaged across all epochs, per batch size
  4. bs_energy_substep.png  — clustered bar: x = substep (fwd/bwd/opt), clusters = batch sizes, mean per-epoch energy (mWh)

Usage
-----
    python pna_result/plot_batchsize.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZES: List[int] = [64, 256, 512, 1024]
BS_COLORS:   List[str] = ["#4C72B0", "#DD8452", "#55A868", "#8338EC"]  # blue / orange / green / purple
BS_LABELS:   List[str] = [f"bs{n}" for n in BATCH_SIZES]

KWH_TO_MWH = 1_000.0     # kWh → mWh


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def _utils_csv(result_dir: Path, bs: int) -> Path:
    return result_dir / "utils" / f"pna_utils_bs{bs}_steps.csv"


def _step_csv(result_dir: Path, bs: int) -> Path:
    return result_dir / "carbon" / f"pna_carbon_bs{bs}_step-steps.csv"


def _substep_csv(result_dir: Path, bs: int) -> Path:
    return result_dir / "carbon" / f"pna_carbon_bs{bs}_substep-substeps.csv"


# ---------------------------------------------------------------------------
# File existence check — error on any missing file
# ---------------------------------------------------------------------------

def check_files(result_dir: Path) -> None:
    missing: List[Path] = []
    for bs in BATCH_SIZES:
        for path in [_utils_csv(result_dir, bs),
                     _step_csv(result_dir, bs),
                     _substep_csv(result_dir, bs)]:
            if not path.exists():
                missing.append(path)
    if missing:
        print("ERROR: the following required CSV files are missing.")
        print("Run the start-pna-*.sh scripts with -bs 64, -bs 256, and -bs 1024 first.\n")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CSV loading & enrichment
# ---------------------------------------------------------------------------

def _load_utils(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_step_task(task_name: str):
    m = re.match(r"e(\d+)_step_(\d+)", str(task_name))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _parse_substep_task(task_name: str):
    m = re.match(r"e(\d+)_(fwd|bwd|opt)_(\d+)", str(task_name))
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else (None, None, None)


def _enrich_step_df(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["task_name"].apply(
        lambda t: pd.Series(_parse_step_task(t), index=["epoch", "step_idx"])
    )
    df = df.copy()
    df["epoch"]    = parsed["epoch"].astype("Int64")
    df["step_idx"] = parsed["step_idx"].astype("Int64")
    return df.dropna(subset=["epoch", "step_idx"]).reset_index(drop=True)


def _enrich_substep_df(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["task_name"].apply(
        lambda t: pd.Series(_parse_substep_task(t), index=["epoch", "phase", "step_idx"])
    )
    df = df.copy()
    df["epoch"]    = parsed["epoch"].astype("Int64")
    df["phase"]    = parsed["phase"]
    df["step_idx"] = parsed["step_idx"].astype("Int64")
    return df.dropna(subset=["epoch", "phase", "step_idx"]).reset_index(drop=True)


def load_all(result_dir: Path):
    """Return (utils_data, step_data, substep_data) dicts keyed by batch size."""
    utils_data:   Dict[int, pd.DataFrame] = {}
    step_data:    Dict[int, pd.DataFrame] = {}
    substep_data: Dict[int, pd.DataFrame] = {}

    for bs in BATCH_SIZES:
        utils_data[bs]   = _load_utils(_utils_csv(result_dir, bs))
        step_data[bs]    = _enrich_step_df(pd.read_csv(_step_csv(result_dir, bs)))
        substep_data[bs] = _enrich_substep_df(pd.read_csv(_substep_csv(result_dir, bs)))

    return utils_data, step_data, substep_data


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_path: Path, label: str) -> None:
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


def _bar_value_labels(ax, bars, fmt="{:.1f}") -> None:
    """Print value above each bar."""
    valid_heights = [b.get_height() for b in bars if not np.isnan(b.get_height())]
    if not valid_heights:
        return
    y_max = max(valid_heights)
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_max * 0.01,
            fmt.format(h),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )


def _simple_bars(ax, totals: List[float], ylabel: str, fmt: str = "{:.3f}",
                 epoch_counts: Optional[List[int]] = None) -> None:
    """Draw a simple 3-bar chart (one bar per batch size) and annotate values.

    If epoch_counts is provided, each bar is labelled with its value and the
    number of epochs that were aggregated, e.g. "0.123 mWh\n(20 epochs)".
    """
    x    = np.arange(len(BATCH_SIZES))
    bars = ax.bar(x, totals, width=0.5, color=BS_COLORS, alpha=0.85, zorder=3)

    valid = [h for h in totals if not np.isnan(h)]
    y_max = max(valid) if valid else 1.0

    for i, (bar, val) in enumerate(zip(bars, totals)):
        if np.isnan(val):
            continue
        top = bar.get_height() + y_max * 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top,
            fmt.format(val),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        if epoch_counts is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top + y_max * 0.055,
                f"({epoch_counts[i]} epochs)",
                ha="center", va="bottom", fontsize=8, color="#555555",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(BS_LABELS, fontsize=10)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, y_max * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Plot 1 — GPU utilisation average
# ---------------------------------------------------------------------------

def plot_gpu_util_avg(utils_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    avgs: List[float] = []
    for bs in BATCH_SIZES:
        df  = utils_data[bs]
        fwd = pd.to_numeric(df.get("fwd_gpu_util", pd.Series(dtype=float)), errors="coerce")
        bwd = pd.to_numeric(df.get("bwd_gpu_util", pd.Series(dtype=float)), errors="coerce")
        combined = pd.concat([fwd, bwd]).dropna()
        avgs.append(float(combined.mean()) if not combined.empty else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "PNA — GPU Utilisation vs Batch Size\n"
        "mean of all sampled forward + backward steps (all epochs)",
        fontsize=13, fontweight="bold",
    )

    _simple_bars(ax, avgs, "GPU Utilisation (%)", fmt="{:.1f}%")
    ax.set_ylim(0, 100)
    _save(fig, out_path, "bs_gpu_util")


# ---------------------------------------------------------------------------
# Plot 2 — CPU utilisation average (per-process)
# ---------------------------------------------------------------------------

def plot_cpu_util_avg(utils_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    avgs: List[float] = []
    for bs in BATCH_SIZES:
        df  = utils_data[bs]
        fwd = pd.to_numeric(df.get("fwd_cpu_util", pd.Series(dtype=float)), errors="coerce")
        bwd = pd.to_numeric(df.get("bwd_cpu_util", pd.Series(dtype=float)), errors="coerce")
        combined = pd.concat([fwd, bwd]).dropna()
        avgs.append(float(combined.mean()) if not combined.empty else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "PNA — CPU Utilisation vs Batch Size (per-process)\n"
        "mean of all sampled forward + backward steps (all epochs)",
        fontsize=13, fontweight="bold",
    )

    _simple_bars(ax, avgs, "CPU Utilisation (%)", fmt="{:.1f}%")
    ax.set_ylim(0, 100)
    _save(fig, out_path, "bs_cpu_util")


# ---------------------------------------------------------------------------
# Plot 3 — total energy aggregated across all epochs
# ---------------------------------------------------------------------------

def plot_energy_total(step_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    avgs: List[float] = []
    for bs in BATCH_SIZES:
        df = step_data[bs]
        avgs.append(float(df.groupby("epoch")["energy_consumed"].sum().mean()) * KWH_TO_MWH)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "PNA — Total Sampled Energy vs Batch Size\n"
        "mean per epoch · CodeCarbon (mWh)",
        fontsize=13, fontweight="bold",
    )

    _simple_bars(ax, avgs, "Energy (mWh)", fmt="{:.3f}")
    _save(fig, out_path, "bs_energy_total")


# ---------------------------------------------------------------------------
# Plots 4–6 — per-phase energy aggregated across all epochs
# ---------------------------------------------------------------------------

_PHASES       = ["fwd", "bwd", "opt"]
_PHASE_LABELS = {"fwd": "Forward", "bwd": "Backward", "opt": "Optimizer"}


def plot_energy_substep(substep_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    """Clustered bar chart: x = substep (fwd/bwd/opt), clusters = batch sizes.

    Each bar is split into two stacked portions:
      - Bottom (hatched //): CPU energy contribution
      - Top    (solid):      non-CPU energy (GPU + RAM)
    """
    # avgs_total / avgs_cpu[phase][bs_index] = mean per-epoch energy (mWh)
    avgs_total: Dict[str, List[float]] = {}
    avgs_cpu:   Dict[str, List[float]] = {}
    for phase in _PHASES:
        avgs_total[phase] = []
        avgs_cpu[phase]   = []
        for bs in BATCH_SIZES:
            df   = substep_data[bs]
            filt = df[df["phase"] == phase]
            by_epoch = filt.groupby("epoch")[["energy_consumed", "cpu_energy"]].sum()
            avgs_total[phase].append(float(by_epoch["energy_consumed"].mean()) * KWH_TO_MWH)
            avgs_cpu[phase].append(float(by_epoch["cpu_energy"].mean()) * KWH_TO_MWH)

    n_phases = len(_PHASES)
    n_bs     = len(BATCH_SIZES)
    x_base   = np.arange(n_phases)
    offsets  = (np.arange(n_bs) - (n_bs - 1) / 2.0) * (0.8 / n_bs)
    width    = 0.8 / n_bs

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "PNA — Substep Energy vs Batch Size\nmean per epoch · CodeCarbon (mWh)",
        fontsize=13, fontweight="bold",
    )

    # Minimum hatched-region height to print a label (6% of tallest bar)
    y_max = max(avgs_total[phase][j] for phase in _PHASES for j in range(n_bs))
    min_label_height = y_max * 0.06

    for i, (bs, color, label) in enumerate(zip(BATCH_SIZES, BS_COLORS, BS_LABELS)):
        xs       = x_base + offsets[i]
        cpu_vals = [avgs_cpu[phase][i]                        for phase in _PHASES]
        top_vals = [avgs_total[phase][i] - avgs_cpu[phase][i] for phase in _PHASES]
        tot_vals = [avgs_total[phase][i]                      for phase in _PHASES]

        # CPU portion — hatched
        ax.bar(xs, cpu_vals, width=width, color=color, alpha=0.85,
               hatch="//", edgecolor="white", linewidth=0.5,
               label=label, zorder=3)
        # Non-CPU portion — solid, stacked on top of CPU portion
        ax.bar(xs, top_vals, width=width, color=color, alpha=0.85,
               bottom=cpu_vals, edgecolor="white", linewidth=0.5,
               zorder=3)

        # Annotate CPU % centred inside the hatched portion
        for x, cpu, tot in zip(xs, cpu_vals, tot_vals):
            if tot == 0 or cpu < min_label_height:
                continue
            pct = cpu / tot * 100
            ax.text(x, cpu / 2, f"{pct:.0f}%",
                    ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="black",
                    zorder=5)

    ax.set_xticks(x_base)
    ax.set_xticklabels([_PHASE_LABELS[p] for p in _PHASES], fontsize=11)
    ax.set_xlabel("Substep")
    ax.set_ylabel("Energy (mWh)")

    # Batch-size legend (colour swatches)
    bs_legend = ax.legend(fontsize=9, loc="upper right")

    # Add a second legend entry explaining the hatch pattern
    import matplotlib.patches as mpatches
    cpu_patch = mpatches.Patch(facecolor="#aaaaaa", hatch="//",
                               edgecolor="white", label="CPU portion (hatched)")
    ax.legend(handles=[*bs_legend.legend_handles, cpu_patch],
              fontsize=9, loc="upper right")

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    _save(fig, out_path, "bs_energy_substep")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).parent.resolve()
    out_dir    = script_dir / "plots"

    print(f"Results directory : {script_dir}")
    print(f"Output directory  : {out_dir}")
    print()

    # Fail fast — check all 9 input files before doing any work.
    check_files(script_dir)

    print("Loading CSVs …")
    utils_data, step_data, substep_data = load_all(script_dir)
    print("Done.\n")

    print("Generating plots:")
    plot_gpu_util_avg(utils_data,    out_dir / "bs_gpu_util.png")
    plot_cpu_util_avg(utils_data,    out_dir / "bs_cpu_util.png")
    plot_energy_total(step_data,     out_dir / "bs_energy_total.png")
    plot_energy_substep(substep_data, out_dir / "bs_energy_substep.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
