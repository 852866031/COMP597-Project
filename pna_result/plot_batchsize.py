#!/usr/bin/env python3
"""
plot_batchsize.py — compare PNA metrics across batch sizes 512, 1024, 2048, and 4096.

All batch sizes must be present; the script exits with an error listing
every missing file rather than silently skipping.

Sources required (8 files total — 2 per batch size):
  pna_result/utils/pna_utils_bs{N}_wk2_steps.csv
      Step-level hardware utilisation samples from PNAUtilsStats.

  pna_result/carbon/pna_carbon_bs{N}_wk2_steps.csv
      Per-step energy data from PNACarbonStats.  Energy columns are empty for
      steps that did not close a 500 ms measurement window; non-empty rows hold
      accumulated energy for the window.

Output — pna_result/plots/
------
  1. bs_gpu_util.png        — avg GPU utilisation (%) across all epochs, per batch size
  2. bs_cpu_util.png        — avg CPU utilisation (%) across all epochs, per batch size (per-process)
  3. bs_energy_total.png    — avg energy per epoch (mWh), per batch size
  4. bs_energy_hardware.png — clustered bar: x = hardware (cpu/gpu/ram), clusters = batch sizes

Usage
-----
    python pna_result/plot_batchsize.py
"""
from __future__ import annotations

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

BATCH_SIZES: List[int] = [512, 1024, 2048, 4096]
BS_COLORS:   List[str] = ["#4C72B0", "#DD8452", "#55A868", "#8338EC"]  # blue / orange / green / purple
BS_LABELS:   List[str] = [f"bs{n}" for n in BATCH_SIZES]

KWH_TO_MWH = 1_000.0     # kWh → mWh


# ---------------------------------------------------------------------------
# File path helpers (glob-based — default wk2)
# ---------------------------------------------------------------------------

def _glob_one(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _utils_csv(result_dir: Path, bs: int) -> Optional[Path]:
    return _glob_one(result_dir / "utils", f"pna_utils_bs{bs}_wk2_steps.csv")


def _step_csv(result_dir: Path, bs: int) -> Optional[Path]:
    return _glob_one(result_dir / "carbon", f"pna_carbon_bs{bs}_wk2_steps.csv")


# ---------------------------------------------------------------------------
# File existence check — error on any missing file
# ---------------------------------------------------------------------------

def check_files(result_dir: Path) -> None:
    missing: List[str] = []
    for bs in BATCH_SIZES:
        checks = [
            (_utils_csv(result_dir, bs), f"utils/pna_utils_bs{bs}_wk2_steps.csv"),
            (_step_csv(result_dir, bs),  f"carbon/pna_carbon_bs{bs}_wk2_steps.csv"),
        ]
        for path, pattern in checks:
            if path is None:
                missing.append(f"{result_dir}/{pattern}")
    if missing:
        print("ERROR: the following required CSV files are missing.")
        print("Run the start-pna-*.sh scripts with -bs 512, -bs 1024, -bs 2048, and -bs 4096 first.\n")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CSV loading & enrichment
# ---------------------------------------------------------------------------

def _load_utils(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _enrich_step_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add window_steps and _per_step normalised energy columns."""
    df = df.copy()
    df["step_idx"] = pd.to_numeric(df["step_idx"], errors="coerce").astype("Int64")
    df["epoch"]    = pd.to_numeric(df["epoch"],    errors="coerce").astype("Int64")
    energy_cols = ("energy_consumed", "cpu_energy", "gpu_energy", "ram_energy", "emissions")
    for col in energy_cols:
        df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
    df = df.dropna(subset=["step_idx"]).sort_values("step_idx").reset_index(drop=True)
    measured_pos = df.index[df["energy_consumed"].notna()].tolist()
    window_steps_arr = np.full(len(df), np.nan)
    prev = -1
    for pos in measured_pos:
        window_steps_arr[pos] = pos - prev
        prev = pos
    df["window_steps"] = window_steps_arr
    for col in energy_cols:
        df[f"{col}_per_step"] = df[col] / df["window_steps"]
    return df


def load_all(result_dir: Path):
    """Return (utils_data, step_data) dicts keyed by batch size."""
    utils_data: Dict[int, pd.DataFrame] = {}
    step_data:  Dict[int, pd.DataFrame] = {}

    for bs in BATCH_SIZES:
        utils_path = _utils_csv(result_dir, bs)
        step_path  = _step_csv(result_dir, bs)
        assert utils_path and step_path, \
            f"Missing CSV for bs={bs} (run check_files first)"
        utils_data[bs] = _load_utils(utils_path)
        step_data[bs]  = _enrich_step_df(pd.read_csv(step_path))

    return utils_data, step_data


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
        col = pd.to_numeric(df.get("gpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        avgs.append(float(col.mean()) if not col.empty else float("nan"))

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
        col = pd.to_numeric(df.get("cpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        avgs.append(float(col.mean()) if not col.empty else float("nan"))

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
        measured = df[df["energy_consumed"].notna()]
        if measured.empty or "epoch" not in measured.columns:
            avgs.append(float("nan"))
            continue
        # Sum all measurement-window energy within each epoch, then average epochs
        epoch_totals = measured.groupby("epoch")["energy_consumed"].sum()
        avgs.append(float(epoch_totals.mean()) * KWH_TO_MWH)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "PNA — Avg Energy per Epoch vs Batch Size\n"
        "mean over all epochs · CodeCarbon (mWh)",
        fontsize=13, fontweight="bold",
    )

    _simple_bars(ax, avgs, "Energy per Epoch (mWh)", fmt="{:.3f}")
    _save(fig, out_path, "bs_energy_total")


# ---------------------------------------------------------------------------
# Plot 4 — per-step avg energy breakdown by hardware component
# ---------------------------------------------------------------------------

_HW_COMPONENTS = ["cpu", "gpu", "ram"]
_HW_LABELS     = {"cpu": "CPU", "gpu": "GPU", "ram": "RAM"}
_HW_COLORS     = {"cpu": "#E76F51", "gpu": "#2A9D8F", "ram": "#8338EC"}


def plot_energy_hardware(step_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    """Clustered bar chart: x = hardware component (cpu/gpu/ram), clusters = batch sizes."""
    avgs: Dict[str, List[float]] = {hw: [] for hw in _HW_COMPONENTS}
    for bs in BATCH_SIZES:
        df = step_data[bs]
        for hw in _HW_COMPONENTS:
            col = f"{hw}_energy_per_step"
            vals = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
            avgs[hw].append(float(vals.mean()) * KWH_TO_MWH if not vals.empty else float("nan"))

    n_hw = len(_HW_COMPONENTS)
    n_bs = len(BATCH_SIZES)
    x_base  = np.arange(n_hw)
    offsets = (np.arange(n_bs) - (n_bs - 1) / 2.0) * (0.8 / n_bs)
    width   = 0.8 / n_bs

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "PNA — Per-Step Avg Energy by Hardware vs Batch Size\n"
        "mean over measurement windows · CodeCarbon (mWh)",
        fontsize=13, fontweight="bold",
    )

    for i, (bs, color, label) in enumerate(zip(BATCH_SIZES, BS_COLORS, BS_LABELS)):
        xs   = x_base + offsets[i]
        vals = [avgs[hw][i] for hw in _HW_COMPONENTS]
        ax.bar(xs, vals, width=width, color=color, alpha=0.85,
               label=label, zorder=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_base)
    ax.set_xticklabels([_HW_LABELS[hw] for hw in _HW_COMPONENTS], fontsize=11)
    ax.set_xlabel("Hardware Component")
    ax.set_ylabel("Energy per step (mWh)")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    _save(fig, out_path, "bs_energy_hardware")


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
    utils_data, step_data = load_all(script_dir)
    print("Done.\n")

    print("Generating plots:")
    plot_gpu_util_avg(utils_data,  out_dir / "bs_gpu_util.png")
    plot_cpu_util_avg(utils_data,  out_dir / "bs_cpu_util.png")
    plot_energy_total(step_data,   out_dir / "bs_energy_total.png")
    plot_energy_hardware(step_data, out_dir / "bs_energy_hardware.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
