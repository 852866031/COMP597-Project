#!/usr/bin/env python3
"""
plot_workers.py — compare PNA metrics across DataLoader worker counts (bs=512 fixed).

All worker counts must be present; the script exits with an error listing
every missing file rather than silently skipping.

Sources required (4 files total — 2 per worker count):
  pna_result/utils/pna_utils_bs512_wk{N}_steps.csv
      Step-level hardware utilisation samples from PNAUtilsStats.

  pna_result/carbon/pna_carbon_bs512_wk{N}_steps.csv
      Per-step energy data from PNACarbonStats.  Energy columns are empty for
      steps that did not close a 500 ms measurement window.

Output — pna_result/plots/
------
  1. wk_gpu_util.png         — avg GPU utilisation (%) per worker count
  2. wk_cpu_util.png         — avg CPU utilisation (%) per worker count (per-process)
  3. wk_energy_total.png     — mean per-step avg total energy (mWh) per worker count
  4. wk_energy_hardware.png  — clustered bar: x = hardware (cpu/gpu/ram), clusters = worker counts

Usage
-----
    python pna_result/plot_workers.py
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

WORKER_COUNTS: List[int] = [0, 2, 4]
BS = 512
WK_COLORS: List[str] = ["#4C72B0", "#DD8452", "#55A868"]   # blue / orange / green
WK_LABELS: List[str] = [f"wk{n}" for n in WORKER_COUNTS]

KWH_TO_MWH = 1_000.0     # kWh → mWh


# ---------------------------------------------------------------------------
# File path helpers (exact match — worker count is the sweep variable)
# ---------------------------------------------------------------------------

def _utils_csv(result_dir: Path, wk: int) -> Path:
    return result_dir / "utils" / f"pna_utils_bs{BS}_wk{wk}_steps.csv"


def _step_csv(result_dir: Path, wk: int) -> Path:
    return result_dir / "carbon" / f"pna_carbon_bs{BS}_wk{wk}_steps.csv"


# ---------------------------------------------------------------------------
# File existence check — error on any missing file
# ---------------------------------------------------------------------------

def check_files(result_dir: Path) -> None:
    missing: List[Path] = []
    for wk in WORKER_COUNTS:
        for path in [_utils_csv(result_dir, wk), _step_csv(result_dir, wk)]:
            if not path.exists():
                missing.append(path)
    if missing:
        print("ERROR: the following required CSV files are missing.")
        print("Run  start-pna-workers.sh -wk all  to generate all worker counts.\n")
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
    """Return (utils_data, step_data) dicts keyed by worker count."""
    utils_data: Dict[int, pd.DataFrame] = {}
    step_data:  Dict[int, pd.DataFrame] = {}

    for wk in WORKER_COUNTS:
        utils_data[wk] = _load_utils(_utils_csv(result_dir, wk))
        step_data[wk]  = _enrich_step_df(pd.read_csv(_step_csv(result_dir, wk)))

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


def _simple_bars(ax, totals: List[float], ylabel: str, fmt: str = "{:.3f}",
                 epoch_counts: Optional[List[int]] = None) -> None:
    x    = np.arange(len(WORKER_COUNTS))
    bars = ax.bar(x, totals, width=0.5, color=WK_COLORS, alpha=0.85, zorder=3)

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
    ax.set_xticklabels(WK_LABELS, fontsize=10)
    ax.set_xlabel("DataLoader Workers")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, y_max * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Plot 1 — GPU utilisation average
# ---------------------------------------------------------------------------

def plot_gpu_util_avg(utils_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    avgs: List[float] = []
    for wk in WORKER_COUNTS:
        df  = utils_data[wk]
        col = pd.to_numeric(df.get("gpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        avgs.append(float(col.mean()) if not col.empty else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"PNA — GPU Utilisation vs DataLoader Workers (bs={BS})\n"
        "mean of all sampled forward + backward steps (all epochs)",
        fontsize=13, fontweight="bold",
    )

    _simple_bars(ax, avgs, "GPU Utilisation (%)", fmt="{:.1f}%")
    ax.set_ylim(0, 100)
    _save(fig, out_path, "wk_gpu_util")


# ---------------------------------------------------------------------------
# Plot 2 — CPU utilisation average (per-process)
# ---------------------------------------------------------------------------

def plot_cpu_util_avg(utils_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    avgs: List[float] = []
    for wk in WORKER_COUNTS:
        df  = utils_data[wk]
        col = pd.to_numeric(df.get("cpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        avgs.append(float(col.mean()) if not col.empty else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"PNA — CPU Utilisation vs DataLoader Workers (bs={BS})\n"
        "mean of all sampled forward + backward steps (all epochs)",
        fontsize=13, fontweight="bold",
    )

    _simple_bars(ax, avgs, "CPU Utilisation (%)", fmt="{:.1f}%")
    ax.set_ylim(0, 100)
    _save(fig, out_path, "wk_cpu_util")


# ---------------------------------------------------------------------------
# Plot 3 — total energy aggregated across all epochs
# ---------------------------------------------------------------------------

def plot_energy_total(step_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    avgs: List[float] = []
    for wk in WORKER_COUNTS:
        df = step_data[wk]
        col = df["energy_consumed_per_step"].dropna()
        avgs.append(float(col.mean()) * KWH_TO_MWH if not col.empty else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"PNA — Per-Step Avg Total Energy vs DataLoader Workers (bs={BS})\n"
        "mean over measurement windows · CodeCarbon (mWh)",
        fontsize=13, fontweight="bold",
    )

    _simple_bars(ax, avgs, "Energy (mWh)", fmt="{:.3f}")
    _save(fig, out_path, "wk_energy_total")


# ---------------------------------------------------------------------------
# Plot 4 — per-step avg energy breakdown by hardware component
# ---------------------------------------------------------------------------

_HW_COMPONENTS = ["cpu", "gpu", "ram"]
_HW_LABELS     = {"cpu": "CPU", "gpu": "GPU", "ram": "RAM"}
_HW_COLORS     = {"cpu": "#E76F51", "gpu": "#2A9D8F", "ram": "#8338EC"}


def plot_energy_hardware(step_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    """Clustered bar chart: x = hardware component (cpu/gpu/ram), clusters = worker counts."""
    avgs: Dict[str, List[float]] = {hw: [] for hw in _HW_COMPONENTS}
    for wk in WORKER_COUNTS:
        df = step_data[wk]
        for hw in _HW_COMPONENTS:
            col = f"{hw}_energy_per_step"
            vals = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
            avgs[hw].append(float(vals.mean()) * KWH_TO_MWH if not vals.empty else float("nan"))

    n_hw = len(_HW_COMPONENTS)
    n_wk = len(WORKER_COUNTS)
    x_base  = np.arange(n_hw)
    offsets = (np.arange(n_wk) - (n_wk - 1) / 2.0) * (0.8 / n_wk)
    width   = 0.8 / n_wk

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"PNA — Per-Step Avg Energy by Hardware vs DataLoader Workers (bs={BS})\n"
        "mean over measurement windows · CodeCarbon (mWh)",
        fontsize=13, fontweight="bold",
    )

    for i, (wk, color, label) in enumerate(zip(WORKER_COUNTS, WK_COLORS, WK_LABELS)):
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

    _save(fig, out_path, "wk_energy_hardware")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).parent.resolve()
    out_dir    = script_dir / "plots"

    print(f"Results directory : {script_dir}")
    print(f"Output directory  : {out_dir}")
    print()

    check_files(script_dir)

    print("Loading CSVs …")
    utils_data, step_data = load_all(script_dir)
    print("Done.\n")

    print("Generating plots:")
    plot_gpu_util_avg(utils_data,   out_dir / "wk_gpu_util.png")
    plot_cpu_util_avg(utils_data,   out_dir / "wk_cpu_util.png")
    plot_energy_total(step_data,    out_dir / "wk_energy_total.png")
    plot_energy_hardware(step_data, out_dir / "wk_energy_hardware.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
