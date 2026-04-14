#!/usr/bin/env python3
"""
plot_workers.py — compare PNA metrics across DataLoader worker counts (bs=4096 fixed).

All worker counts must be present; the script exits with an error listing
every missing file rather than silently skipping.

Sources required (6 files total — 2 per worker count):
  pna_result/utils/pna_utils_bs4096_wk{N}_steps.csv
      Step-level hardware utilization samples from PNAUtilsStats.

  pna_result/carbon/pna_carbon_bs4096_wk{N}_steps.csv
      Per-step energy data from PNACarbonStats.  Energy columns are empty for
      steps that did not close a 500 ms measurement window.

Output — pna_result/plots/
------
  1. wk_util_combined.png    — GPU + CPU utilization side by side
  2. wk_energy_combined.png  — total energy + energy by hardware side by side

Usage
-----
    python pna_result/plot_workers.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKER_COUNTS: List[int] = [0, 2, 4]
BS = 4096
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


_HW_COMPONENTS = ["cpu", "gpu", "ram"]
_HW_LABELS     = {"cpu": "CPU", "gpu": "GPU", "ram": "RAM"}


# ---------------------------------------------------------------------------
# Plot 1 — GPU + CPU utilization side by side
# ---------------------------------------------------------------------------

def plot_util_combined(utils_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    gpu_avgs: List[float] = []
    cpu_avgs: List[float] = []
    for wk in WORKER_COUNTS:
        df = utils_data[wk]
        gpu = pd.to_numeric(df.get("gpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        cpu = pd.to_numeric(df.get("cpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        gpu_avgs.append(float(gpu.mean()) if not gpu.empty else float("nan"))
        cpu_avgs.append(float(cpu.mean()) if not cpu.empty else float("nan"))

    fig, (ax_gpu, ax_cpu) = plt.subplots(1, 2, figsize=(12, 3.2))
    fig.suptitle(
        "Utilization Varying Workers",
        fontsize=15, fontweight="bold", y=1,
    )

    x = np.arange(len(WORKER_COUNTS))

    # Left: GPU
    bars_g = ax_gpu.bar(x, gpu_avgs, width=0.5, color=WK_COLORS, alpha=0.85, zorder=3)
    valid_g = [v for v in gpu_avgs if not np.isnan(v)]
    y_max_g = max(valid_g) if valid_g else 1.0
    for bar, val in zip(bars_g, gpu_avgs):
        if np.isnan(val):
            continue
        ax_gpu.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max_g * 0.01,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_gpu.set_xticks(x)
    ax_gpu.set_xticklabels(WK_LABELS, fontsize=13)
    ax_gpu.set_xlabel("Workers", fontsize=14)
    ax_gpu.set_ylabel("GPU Utilization (%)", fontsize=14)
    ax_gpu.set_title("GPU Utilization", fontsize=15)
    ax_gpu.tick_params(axis="y", labelsize=13)
    ax_gpu.set_ylim(0, 105)
    ax_gpu.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_gpu.set_axisbelow(True)

    # Right: CPU
    bars_c = ax_cpu.bar(x, cpu_avgs, width=0.5, color=WK_COLORS, alpha=0.85, zorder=3)
    valid_c = [v for v in cpu_avgs if not np.isnan(v)]
    y_max_c = max(valid_c) if valid_c else 1.0
    for bar, val in zip(bars_c, cpu_avgs):
        if np.isnan(val):
            continue
        ax_cpu.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max_c * 0.01,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_cpu.set_xticks(x)
    ax_cpu.set_xticklabels(WK_LABELS, fontsize=13)
    ax_cpu.set_xlabel("Workers", fontsize=14)
    ax_cpu.set_ylabel("CPU Utilization (%)", fontsize=14)
    ax_cpu.set_title("CPU Utilization", fontsize=15)
    ax_cpu.tick_params(axis="y", labelsize=13)
    ax_cpu.set_ylim(0, 105)
    ax_cpu.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_cpu.set_axisbelow(True)

    fig.subplots_adjust(wspace=0.35)
    fig.tight_layout(rect=[0, 0, 1, 1.05])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 2 — total energy + energy by hardware side by side
# ---------------------------------------------------------------------------

def plot_energy_combined(step_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    """Left: total energy per epoch. Right: clustered hw energy per epoch."""

    # --- Compute total energy per epoch ---
    total_avgs: List[float] = []
    for wk in WORKER_COUNTS:
        df = step_data[wk]
        measured = df[df["energy_consumed"].notna()]
        if measured.empty or "epoch" not in measured.columns:
            total_avgs.append(float("nan"))
            continue
        epoch_totals = measured.groupby("epoch")["energy_consumed"].sum()
        total_avgs.append(float(epoch_totals.mean()) * KWH_TO_MWH)

    # --- Compute per-epoch energy by hardware ---
    hw_avgs: Dict[str, List[float]] = {hw: [] for hw in _HW_COMPONENTS}
    for wk in WORKER_COUNTS:
        df = step_data[wk]
        for hw in _HW_COMPONENTS:
            col = f"{hw}_energy"
            if col not in df.columns or "epoch" not in df.columns:
                hw_avgs[hw].append(float("nan"))
                continue
            measured = df[df[col].notna()]
            if measured.empty:
                hw_avgs[hw].append(float("nan"))
                continue
            epoch_totals = measured.groupby("epoch")[col].sum()
            hw_avgs[hw].append(float(epoch_totals.mean()) * KWH_TO_MWH)

    # --- Figure ---
    fig, (ax_tot, ax_hw) = plt.subplots(1, 2, figsize=(12, 3.2))
    fig.suptitle(
        "Epoch Energy Varying Workers",
        fontsize=15, fontweight="bold", y=1.02,
    )

    x = np.arange(len(WORKER_COUNTS))

    # Left: total energy per epoch
    bars = ax_tot.bar(x, total_avgs, width=0.5, color=WK_COLORS,
                      alpha=0.85, zorder=3)
    valid = [v for v in total_avgs if not np.isnan(v)]
    y_max = max(valid) if valid else 1.0
    for bar, val in zip(bars, total_avgs):
        if np.isnan(val):
            continue
        ax_tot.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_tot.set_xticks(x)
    ax_tot.set_xticklabels(WK_LABELS, fontsize=13)
    ax_tot.set_xlabel("Workers", fontsize=14)
    ax_tot.set_ylabel("Energy per Epoch (mWh)", fontsize=14)
    ax_tot.set_title("Total Energy", fontsize=15)
    ax_tot.tick_params(axis="y", labelsize=13)
    ax_tot.set_ylim(0, y_max * 1.18)
    ax_tot.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_tot.set_axisbelow(True)

    # Right: clustered hardware energy per epoch
    n_hw = len(_HW_COMPONENTS)
    n_wk = len(WORKER_COUNTS)
    x_hw = np.arange(n_hw)
    offsets = (np.arange(n_wk) - (n_wk - 1) / 2.0) * (0.8 / n_wk)
    width = 0.8 / n_wk

    for i, (wk, color, label) in enumerate(zip(WORKER_COUNTS, WK_COLORS, WK_LABELS)):
        xs = x_hw + offsets[i]
        vals = [hw_avgs[hw][i] for hw in _HW_COMPONENTS]
        ax_hw.bar(xs, vals, width=width, color=color, alpha=0.85,
                  label=label, zorder=3, edgecolor="white", linewidth=0.5)

    ax_hw.set_xticks(x_hw)
    ax_hw.set_xticklabels([_HW_LABELS[hw] for hw in _HW_COMPONENTS], fontsize=13)
    ax_hw.set_xlabel("Hardware", fontsize=14)
    ax_hw.set_ylabel("Energy per Epoch (mWh)", fontsize=14)
    ax_hw.set_title("Energy by Hardware", fontsize=15)
    ax_hw.tick_params(axis="y", labelsize=13)
    ax_hw.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_hw.set_axisbelow(True)

    fig.subplots_adjust(wspace=0.35)
    fig.tight_layout(rect=[0, 0, 1, 1.05])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 3 — epoch latency + substep breakdown side by side
# ---------------------------------------------------------------------------

_SUBSTEPS       = ["forward_ms", "backward_ms", "optimizer_ms"]
_SUBSTEP_LABELS = ["Forward", "Backward", "Optimizer"]
_SUBSTEP_COLORS = ["#2A9D8F", "#E76F51", "#8338EC"]   # teal / coral / purple


def plot_latency_combined(step_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    """Left: avg epoch latency. Right: clustered substep bars (x = substep, clusters = wk)."""

    # --- Compute avg epoch latency ---
    epoch_avgs: List[float] = []
    for wk in WORKER_COUNTS:
        df = step_data[wk]
        if "epoch" in df.columns:
            epoch_totals = df.groupby("epoch")["step_ms"].sum()
            epoch_avgs.append(float(epoch_totals.mean()) / 1000.0
                              if not epoch_totals.empty else float("nan"))
        else:
            epoch_avgs.append(float("nan"))

    # --- Compute avg substep times ---
    substep_avgs: Dict[str, List[float]] = {s: [] for s in _SUBSTEPS}
    for wk in WORKER_COUNTS:
        df = step_data[wk]
        for s in _SUBSTEPS:
            col = pd.to_numeric(df.get(s, pd.Series(dtype=float)), errors="coerce").dropna()
            substep_avgs[s].append(float(col.mean()) if not col.empty else float("nan"))

    # --- Figure ---
    fig, (ax_lat, ax_sub) = plt.subplots(1, 2, figsize=(12, 3.2))
    fig.suptitle(
        "Epoch Latency Varying Workers",
        fontsize=15, fontweight="bold", y=1.02,
    )

    x = np.arange(len(WORKER_COUNTS))

    # Left: epoch latency
    bars = ax_lat.bar(x, epoch_avgs, width=0.5, color=WK_COLORS,
                      alpha=0.85, zorder=3)
    valid = [v for v in epoch_avgs if not np.isnan(v)]
    y_max = max(valid) if valid else 1.0
    for bar, val in zip(bars, epoch_avgs):
        if np.isnan(val):
            continue
        ax_lat.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.01,
                    f"{val:.1f}s",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(WK_LABELS, fontsize=13)
    ax_lat.set_xlabel("Workers", fontsize=14)
    ax_lat.set_ylabel("Epoch Latency (s)", fontsize=14)
    ax_lat.set_title("Epoch Latency", fontsize=15)
    ax_lat.tick_params(axis="y", labelsize=13)
    ax_lat.set_ylim(0, y_max * 1.18)
    ax_lat.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_lat.set_axisbelow(True)

    # Right: clustered substep breakdown
    n_sub = len(_SUBSTEPS)
    n_wk  = len(WORKER_COUNTS)
    x_sub = np.arange(n_sub)
    offsets = (np.arange(n_wk) - (n_wk - 1) / 2.0) * (0.8 / n_wk)
    width = 0.8 / n_wk

    for i, (wk, color, label) in enumerate(zip(WORKER_COUNTS, WK_COLORS, WK_LABELS)):
        xs = x_sub + offsets[i]
        vals = [substep_avgs[s][i] for s in _SUBSTEPS]
        ax_sub.bar(xs, vals, width=width, color=color, alpha=0.85,
                   label=label, zorder=3, edgecolor="white", linewidth=0.5)

    ax_sub.set_xticks(x_sub)
    ax_sub.set_xticklabels(_SUBSTEP_LABELS, fontsize=13)
    ax_sub.set_xlabel("Substep", fontsize=14)
    ax_sub.set_ylabel("Mean Substep Time (ms)", fontsize=14)
    ax_sub.set_title("Substep Breakdown", fontsize=15)
    ax_sub.tick_params(axis="y", labelsize=13)
    ax_sub.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_sub.set_axisbelow(True)

    # Global legend at top
    handles, labels = ax_sub.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), ncol=len(labels),
               fontsize=13, frameon=False)

    fig.subplots_adjust(wspace=0.35)
    fig.tight_layout(rect=[0, 0, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


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
    plot_util_combined(utils_data, out_dir / "wk_util_combined.png")
    plot_energy_combined(step_data, out_dir / "wk_energy_combined.png")
    plot_latency_combined(step_data, out_dir / "wk_latency_combined.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
