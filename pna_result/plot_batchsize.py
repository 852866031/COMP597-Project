#!/usr/bin/env python3
"""
plot_batchsize.py — compare PNA metrics across batch sizes 512, 1024, 2048, and 4096.

All batch sizes must be present; the script exits with an error listing
every missing file rather than silently skipping.

Sources required (8 files total — 2 per batch size):
  pna_result/utils/pna_utils_bs{N}_wk2_steps.csv
      Step-level hardware utization samples from PNAUtilsStats.

  pna_result/carbon/pna_carbon_bs{N}_wk2_steps.csv
      Per-step energy data from PNACarbonStats.  Energy columns are empty for
      steps that did not close a 500 ms measurement window; non-empty rows hold
      accumulated energy for the window.

Output — pna_result/plots/
------
  1. bs_gpu_util.png        — avg GPU utization (%) across all epochs, per batch size
  2. bs_cpu_util.png        — avg CPU utization (%) across all epochs, per batch size (per-process)
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
            ha="center", va="bottom", fontsize=12, fontweight="bold",
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
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
        if epoch_counts is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top + y_max * 0.055,
                f"({epoch_counts[i]} epochs)",
                ha="center", va="bottom", fontsize=11, color="#555555",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(BS_LABELS, fontsize=13)
    ax.set_xlabel("Batch Size", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, y_max * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Plot 1 — GPU utization average
# ---------------------------------------------------------------------------

def plot_gpu_util_avg(utils_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    avgs: List[float] = []
    for bs in BATCH_SIZES:
        df  = utils_data[bs]
        col = pd.to_numeric(df.get("gpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        avgs.append(float(col.mean()) if not col.empty else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "PNA — GPU Utization vs Batch Size\n"
        "mean of all sampled forward + backward steps (all epochs)",
        fontsize=15, fontweight="bold",
    )

    _simple_bars(ax, avgs, "GPU Utization (%)", fmt="{:.1f}%")
    ax.set_ylim(0, 100)
    _save(fig, out_path, "bs_gpu_util")


# ---------------------------------------------------------------------------
# Plot 2 — CPU utization average (per-process)
# ---------------------------------------------------------------------------

def plot_cpu_util_avg(utils_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    avgs: List[float] = []
    for bs in BATCH_SIZES:
        df  = utils_data[bs]
        col = pd.to_numeric(df.get("cpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        avgs.append(float(col.mean()) if not col.empty else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "PNA — CPU Utization vs Batch Size (per-process)\n"
        "mean of all sampled forward + backward steps (all epochs)",
        fontsize=15, fontweight="bold",
    )

    _simple_bars(ax, avgs, "CPU Utization (%)", fmt="{:.1f}%")
    ax.set_ylim(0, 100)
    _save(fig, out_path, "bs_cpu_util")


# ---------------------------------------------------------------------------
# Plot 2b — GPU + CPU utilization side by side
# ---------------------------------------------------------------------------

def plot_util_combined(utils_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    gpu_avgs: List[float] = []
    cpu_avgs: List[float] = []
    for bs in BATCH_SIZES:
        df = utils_data[bs]
        gpu = pd.to_numeric(df.get("gpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        cpu = pd.to_numeric(df.get("cpu_util", pd.Series(dtype=float)), errors="coerce").dropna()
        gpu_avgs.append(float(gpu.mean()) if not gpu.empty else float("nan"))
        cpu_avgs.append(float(cpu.mean()) if not cpu.empty else float("nan"))

    fig, (ax_gpu, ax_cpu) = plt.subplots(1, 2, figsize=(12, 3))

    x = np.arange(len(BATCH_SIZES))

    # Left: GPU
    bars_g = ax_gpu.bar(x, gpu_avgs, width=0.5, color=BS_COLORS, alpha=0.85, zorder=3)
    valid_g = [v for v in gpu_avgs if not np.isnan(v)]
    y_max_g = max(valid_g) if valid_g else 1.0
    for bar, val in zip(bars_g, gpu_avgs):
        if np.isnan(val):
            continue
        ax_gpu.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max_g * 0.01,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_gpu.set_xticks(x)
    ax_gpu.set_xticklabels(BS_LABELS, fontsize=13)
    ax_gpu.set_xlabel("Batch Size", fontsize=14)
    ax_gpu.set_ylabel("GPU Utilization (%)", fontsize=14)
    ax_gpu.set_title("GPU Utilization", fontsize=15)
    ax_gpu.tick_params(axis="y", labelsize=13)
    ax_gpu.set_ylim(0, 105)
    ax_gpu.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_gpu.set_axisbelow(True)

    # Right: CPU
    bars_c = ax_cpu.bar(x, cpu_avgs, width=0.5, color=BS_COLORS, alpha=0.85, zorder=3)
    valid_c = [v for v in cpu_avgs if not np.isnan(v)]
    y_max_c = max(valid_c) if valid_c else 1.0
    for bar, val in zip(bars_c, cpu_avgs):
        if np.isnan(val):
            continue
        ax_cpu.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max_c * 0.01,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_cpu.set_xticks(x)
    ax_cpu.set_xticklabels(BS_LABELS, fontsize=13)
    ax_cpu.set_xlabel("Batch Size", fontsize=14)
    ax_cpu.set_ylabel("CPU Utilization (%)", fontsize=14)
    ax_cpu.set_title("CPU Utilization", fontsize=15)
    ax_cpu.tick_params(axis="y", labelsize=13)
    ax_cpu.set_ylim(0, 105)
    ax_cpu.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_cpu.set_axisbelow(True)

    _save(fig, out_path, "bs_util_combined")


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
        fontsize=15, fontweight="bold",
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
        fontsize=15, fontweight="bold",
    )

    for i, (bs, color, label) in enumerate(zip(BATCH_SIZES, BS_COLORS, BS_LABELS)):
        xs   = x_base + offsets[i]
        vals = [avgs[hw][i] for hw in _HW_COMPONENTS]
        ax.bar(xs, vals, width=width, color=color, alpha=0.85,
               label=label, zorder=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_base)
    ax.set_xticklabels([_HW_LABELS[hw] for hw in _HW_COMPONENTS], fontsize=13)
    ax.set_xlabel("Hardware Component", fontsize=14)
    ax.set_ylabel("Energy per step (mWh)", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(fontsize=13, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    _save(fig, out_path, "bs_energy_hardware")


# ---------------------------------------------------------------------------
# Plot 4b — per-epoch energy breakdown by hardware component
# ---------------------------------------------------------------------------

def plot_energy_hardware_epoch(step_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    """Clustered bar chart: x = hardware (cpu/gpu/ram), clusters = batch sizes.
    Values are mean per-epoch energy (sum within epoch, averaged across epochs)."""
    avgs: Dict[str, List[float]] = {hw: [] for hw in _HW_COMPONENTS}
    for bs in BATCH_SIZES:
        df = step_data[bs]
        for hw in _HW_COMPONENTS:
            col = f"{hw}_energy"
            if col not in df.columns or "epoch" not in df.columns:
                avgs[hw].append(float("nan"))
                continue
            measured = df[df[col].notna()]
            if measured.empty:
                avgs[hw].append(float("nan"))
                continue
            epoch_totals = measured.groupby("epoch")[col].sum()
            avgs[hw].append(float(epoch_totals.mean()) * KWH_TO_MWH)

    n_hw = len(_HW_COMPONENTS)
    n_bs = len(BATCH_SIZES)
    x_base  = np.arange(n_hw)
    offsets = (np.arange(n_bs) - (n_bs - 1) / 2.0) * (0.8 / n_bs)
    width   = 0.8 / n_bs

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "PNA — Avg Energy per Epoch by Hardware vs Batch Size\n"
        "mean over all epochs · CodeCarbon (mWh)",
        fontsize=15, fontweight="bold",
    )

    for i, (bs, color, label) in enumerate(zip(BATCH_SIZES, BS_COLORS, BS_LABELS)):
        xs   = x_base + offsets[i]
        vals = [avgs[hw][i] for hw in _HW_COMPONENTS]
        ax.bar(xs, vals, width=width, color=color, alpha=0.85,
               label=label, zorder=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_base)
    ax.set_xticklabels([_HW_LABELS[hw] for hw in _HW_COMPONENTS], fontsize=13)
    ax.set_xlabel("Hardware Component", fontsize=14)
    ax.set_ylabel("Energy per epoch (mWh)", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(fontsize=13, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    _save(fig, out_path, "bs_energy_hardware_epoch")


# ---------------------------------------------------------------------------
# Plot 5 — avg step latency + avg epoch latency (two subplots)
# ---------------------------------------------------------------------------

def plot_latency(utils_data: Dict[int, pd.DataFrame], out_path: Path) -> None:
    """Side-by-side bar charts: mean step latency (ms) and mean epoch latency (s)."""
    step_avgs:  List[float] = []
    epoch_avgs: List[float] = []

    for bs in BATCH_SIZES:
        df = utils_data[bs]
        step_ms = pd.to_numeric(df.get("step_ms", pd.Series(dtype=float)),
                                errors="coerce").dropna()
        step_avgs.append(float(step_ms.mean()) if not step_ms.empty else float("nan"))

        if "epoch" in df.columns:
            epoch_totals = df.groupby("epoch")["step_ms"].sum()
            epoch_avgs.append(float(epoch_totals.mean()) / 1000.0 if not epoch_totals.empty
                              else float("nan"))
        else:
            epoch_avgs.append(float("nan"))

    fig, (ax_step, ax_epoch) = plt.subplots(1, 2, figsize=(12, 3.5))

    # Left: avg step latency (ms)
    x = np.arange(len(BATCH_SIZES))
    bars_s = ax_step.bar(x, step_avgs, width=0.5, color=BS_COLORS,
                         alpha=0.85, zorder=3)
    valid_s = [v for v in step_avgs if not np.isnan(v)]
    y_max_s = max(valid_s) if valid_s else 1.0
    for bar, val in zip(bars_s, step_avgs):
        if np.isnan(val):
            continue
        ax_step.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + y_max_s * 0.01,
                     f"{val:.0f} ms",
                     ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_step.set_xticks(x)
    ax_step.set_xticklabels(BS_LABELS, fontsize=13)
    ax_step.set_xlabel("Batch Size", fontsize=14)
    ax_step.set_ylabel("Mean Step Latency (ms)", fontsize=14)
    ax_step.set_title("Per Step Latency", fontsize=15)
    ax_step.tick_params(axis="y", labelsize=13)
    ax_step.set_ylim(0, y_max_s * 1.18)
    ax_step.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_step.set_axisbelow(True)

    # Right: avg epoch latency (s)
    bars_e = ax_epoch.bar(x, epoch_avgs, width=0.5, color=BS_COLORS,
                          alpha=0.85, zorder=3)
    valid_e = [v for v in epoch_avgs if not np.isnan(v)]
    y_max_e = max(valid_e) if valid_e else 1.0
    for bar, val in zip(bars_e, epoch_avgs):
        if np.isnan(val):
            continue
        ax_epoch.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + y_max_e * 0.01,
                      f"{val:.1f}s",
                      ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_epoch.set_xticks(x)
    ax_epoch.set_xticklabels(BS_LABELS, fontsize=13)
    ax_epoch.set_xlabel("Batch Size", fontsize=14)
    ax_epoch.set_ylabel("Mean Epoch Latency (s)", fontsize=14)
    ax_epoch.set_title("Per Epoch Latency", fontsize=15)
    ax_epoch.tick_params(axis="y", labelsize=13)
    ax_epoch.set_ylim(0, y_max_e * 1.18)
    ax_epoch.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_epoch.set_axisbelow(True)

    _save(fig, out_path, "bs_latency")


# ---------------------------------------------------------------------------
# Plot 6 — epoch latency + epoch energy by hardware (side by side)
# ---------------------------------------------------------------------------

def plot_latency_energy_combined(
    utils_data: Dict[int, pd.DataFrame],
    step_data: Dict[int, pd.DataFrame],
    out_path: Path,
) -> None:
    """Left: total energy per epoch bar chart. Right: clustered hw energy per epoch."""

    # --- Compute total energy per epoch ---
    total_avgs: List[float] = []
    for bs in BATCH_SIZES:
        df = step_data[bs]
        measured = df[df["energy_consumed"].notna()]
        if measured.empty or "epoch" not in measured.columns:
            total_avgs.append(float("nan"))
            continue
        epoch_totals = measured.groupby("epoch")["energy_consumed"].sum()
        total_avgs.append(float(epoch_totals.mean()) * KWH_TO_MWH)

    # --- Compute per-epoch energy by hardware ---
    hw_avgs: Dict[str, List[float]] = {hw: [] for hw in _HW_COMPONENTS}
    for bs in BATCH_SIZES:
        df = step_data[bs]
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
    fig, (ax_tot, ax_eng) = plt.subplots(1, 2, figsize=(12, 3))

    x = np.arange(len(BATCH_SIZES))

    # Left: total energy per epoch
    bars = ax_tot.bar(x, total_avgs, width=0.5, color=BS_COLORS,
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
    ax_tot.set_xticklabels(BS_LABELS, fontsize=13)
    ax_tot.set_xlabel("Batch Size", fontsize=14)
    ax_tot.set_ylabel("Energy per Epoch (mWh)", fontsize=14)
    ax_tot.set_title("Total Energy", fontsize=15)
    ax_tot.tick_params(axis="y", labelsize=13)
    ax_tot.set_ylim(0, y_max * 1.18)
    ax_tot.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_tot.set_axisbelow(True)

    # Right: clustered hardware energy per epoch
    n_hw = len(_HW_COMPONENTS)
    n_bs = len(BATCH_SIZES)
    x_hw = np.arange(n_hw)
    offsets = (np.arange(n_bs) - (n_bs - 1) / 2.0) * (0.8 / n_bs)
    width = 0.8 / n_bs

    for i, (bs, color, label) in enumerate(zip(BATCH_SIZES, BS_COLORS, BS_LABELS)):
        xs = x_hw + offsets[i]
        vals = [hw_avgs[hw][i] for hw in _HW_COMPONENTS]
        ax_eng.bar(xs, vals, width=width, color=color, alpha=0.85,
                   label=label, zorder=3, edgecolor="white", linewidth=0.5)

    ax_eng.set_xticks(x_hw)
    ax_eng.set_xticklabels([_HW_LABELS[hw] for hw in _HW_COMPONENTS], fontsize=13)
    ax_eng.set_xlabel("Hardware", fontsize=14)
    ax_eng.set_ylabel("Energy per Epoch (mWh)", fontsize=14)
    ax_eng.set_title("Energy by Hardware", fontsize=15)
    ax_eng.tick_params(axis="y", labelsize=13)
    ax_eng.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_eng.set_axisbelow(True)

    # Global legend at top
    handles, labels = ax_eng.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.09), ncol=len(labels),
               fontsize=13, frameon=False)

    fig.subplots_adjust(wspace=0.35)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, out_path, "bs_energy_combined")


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
    plot_util_combined(utils_data, out_dir / "bs_util_combined.png")
    plot_latency(utils_data,       out_dir / "bs_latency.png")
    plot_energy_total(step_data,   out_dir / "bs_energy_total.png")
    plot_energy_hardware(step_data, out_dir / "bs_energy_hardware.png")
    plot_energy_hardware_epoch(step_data, out_dir / "bs_energy_hardware_epoch.png")
    plot_latency_energy_combined(utils_data, step_data, out_dir / "bs_energy_combined.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
