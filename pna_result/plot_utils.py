#!/usr/bin/env python3
"""
plot_utils.py — visualise outputs produced by PNAUtilsStats.

Usage
-----
# Auto-discover utils/pna_utils_bs*_steps.csv in the same directory:
    python pna_result/plot_utils.py

# Point at a specific steps CSV:
    python pna_result/plot_utils.py --steps pna_result/utils/pna_utils_bs512_steps.csv

Output
------
Plots are saved to  pna_result/utils/plots/

Up to three plots are produced per CSV:
  1. util_gpu.png      — GPU utilisation (%) sampled over steps
  2. util_cpu.png      — per-process CPU utilisation (%) sampled over steps
  3. util_ram.png      — RAM used (GB) sampled over steps

Plots are skipped when the respective column has no sampled data.
Epoch boundaries are annotated with red dashed vertical lines.
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

EPOCH_LINE_COLOR = "#E63946"   # red for epoch boundary markers

GPU_COLOR = "#4C72B0"  # blue — GPU utilisation samples


def _parse_meta(stem: str) -> dict:
    meta = {"batch_size": "?"}
    m = re.search(r"bs(\d+)", stem)
    if m:
        meta["batch_size"] = m.group(1)
    return meta


def _subtitle(meta: dict) -> str:
    return f"batch size {meta['batch_size']}"


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


def _draw_epoch_vlines(ax, epoch_step_ids: List[int]) -> None:
    """Draw red dashed vertical lines at the start of each new epoch."""
    first = True
    for step_id in epoch_step_ids:
        ax.axvline(x=step_id, color=EPOCH_LINE_COLOR,
                   linestyle=":", linewidth=1.2, alpha=0.85,
                   label="Epoch start" if first else None)
        first = False


# ---------------------------------------------------------------------------
# Plot 1 — GPU utilisation (forward and backward as independent series)
# ---------------------------------------------------------------------------

def plot_util_gpu(df: pd.DataFrame, out_path: Path, meta: dict,
                  epoch_step_ids: List[int]) -> None:
    """GPU utilisation plot — single series sampled at stop_forward or stop_backward."""
    series = pd.to_numeric(df.get("gpu_util", pd.Series(dtype=float)), errors="coerce")
    mask   = series.notna()

    if mask.sum() == 0:
        print(f"  1. util_gpu  -> [skip] no sampled data in gpu_util")
        return

    steps = df.loc[mask, "step_idx"].to_numpy()
    vals  = series[mask].to_numpy()
    avg   = float(vals.mean())

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Utils — GPU Utilisation\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.plot(steps, vals, color=GPU_COLOR, linewidth=0.7, alpha=0.45, zorder=1)
    ax.scatter(steps, vals, color=GPU_COLOR, s=22, zorder=3)
    ax.axhline(y=avg, color=GPU_COLOR, linestyle="--", linewidth=1.4,
               alpha=0.85, zorder=2, label=f"avg ({avg:.1f}%)")

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
    print(f"  1. util_gpu  -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — per-process CPU utilisation
# ---------------------------------------------------------------------------

def plot_util_cpu(df: pd.DataFrame, out_path: Path, meta: dict,
                  epoch_step_ids: List[int]) -> None:
    """CPU utilisation plot — single series (per-process)."""
    series = pd.to_numeric(df.get("cpu_util", pd.Series(dtype=float)), errors="coerce")
    mask   = series.notna()
    if mask.sum() == 0:
        print(f"  2. util_cpu  -> [skip] no sampled data for cpu_util")
        return

    sampled_steps = df.loc[mask, "step_idx"].to_numpy()
    sampled_vals  = series[mask].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Utils — CPU Utilisation (per-process)\n{_subtitle(meta)}",
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
    print(f"  2. util_cpu  -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — RAM utilisation
# ---------------------------------------------------------------------------

def plot_util_ram(df: pd.DataFrame, out_path: Path, meta: dict,
                  epoch_step_ids: List[int]) -> None:
    """RAM utilisation plot — y-axis in GB, starting from 0."""
    series = pd.to_numeric(df.get("ram_used_mb", pd.Series(dtype=float)), errors="coerce")
    mask   = series.notna()
    if mask.sum() == 0:
        print(f"  3. util_ram  -> [skip] no sampled data for ram_used_mb")
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
    print(f"  3. util_ram  -> {out_path}")


# ---------------------------------------------------------------------------
# Discovery + orchestration
# ---------------------------------------------------------------------------

def discover_steps_csvs(directory: Path) -> List[Path]:
    """Find pna_utils step CSV files in utils/ subdir (wk2, all batch sizes)."""
    utils_dir = directory / "utils"
    if not utils_dir.is_dir():
        return []
    return sorted(utils_dir.glob("pna_utils_bs*_wk2_steps.csv"))


def process_file(steps_path: Path) -> None:
    print(f"\nProcessing: {steps_path.name}")

    df = pd.read_csv(steps_path)
    required = {"step_idx"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [skip] missing columns: {missing}")
        return

    meta = _parse_meta(steps_path.stem)
    out  = _out_dir(steps_path)
    stem = steps_path.stem

    epoch_starts = _epoch_starts(df)

    plot_util_gpu(df, out / f"{stem}_util_gpu.png", meta, epoch_starts)
    plot_util_cpu(df, out / f"{stem}_util_cpu.png", meta, epoch_starts)
    plot_util_ram(df, out / f"{stem}_util_ram.png", meta, epoch_starts)


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
                f"No pna_utils_bs*_wk2_steps.csv found in {script_dir / 'utils'}.\n"
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
