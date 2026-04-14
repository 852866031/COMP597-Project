#!/usr/bin/env python3
"""
plot_carbon.py — visualise outputs produced by PNACarbonStats.

CSV format (pna_carbon_bs<N>_wk<M>_steps.csv)
----------------------------------------------
One row per training step.  Energy columns are empty for steps that did not
close a 500 ms measurement window.  For rows that do have energy data the
value is the *accumulated* energy since the previous measurement; the plotting
code divides by ``window_steps`` (automatically computed) to obtain a
per-step average before displaying.

Usage
-----
# Auto-discover carbon/pna_carbon_bs*_wk*_steps.csv in the same directory:
    python pna_result/plot_carbon.py

# Point at a specific steps CSV:
    python pna_result/plot_carbon.py --step pna_result/carbon/pna_carbon_bs4096_wk2_steps.csv

Output
------
Plots are saved to  pna_result/carbon/plots/

Six plots are produced per run:
  1. energy_total.png      — per-step avg total energy (mWh) at each measured window
  2. energy_hardware.png   — stacked area: CPU / GPU / RAM per-step avg energy
  3. carbon_total.png      — per-step avg CO₂ emissions (µg CO₂eq) at each window
  4. carbon_hardware.png   — stacked area: CPU / GPU / RAM derived emissions
  5. pancake_energy_hardware.png  — donut chart: avg energy share by hardware
  6. pancake_carbon_hardware.png  — donut chart: avg emissions share by hardware
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

HW_COLORS = {
    "cpu": "#E76F51",   # terracotta
    "gpu": "#2A9D8F",   # teal
    "ram": "#8338EC",   # purple
}
HW_LABELS = {
    "cpu": "CPU",
    "gpu": "GPU",
    "ram": "RAM",
}
HW_ORDER = ["cpu", "gpu", "ram"]

EPOCH_LINE_COLOR = "#E63946"

KWH_TO_MWH  = 1_000.0
KG_TO_UG    = 1_000_000_000.0


# ---------------------------------------------------------------------------
# Step-CSV loading and window normalisation
# ---------------------------------------------------------------------------

def _enrich_step_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the per-step CSV; add window_steps and _per_step normalised columns.

    For each row with energy data, ``window_steps`` is the number of steps
    (including this one) since the previous measured row.  The ``_per_step``
    columns divide each energy field by its window_steps to give a per-step
    average; these are used in all plots.
    """
    df = df.copy()
    df["step_idx"] = pd.to_numeric(df["step_idx"], errors="coerce").astype("Int64")
    df["epoch"]    = pd.to_numeric(df["epoch"],    errors="coerce").astype("Int64")
    df["step_ms"]  = pd.to_numeric(df.get("step_ms", pd.Series(dtype=float)),
                                   errors="coerce")
    energy_cols = ("energy_consumed", "cpu_energy", "gpu_energy", "ram_energy", "emissions")
    for col in energy_cols:
        df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")

    df = df.dropna(subset=["step_idx"]).sort_values("step_idx").reset_index(drop=True)

    # Compute window_steps for each measured row.
    measured_pos = df.index[df["energy_consumed"].notna()].tolist()
    window_steps_arr = np.full(len(df), np.nan)
    prev = -1
    for pos in measured_pos:
        window_steps_arr[pos] = pos - prev   # steps covered: prev+1 … pos
        prev = pos
    df["window_steps"] = window_steps_arr

    # Per-step normalised energy columns
    for col in energy_cols:
        df[f"{col}_per_step"] = df[col] / df["window_steps"]

    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_meta(stem: str) -> dict:
    meta = {"batch_size": "?"}
    m = re.search(r"bs(\d+)", stem)
    if m:
        meta["batch_size"] = m.group(1)
    return meta


def _subtitle(meta: dict) -> str:
    return f"batch size {meta['batch_size']}"


def _out_dir(step_csv_path: Path) -> Path:
    d = step_csv_path.parent / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _epoch_starts(df: pd.DataFrame) -> List[int]:
    """Return step_idx values where the epoch number changes (skip first row)."""
    if "epoch" not in df.columns or "step_idx" not in df.columns:
        return []
    changed = df["epoch"].diff().fillna(0) != 0
    changed.iloc[0] = False
    return df.loc[changed, "step_idx"].tolist()


def _draw_epoch_vlines(ax, epoch_starts: List[int]) -> None:
    first = True
    for s in epoch_starts:
        ax.axvline(x=s, color=EPOCH_LINE_COLOR, linestyle=":",
                   linewidth=1.2, alpha=0.85,
                   label="Epoch start" if first else None)
        first = False


def _measured_bars(df: pd.DataFrame, col_per_step: str, scale: float = 1.0):
    """Return (x_center, widths, y) arrays for measured rows only.

    Each bar is centred on the window's midpoint and spans ``window_steps``
    wide — visually conveying the interval covered by each measurement.
    """
    m = df[col_per_step].notna()
    dm = df[m]
    window_start = (dm["step_idx"] - dm["window_steps"] + 1).to_numpy()
    window_end   = dm["step_idx"].to_numpy()
    x_center     = (window_start + window_end) / 2.0
    widths       = dm["window_steps"].to_numpy()
    y            = dm[col_per_step].to_numpy() * scale
    return x_center, widths, y


# ---------------------------------------------------------------------------
# Plot 1 — per-step avg total energy
# ---------------------------------------------------------------------------

def plot_energy_total(step_df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    x, w, y = _measured_bars(step_df, "energy_consumed_per_step", KWH_TO_MWH)
    if len(x) == 0:
        print("  1. energy_total     -> [skip] no measured rows")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — Per-Step Avg Total Energy\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.bar(x, y, width=w, color="#457B9D", alpha=0.80, align="center")

    ax.set_xlabel("Step")
    ax.set_ylabel("Energy per step (mWh)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  1. energy_total     -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — per-step avg energy by hardware (stacked area over measured windows)
# ---------------------------------------------------------------------------

def plot_energy_hardware(step_df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    m = step_df["energy_consumed_per_step"].notna()
    dm = step_df[m].copy()
    if dm.empty:
        print("  2. energy_hardware  -> [skip] no measured rows")
        return

    x = dm["step_idx"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — Per-Step Avg Energy by Hardware\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    bottoms = np.zeros(len(dm))
    for hw in HW_ORDER:
        col = f"{hw}_energy_per_step"
        if col not in dm.columns:
            continue
        vals = dm[col].fillna(0.0).to_numpy() * KWH_TO_MWH
        ax.fill_between(x, bottoms, bottoms + vals,
                        alpha=0.80, color=HW_COLORS[hw],
                        label=HW_LABELS[hw], step="mid")
        bottoms += vals

    ax.set_xlabel("Step")
    ax.set_ylabel("Energy per step (mWh)")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  2. energy_hardware  -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — per-step avg CO₂ emissions
# ---------------------------------------------------------------------------

def plot_carbon_total(step_df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    x, w, y = _measured_bars(step_df, "emissions_per_step", KG_TO_UG)
    if len(x) == 0:
        print("  3. carbon_total     -> [skip] no measured rows")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — Per-Step Avg CO₂ Emissions\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.bar(x, y, width=w, color="#E63946", alpha=0.80, align="center")

    ax.set_xlabel("Step")
    ax.set_ylabel("CO₂ per step (µg CO₂eq)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  3. carbon_total     -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — per-step avg CO₂ emissions by hardware
# ---------------------------------------------------------------------------

def plot_carbon_hardware(step_df: pd.DataFrame, out_path: Path, meta: dict) -> None:
    """Derive per-hardware emissions by apportioning total emissions by energy share."""
    m = step_df["energy_consumed_per_step"].notna()
    dm = step_df[m].copy()
    if dm.empty:
        print("  4. carbon_hardware  -> [skip] no measured rows")
        return

    x            = dm["step_idx"].to_numpy()
    total_energy = dm["energy_consumed_per_step"].to_numpy()
    total_emit   = dm["emissions_per_step"].fillna(0.0).to_numpy()
    safe_energy  = np.where(total_energy > 0, total_energy, np.nan)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — Per-Step Avg CO₂ by Hardware\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    bottoms = np.zeros(len(dm))
    for hw in HW_ORDER:
        col = f"{hw}_energy_per_step"
        if col not in dm.columns:
            continue
        hw_energy = dm[col].fillna(0.0).to_numpy()
        hw_emit = np.where(
            np.isfinite(safe_energy),
            total_emit * (hw_energy / safe_energy),
            0.0,
        ) * KG_TO_UG
        ax.fill_between(x, bottoms, bottoms + hw_emit,
                        alpha=0.80, color=HW_COLORS[hw],
                        label=HW_LABELS[hw], step="mid")
        bottoms += hw_emit

    ax.set_xlabel("Step")
    ax.set_ylabel("CO₂ per step (µg CO₂eq)")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  4. carbon_hardware  -> {out_path}")


# ---------------------------------------------------------------------------
# Shared pancake helper
# ---------------------------------------------------------------------------

def _draw_pancake(ax, values, labels, colors, center_text: str):
    nonzero = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
    if not nonzero:
        ax.text(0, 0, "no data", ha="center", va="center", fontsize=11)
        return None
    vals, lbls, cols = zip(*nonzero)

    wedges, _, autotexts = ax.pie(
        vals,
        labels=None,
        colors=cols,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.text(0, 0, center_text, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333333")
    return wedges, lbls, vals


# ---------------------------------------------------------------------------
# Plot 5 — energy pancake by hardware
# ---------------------------------------------------------------------------

def plot_pancake_energy_hardware(step_df: pd.DataFrame, out_path: Path,
                                 meta: dict) -> None:
    m = step_df["energy_consumed_per_step"].notna()
    dm = step_df[m]
    values, labels, colors = [], [], []
    for hw in HW_ORDER:
        col = f"{hw}_energy_per_step"
        if col not in dm.columns:
            continue
        values.append(float(dm[col].mean()) * KWH_TO_MWH)
        labels.append(HW_LABELS[hw])
        colors.append(HW_COLORS[hw])
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(
        f"PNA Carbon — Avg Energy Share by Hardware\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    result = _draw_pancake(ax, values, labels, colors,
                           f"mean step\n{total:.3f} mWh")
    if result:
        wedges, lbls, vals = result
        ax.legend(wedges, [f"{l}  ({v:.3f} mWh)" for l, v in zip(lbls, vals)],
                  loc="lower center", bbox_to_anchor=(0.5, -0.08),
                  ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  5. pancake_energy_hardware  -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 6 — carbon pancake by hardware
# ---------------------------------------------------------------------------

def plot_pancake_carbon_hardware(step_df: pd.DataFrame, out_path: Path,
                                 meta: dict) -> None:
    m = step_df["energy_consumed_per_step"].notna()
    dm = step_df[m]
    total_energy = dm["energy_consumed_per_step"].to_numpy()
    total_emit   = dm["emissions_per_step"].fillna(0.0).to_numpy()
    safe_energy  = np.where(total_energy > 0, total_energy, np.nan)

    values, labels, colors = [], [], []
    for hw in HW_ORDER:
        col = f"{hw}_energy_per_step"
        if col not in dm.columns:
            continue
        hw_energy = dm[col].fillna(0.0).to_numpy()
        hw_emit_ug = float(np.nanmean(
            np.where(np.isfinite(safe_energy),
                     total_emit * (hw_energy / safe_energy), 0.0)
        )) * KG_TO_UG
        values.append(hw_emit_ug)
        labels.append(HW_LABELS[hw])
        colors.append(HW_COLORS[hw])
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(
        f"PNA Carbon — Avg CO₂ Share by Hardware\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    result = _draw_pancake(ax, values, labels, colors,
                           f"mean step\n{total:.2f} µg")
    if result:
        wedges, lbls, vals = result
        ax.legend(wedges, [f"{l}  ({v:.2f} µg)" for l, v in zip(lbls, vals)],
                  loc="lower center", bbox_to_anchor=(0.5, -0.08),
                  ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  6. pancake_carbon_hardware  -> {out_path}")


# ---------------------------------------------------------------------------
# Combined plot: pancake (left, 2 rows) + energy hardware + carbon hardware
# ---------------------------------------------------------------------------

def plot_combined_energy_carbon(step_df: pd.DataFrame, out_path: Path,
                                meta: dict) -> None:
    """Layout: left = energy pancake spanning 2 rows, right-top = energy hardware,
    right-bottom = carbon hardware."""
    import matplotlib.gridspec as gridspec

    m = step_df["energy_consumed_per_step"].notna()
    dm = step_df[m].copy()
    if dm.empty:
        print("  combined -> [skip] no measured rows")
        return
    dm = dm[:100]
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1.2], hspace=0.4, wspace=0.05)

    ax_energy  = fig.add_subplot(gs[0, 0])   # left top
    ax_carbon  = fig.add_subplot(gs[1, 0])   # left bottom
    ax_pancake = fig.add_subplot(gs[:, 1])   # right, spans both rows

    # --- Left: energy pancake ---
    values, labels, colors = [], [], []
    for hw in HW_ORDER:
        col = f"{hw}_energy_per_step"
        if col not in dm.columns:
            continue
        values.append(float(dm[col].mean()) * KWH_TO_MWH)
        labels.append(HW_LABELS[hw])
        colors.append(HW_COLORS[hw])
    total = sum(values)

    result = _draw_pancake(ax_pancake, values, labels, colors,
                           f"mean step\n{total:.3f} mWh")
    ax_pancake.set_title("Energy Share", fontsize=15)

    # Collect legend handles from pancake
    hw_handles = []
    if result:
        wedges, lbls, vals = result
        hw_handles = list(wedges)

    # --- Right top: energy by hardware ---
    x = dm["step_idx"].to_numpy()
    bottoms = np.zeros(len(dm))
    for hw in HW_ORDER:
        col = f"{hw}_energy_per_step"
        if col not in dm.columns:
            continue
        vals_hw = dm[col].fillna(0.0).to_numpy() * KWH_TO_MWH
        ax_energy.fill_between(x, bottoms, bottoms + vals_hw,
                               alpha=0.80, color=HW_COLORS[hw], step="mid")
        bottoms += vals_hw
    ax_energy.set_ylabel("Energy (mWh)", fontsize=14)
    ax_energy.set_title("Per-Step Energy by Hardware", fontsize=15)
    ax_energy.tick_params(axis="both", labelsize=13)
    ax_energy.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_energy.set_axisbelow(True)

    # --- Right bottom: carbon by hardware ---
    total_energy = dm["energy_consumed_per_step"].to_numpy()
    total_emit   = dm["emissions_per_step"].fillna(0.0).to_numpy()
    safe_energy  = np.where(total_energy > 0, total_energy, np.nan)

    bottoms = np.zeros(len(dm))
    for hw in HW_ORDER:
        col = f"{hw}_energy_per_step"
        if col not in dm.columns:
            continue
        hw_energy = dm[col].fillna(0.0).to_numpy()
        hw_emit = np.where(
            np.isfinite(safe_energy),
            total_emit * (hw_energy / safe_energy),
            0.0,
        ) * KG_TO_UG
        ax_carbon.fill_between(x, bottoms, bottoms + hw_emit,
                               alpha=0.80, color=HW_COLORS[hw], step="mid")
        bottoms += hw_emit
    ax_carbon.set_xlabel("Step", fontsize=14)
    ax_carbon.set_ylabel("CO₂ (µg CO₂eq)", fontsize=14)
    ax_carbon.set_title("Per-Step CO₂ by Hardware", fontsize=15)
    ax_carbon.tick_params(axis="both", labelsize=13)
    ax_carbon.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_carbon.set_axisbelow(True)

    # --- Global legend at top ---
    hw_labels = [HW_LABELS[hw] for hw in HW_ORDER if f"{hw}_energy_per_step" in dm.columns]
    if hw_handles:
        fig.legend(hw_handles, hw_labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.05), ncol=len(hw_labels),
                   fontsize=13, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  combined             -> {out_path}")


# ---------------------------------------------------------------------------
# Discovery + orchestration
# ---------------------------------------------------------------------------

def discover_step_csvs(directory: Path) -> List[Path]:
    """Find pna_carbon_bs*_wk2_steps.csv files in carbon/ subdir."""
    carbon_dir = directory / "carbon"
    if not carbon_dir.is_dir():
        return []
    return sorted(carbon_dir.glob("pna_carbon_bs*_wk2_steps.csv"))


def process_file(step_csv: Path) -> None:
    print(f"\nProcessing: {step_csv.name}")

    df_raw = pd.read_csv(step_csv)
    required = {"step_idx", "step_ms"}
    missing = required - set(df_raw.columns)
    if missing:
        print(f"  [skip] missing columns: {missing}")
        return

    step_df = _enrich_step_df(df_raw)
    if step_df.empty:
        print("  [skip] no parseable step rows")
        return

    n_measured = step_df["energy_consumed"].notna().sum()
    print(f"  {len(step_df)} step rows, {n_measured} measurement windows")

    meta = _parse_meta(step_csv.stem)
    out  = _out_dir(step_csv)
    stem = re.sub(r"_steps$", "", step_csv.stem)

    plot_energy_total(   step_df, out / f"{stem}_energy_total.png",          meta)
    plot_energy_hardware(step_df, out / f"{stem}_energy_hardware.png",       meta)
    plot_carbon_total(   step_df, out / f"{stem}_carbon_total.png",          meta)
    plot_carbon_hardware(step_df, out / f"{stem}_carbon_hardware.png",       meta)
    plot_pancake_energy_hardware(step_df, out / f"{stem}_pancake_energy_hardware.png", meta)
    plot_pancake_carbon_hardware(step_df, out / f"{stem}_pancake_carbon_hardware.png", meta)

    # For bs4096, also generate plots using only the first 100 steps in a separate dir
    if meta.get("batch_size") == "4096" and len(step_df) > 100:
        out_100 = step_csv.parent / "plots" / "bs4096_first100"
        out_100.mkdir(parents=True, exist_ok=True)
        df_100 = _enrich_step_df(df_raw.head(100))
        meta_100 = dict(meta, batch_size="4096 (first 100 steps)")
        plot_energy_total(   df_100, out_100 / f"{stem}_energy_total.png",          meta_100)
        plot_energy_hardware(df_100, out_100 / f"{stem}_energy_hardware.png",       meta_100)
        plot_carbon_total(   df_100, out_100 / f"{stem}_carbon_total.png",          meta_100)
        plot_carbon_hardware(df_100, out_100 / f"{stem}_carbon_hardware.png",       meta_100)
        plot_pancake_energy_hardware(df_100, out_100 / f"{stem}_pancake_energy_hardware.png", meta_100)
        plot_pancake_carbon_hardware(df_100, out_100 / f"{stem}_pancake_carbon_hardware.png", meta_100)
        plot_combined_energy_carbon(df_100, out_100 / f"{stem}_combined.png", meta_100)
        print(f"  Also generated first-100 plots in {out_100}")

    # Combined plot for all batch sizes
    plot_combined_energy_carbon(step_df, out / f"{stem}_combined.png", meta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PNACarbonStats CSV output."
    )
    parser.add_argument(
        "--step",
        nargs="*",
        metavar="CSV",
        help=(
            "Path(s) to pna_carbon_bs*_wk*_steps.csv file(s). "
            "If omitted, auto-discovers all matching files in carbon/."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    if args.step:
        csv_files = [Path(p).resolve() for p in args.step]
    else:
        csv_files = discover_step_csvs(script_dir)
        if not csv_files:
            print(
                f"No pna_carbon_bs*_wk2_steps.csv found in {script_dir / 'carbon'}.\n"
                "Run start-pna-carbon.sh first."
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
