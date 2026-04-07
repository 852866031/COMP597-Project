#!/usr/bin/env python3
"""
plot_carbon.py — visualise outputs produced by PNACarbonStats.

Usage
-----
# Auto-discover carbon/pna_carbon_bs*_step-steps.csv in the same directory:
    python pna_result/plot_carbon.py

# Point at a specific step CSV explicitly:
    python pna_result/plot_carbon.py --step pna_result/carbon/pna_carbon_bs256_step-steps.csv

Output
------
Plots are saved to  pna_result/carbon/plots/

Six plots are produced per batch-size:
  1. energy_total.png          — total energy (mWh) per step
  2. energy_substep.png        — stacked area: forward / backward / optimizer energy per step
  3. energy_hardware.png       — stacked area: CPU / GPU / RAM energy per step
  4. carbon_total.png          — total CO₂ emissions (µg CO₂eq) per step
  5. carbon_substep.png        — stacked area: forward / backward / optimizer emissions per step
  6. carbon_hardware.png       — stacked area: CPU / GPU / RAM derived emissions per step

CSV files consumed
------------------
  carbon/pna_carbon_bs<N>_step-steps.csv
      One row per training step; columns include:
        task_name, duration, energy_consumed, cpu_energy, gpu_energy,
        ram_energy, emissions  (kWh / kg CO₂eq)

  carbon/pna_carbon_bs<N>_substep-substeps.csv
      One row per substep call; same columns; task_name encodes phase.
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

# Substep (phase) colours — consistent with plot_utils.py
PHASE_COLORS = {
    "fwd": "#4C72B0",   # blue
    "bwd": "#DD8452",   # orange
    "opt": "#55A868",   # green
}
PHASE_LABELS = {
    "fwd": "Forward",
    "bwd": "Backward",
    "opt": "Optimizer",
}
PHASE_ORDER = ["fwd", "bwd", "opt"]

# Hardware colours
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

# Unit conversions
KWH_TO_MWH  = 1_000.0          # kWh → mWh
KG_TO_UG    = 1_000_000_000.0  # kg  → µg CO₂eq


# ---------------------------------------------------------------------------
# Task-name parsing helpers
# ---------------------------------------------------------------------------

def _parse_step_task(task_name: str) -> Tuple[Optional[int], Optional[int]]:
    """'e{epoch}_step_{idx}' → (epoch, step_idx) or (None, None)."""
    m = re.match(r"e(\d+)_step_(\d+)", str(task_name))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _parse_substep_task(task_name: str) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    """'e{epoch}_{fwd|bwd|opt}_{idx}' → (epoch, phase, step_idx) or (None,None,None)."""
    m = re.match(r"e(\d+)_(fwd|bwd|opt)_(\d+)", str(task_name))
    if not m:
        return None, None, None
    return int(m.group(1)), m.group(2), int(m.group(3))


def _enrich_step_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse task_name; add epoch, step_idx, global_step columns."""
    parsed = df["task_name"].apply(
        lambda t: pd.Series(_parse_step_task(t), index=["epoch", "step_idx"])
    )
    df = df.copy()
    df["epoch"]    = parsed["epoch"].astype("Int64")
    df["step_idx"] = parsed["step_idx"].astype("Int64")
    df = df.dropna(subset=["step_idx"]).sort_values("step_idx").reset_index(drop=True)
    df["global_step"] = np.arange(1, len(df) + 1)
    return df


def _enrich_substep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse task_name; add epoch, phase, step_idx columns."""
    parsed = df["task_name"].apply(
        lambda t: pd.Series(_parse_substep_task(t), index=["epoch", "phase", "step_idx"])
    )
    df = df.copy()
    df["epoch"]    = parsed["epoch"].astype("Int64")
    df["phase"]    = parsed["phase"]
    df["step_idx"] = parsed["step_idx"].astype("Int64")
    df = df.dropna(subset=["step_idx", "phase"]).reset_index(drop=True)
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
    """Return global_step values where the epoch number changes (skip first)."""
    if "epoch" not in df.columns or "global_step" not in df.columns:
        return []
    changed = df["epoch"].diff().fillna(0) != 0
    changed.iloc[0] = False
    return df.loc[changed, "global_step"].tolist()


def _draw_epoch_vlines(ax, epoch_starts: List[int]) -> None:
    first = True
    for s in epoch_starts:
        ax.axvline(x=s, color=EPOCH_LINE_COLOR, linestyle=":",
                   linewidth=1.2, alpha=0.85,
                   label="Epoch start" if first else None)
        first = False


# ---------------------------------------------------------------------------
# Plot 1 — total energy per step
# ---------------------------------------------------------------------------

def plot_energy_total(step_df: pd.DataFrame, out_path: Path, meta: dict,
                      epoch_starts: List[int]) -> None:
    x   = step_df["global_step"].to_numpy()
    y   = step_df["energy_consumed"].to_numpy() * KWH_TO_MWH  # → mWh

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — Total Energy per Step\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.bar(x, y, width=1.0, color="#457B9D", alpha=0.80)
    _draw_epoch_vlines(ax, epoch_starts)
    if epoch_starts:
        ax.legend(fontsize=9, loc="upper right")

    ax.set_xlabel("Step")
    ax.set_ylabel("Energy (mWh)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  1. energy_total     -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — energy breakdown by substep (forward / backward / optimizer)
# ---------------------------------------------------------------------------

def plot_energy_substep(step_df: pd.DataFrame, substep_df: pd.DataFrame,
                        out_path: Path, meta: dict,
                        epoch_starts: List[int]) -> None:
    """Stacked area of per-phase energy aligned to global step order."""

    # Pivot substep data: one row per step_idx, columns = phases
    pivot = (
        substep_df
        .pivot_table(index="step_idx", columns="phase",
                     values="energy_consumed", aggfunc="sum")
        .reindex(columns=PHASE_ORDER)
        .fillna(0.0)
    )

    # Align to step_df order via step_idx
    merged = step_df[["global_step", "step_idx"]].merge(
        pivot.reset_index(), on="step_idx", how="left"
    ).sort_values("global_step").reset_index(drop=True)

    x = merged["global_step"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — Energy Breakdown by Substep\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    bottoms = np.zeros(len(merged))
    for phase in PHASE_ORDER:
        if phase not in merged.columns:
            continue
        vals = merged[phase].to_numpy() * KWH_TO_MWH  # mWh
        ax.fill_between(x, bottoms, bottoms + vals,
                        alpha=0.80, color=PHASE_COLORS[phase],
                        label=PHASE_LABELS[phase])
        bottoms += vals

    _draw_epoch_vlines(ax, epoch_starts)
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy (mWh)")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  2. energy_substep   -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — energy breakdown by hardware (CPU / GPU / RAM)
# ---------------------------------------------------------------------------

def plot_energy_hardware(step_df: pd.DataFrame, out_path: Path, meta: dict,
                         epoch_starts: List[int]) -> None:
    x = step_df["global_step"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — Energy Breakdown by Hardware\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    bottoms = np.zeros(len(step_df))
    for hw in HW_ORDER:
        col = f"{hw}_energy"
        if col not in step_df.columns:
            continue
        vals = step_df[col].to_numpy() * KWH_TO_MWH  # mWh
        ax.fill_between(x, bottoms, bottoms + vals,
                        alpha=0.80, color=HW_COLORS[hw],
                        label=HW_LABELS[hw])
        bottoms += vals

    _draw_epoch_vlines(ax, epoch_starts)
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy (mWh)")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  3. energy_hardware  -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — total CO₂ emissions per step
# ---------------------------------------------------------------------------

def plot_carbon_total(step_df: pd.DataFrame, out_path: Path, meta: dict,
                      epoch_starts: List[int]) -> None:
    x = step_df["global_step"].to_numpy()
    y = step_df["emissions"].to_numpy() * KG_TO_UG  # → µg CO₂eq

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — CO₂ Emissions per Step\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    ax.bar(x, y, width=1.0, color="#E63946", alpha=0.80)
    _draw_epoch_vlines(ax, epoch_starts)
    if epoch_starts:
        ax.legend(fontsize=9, loc="upper right")

    ax.set_xlabel("Step")
    ax.set_ylabel("CO₂ Emissions (µg CO₂eq)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  4. carbon_total     -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 5 — CO₂ emissions breakdown by substep
# ---------------------------------------------------------------------------

def plot_carbon_substep(step_df: pd.DataFrame, substep_df: pd.DataFrame,
                        out_path: Path, meta: dict,
                        epoch_starts: List[int]) -> None:
    pivot = (
        substep_df
        .pivot_table(index="step_idx", columns="phase",
                     values="emissions", aggfunc="sum")
        .reindex(columns=PHASE_ORDER)
        .fillna(0.0)
    )

    merged = step_df[["global_step", "step_idx"]].merge(
        pivot.reset_index(), on="step_idx", how="left"
    ).sort_values("global_step").reset_index(drop=True)

    x = merged["global_step"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — CO₂ Emissions Breakdown by Substep\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    bottoms = np.zeros(len(merged))
    for phase in PHASE_ORDER:
        if phase not in merged.columns:
            continue
        vals = merged[phase].to_numpy() * KG_TO_UG  # µg CO₂eq
        ax.fill_between(x, bottoms, bottoms + vals,
                        alpha=0.80, color=PHASE_COLORS[phase],
                        label=PHASE_LABELS[phase])
        bottoms += vals

    _draw_epoch_vlines(ax, epoch_starts)
    ax.set_xlabel("Step")
    ax.set_ylabel("CO₂ Emissions (µg CO₂eq)")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  5. carbon_substep   -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 6 — CO₂ emissions breakdown by hardware
# ---------------------------------------------------------------------------

def plot_carbon_hardware(step_df: pd.DataFrame, out_path: Path, meta: dict,
                         epoch_starts: List[int]) -> None:
    """Derive per-hardware emissions by apportioning total emissions by energy share."""
    x = step_df["global_step"].to_numpy()
    total_energy = step_df["energy_consumed"].to_numpy()
    total_emit   = step_df["emissions"].to_numpy()

    # Avoid division by zero for steps where energy is 0
    safe_energy = np.where(total_energy > 0, total_energy, np.nan)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"PNA Carbon — CO₂ Emissions Breakdown by Hardware\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    bottoms = np.zeros(len(step_df))
    for hw in HW_ORDER:
        col = f"{hw}_energy"
        if col not in step_df.columns:
            continue
        hw_energy = step_df[col].to_numpy()
        # emissions_hw = emissions_total × (energy_hw / energy_total)
        hw_emit = np.where(
            np.isfinite(safe_energy),
            total_emit * (hw_energy / safe_energy),
            0.0,
        ) * KG_TO_UG  # µg CO₂eq
        ax.fill_between(x, bottoms, bottoms + hw_emit,
                        alpha=0.80, color=HW_COLORS[hw],
                        label=HW_LABELS[hw])
        bottoms += hw_emit

    _draw_epoch_vlines(ax, epoch_starts)
    ax.set_xlabel("Step")
    ax.set_ylabel("CO₂ Emissions (µg CO₂eq)")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  6. carbon_hardware  -> {out_path}")


# ---------------------------------------------------------------------------
# Pancake helpers
# ---------------------------------------------------------------------------

def _draw_pancake(ax, values, labels, colors, center_text: str) -> None:
    """Draw a donut (pancake) chart on *ax*. Filters out zero-value slices."""
    nonzero = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
    if not nonzero:
        ax.text(0, 0, "no data", ha="center", va="center", fontsize=11)
        return
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
# Pancake plot 7 — energy pancake by substep
# ---------------------------------------------------------------------------

def plot_pancake_energy_substep(substep_df: pd.DataFrame, out_path: Path,
                                meta: dict) -> None:
    means = {
        phase: substep_df.loc[substep_df["phase"] == phase, "energy_consumed"].mean()
        for phase in PHASE_ORDER
    }
    values = [means.get(p, 0.0) * KWH_TO_MWH for p in PHASE_ORDER]
    labels = [PHASE_LABELS[p] for p in PHASE_ORDER]
    colors = [PHASE_COLORS[p] for p in PHASE_ORDER]
    total  = sum(values)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(
        f"PNA Carbon — Avg Energy Share by Substep\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    result = _draw_pancake(ax, values, labels, colors,
                           f"mean step\n{total:.3f} mWh")
    if result:
        wedges, lbls, vals = result
        legend_labels = [f"{l}  ({v:.3f} mWh)" for l, v in zip(lbls, vals)]
        ax.legend(wedges, legend_labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  7. pancake_energy_substep   -> {out_path}")


# ---------------------------------------------------------------------------
# Pancake plot 8 — energy pancake by hardware
# ---------------------------------------------------------------------------

def plot_pancake_energy_hardware(step_df: pd.DataFrame, out_path: Path,
                                 meta: dict) -> None:
    values, labels, colors = [], [], []
    for hw in HW_ORDER:
        col = f"{hw}_energy"
        if col not in step_df.columns:
            continue
        values.append(float(step_df[col].mean()) * KWH_TO_MWH)
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
        legend_labels = [f"{l}  ({v:.3f} mWh)" for l, v in zip(lbls, vals)]
        ax.legend(wedges, legend_labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  8. pancake_energy_hardware  -> {out_path}")


# ---------------------------------------------------------------------------
# Pancake plot 9 — carbon pancake by substep
# ---------------------------------------------------------------------------

def plot_pancake_carbon_substep(substep_df: pd.DataFrame, out_path: Path,
                                meta: dict) -> None:
    means = {
        phase: substep_df.loc[substep_df["phase"] == phase, "emissions"].mean()
        for phase in PHASE_ORDER
    }
    values = [means.get(p, 0.0) * KG_TO_UG for p in PHASE_ORDER]
    labels = [PHASE_LABELS[p] for p in PHASE_ORDER]
    colors = [PHASE_COLORS[p] for p in PHASE_ORDER]
    total  = sum(values)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(
        f"PNA Carbon — Avg CO₂ Share by Substep\n{_subtitle(meta)}",
        fontsize=13, fontweight="bold",
    )

    result = _draw_pancake(ax, values, labels, colors,
                           f"mean step\n{total:.2f} µg")
    if result:
        wedges, lbls, vals = result
        legend_labels = [f"{l}  ({v:.2f} µg)" for l, v in zip(lbls, vals)]
        ax.legend(wedges, legend_labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  9. pancake_carbon_substep   -> {out_path}")


# ---------------------------------------------------------------------------
# Pancake plot 10 — carbon pancake by hardware
# ---------------------------------------------------------------------------

def plot_pancake_carbon_hardware(step_df: pd.DataFrame, out_path: Path,
                                 meta: dict) -> None:
    total_energy = step_df["energy_consumed"].to_numpy()
    total_emit   = step_df["emissions"].to_numpy()
    safe_energy  = np.where(total_energy > 0, total_energy, np.nan)

    values, labels, colors = [], [], []
    for hw in HW_ORDER:
        col = f"{hw}_energy"
        if col not in step_df.columns:
            continue
        hw_energy = step_df[col].to_numpy()
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
        legend_labels = [f"{l}  ({v:.2f} µg)" for l, v in zip(lbls, vals)]
        ax.legend(wedges, legend_labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" 10. pancake_carbon_hardware  -> {out_path}")


# ---------------------------------------------------------------------------
# Discovery + orchestration
# ---------------------------------------------------------------------------

def discover_step_csvs(directory: Path) -> List[Path]:
    """Find pna_carbon_bs*_step-steps.csv in carbon/ subdir."""
    carbon_dir = directory / "carbon"
    if not carbon_dir.is_dir():
        return []
    return sorted(carbon_dir.glob("pna_carbon_bs*_step-steps.csv"))


def _find_substep_csv(step_csv: Path) -> Optional[Path]:
    """pna_carbon_bs<N>_step-steps.csv → pna_carbon_bs<N>_substep-substeps.csv"""
    m = re.match(r"(pna_carbon_bs\d+)_step-steps\.csv", step_csv.name)
    if not m:
        return None
    candidate = step_csv.parent / f"{m.group(1)}_substep-substeps.csv"
    return candidate if candidate.exists() else None


def process_file(step_csv: Path) -> None:
    print(f"\nProcessing: {step_csv.name}")

    step_df_raw = pd.read_csv(step_csv)
    required = {"task_name", "energy_consumed", "emissions"}
    missing = required - set(step_df_raw.columns)
    if missing:
        print(f"  [skip] missing columns in step CSV: {missing}")
        return

    step_df = _enrich_step_df(step_df_raw)
    if step_df.empty:
        print("  [skip] no parseable step rows")
        return

    substep_csv = _find_substep_csv(step_csv)
    if substep_csv is None:
        print("  [warn] substep CSV not found — plots 2 and 5 will be skipped")
        substep_df = None
    else:
        substep_df_raw = pd.read_csv(substep_csv)
        substep_df = _enrich_substep_df(substep_df_raw)
        if substep_df.empty:
            print("  [warn] no parseable substep rows — plots 2 and 5 will be skipped")
            substep_df = None

    meta = _parse_meta(step_csv.stem)
    out  = _out_dir(step_csv)
    stem = re.sub(r"_step-steps$", "", step_csv.stem)

    epoch_starts = _epoch_starts(step_df)

    plot_energy_total(   step_df,             out / f"{stem}_energy_total.png",    meta, epoch_starts)
    if substep_df is not None:
        plot_energy_substep( step_df, substep_df, out / f"{stem}_energy_substep.png",  meta, epoch_starts)
    else:
        print("  2. energy_substep   -> [skip] no substep CSV")
    plot_energy_hardware(step_df,             out / f"{stem}_energy_hardware.png", meta, epoch_starts)
    plot_carbon_total(   step_df,             out / f"{stem}_carbon_total.png",    meta, epoch_starts)
    if substep_df is not None:
        plot_carbon_substep( step_df, substep_df, out / f"{stem}_carbon_substep.png",  meta, epoch_starts)
    else:
        print("  5. carbon_substep   -> [skip] no substep CSV")
    plot_carbon_hardware(step_df,             out / f"{stem}_carbon_hardware.png", meta, epoch_starts)

    # Pancake plots — mean share over all steps
    if substep_df is not None:
        plot_pancake_energy_substep( substep_df, out / f"{stem}_pancake_energy_substep.png",  meta)
        plot_pancake_carbon_substep( substep_df, out / f"{stem}_pancake_carbon_substep.png",  meta)
    else:
        print("  7. pancake_energy_substep   -> [skip] no substep CSV")
        print("  9. pancake_carbon_substep   -> [skip] no substep CSV")
    plot_pancake_energy_hardware(step_df, out / f"{stem}_pancake_energy_hardware.png", meta)
    plot_pancake_carbon_hardware(step_df, out / f"{stem}_pancake_carbon_hardware.png", meta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PNACarbonStats CSV output."
    )
    parser.add_argument(
        "--step",
        nargs="*",
        metavar="CSV",
        help=(
            "Path(s) to pna_carbon_bs*_step-steps.csv file(s). "
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
                f"No pna_carbon_bs*_step-steps.csv found in {script_dir / 'carbon'}.\n"
                "Run start-pna-carbon.sh first."
            )
            sys.exit(0)
        print(f"Auto-discovered {len(csv_files)} step CSV(s)")

    for path in csv_files:
        if not path.exists():
            print(f"[warn] file not found, skipping: {path}")
            continue
        process_file(path)

    print("\nDone.")


if __name__ == "__main__":
    main()
