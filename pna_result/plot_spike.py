#!/usr/bin/env python3
"""
plot_spike.py — visualise GC spike analysis output from PNATrainerSpikeStats.

Reads spike/pna_spike_bs*_gc_on.csv and spike/pna_spike_bs*_gc_off.csv (one
file per run) and the corresponding *_gc_events.csv (gen-2 GC events recorded
during gc_on).

Output
------
Plots are saved to  pna_result/spike/plots/

Three plots are produced:
  1. {stem}_breakdown_gc_on.png           — stacked breakdown (gc_on only)
  2. {stem}_breakdown_gc_on_annotated.png — stacked breakdown (gc_on) + GC gen-2 vlines
  3. {stem}_breakdown_gc_off.png          — stacked breakdown (gc_off), flat baseline
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared style constants (same as plot_simple.py)
# ---------------------------------------------------------------------------

SUBSTEP_COLS   = ["forward_ms", "backward_ms", "optimizer_ms"]
SUBSTEP_LABELS = ["Forward", "Backward", "Optimizer"]
SUBSTEP_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
STEP_COLOR     = "#888888"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_gc_on_csvs(directory: Path):
    """Find all pna_spike_bs*_wk2_gc_on.csv files in spike/ subdir."""
    spike_dir = directory / "spike"
    if not spike_dir.is_dir():
        return []
    return sorted(spike_dir.glob("pna_spike_bs*_wk0_gc_on.csv"))


def find_gc_off_csv(gc_on_path: Path) -> Optional[Path]:
    p = gc_on_path.parent / gc_on_path.name.replace("_gc_on.csv", "_gc_off.csv")
    return p if p.exists() else None


def find_gc_events_csv(gc_on_path: Path) -> Optional[Path]:
    p = gc_on_path.parent / gc_on_path.name.replace("_gc_on.csv", "_gc_events.csv")
    return p if p.exists() else None


def _parse_meta(stem: str) -> dict:
    import re
    meta = {"batch_size": "?"}
    m = re.search(r"bs(\d+)", stem)
    if m:
        meta["batch_size"] = m.group(1)
    return meta


def _subtitle(meta: dict) -> str:
    return f"batch size {meta['batch_size']}"


# ---------------------------------------------------------------------------
# Plot 1 — Breakdown (gc_on only)
# ---------------------------------------------------------------------------

def plot_breakdown_gc_on(df_gc_on: pd.DataFrame, out_path: Path,
                         meta: dict, y_max: float) -> None:
    steps   = df_gc_on["step_idx"].to_numpy()
    bottoms = np.zeros(len(df_gc_on))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.suptitle(
        f"Spike Analysis: Execution Time Breakdown",
        fontsize=15, fontweight="bold",
    )

    for col, label, color in zip(SUBSTEP_COLS, SUBSTEP_LABELS, SUBSTEP_COLORS):
        vals = df_gc_on[col].to_numpy()
        ax.fill_between(steps, bottoms, bottoms + vals,
                        alpha=0.80, color=color, label=label)
        bottoms += vals

    ax.plot(steps, df_gc_on["step_ms"].to_numpy(), color=STEP_COLOR,
            linewidth=0.7, alpha=0.45, label="Total step (raw)")

    ax.set_ylim(0, y_max * 1.4)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel("Time (ms)", fontsize=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  1. breakdown_gc_on           -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Breakdown (gc_on) annotated with GC gen-2 vertical lines
# ---------------------------------------------------------------------------

def plot_breakdown_gc_on_annotated(df_gc_on: pd.DataFrame, gc_df: pd.DataFrame,
                                   out_path: Path, meta: dict, y_max: float) -> None:
    steps   = df_gc_on["step_idx"].to_numpy()
    bottoms = np.zeros(len(df_gc_on))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.suptitle(
        f"Spike Analysis: Breakdown with GC Gen-2 Annotated",
        fontsize=15, fontweight="bold",
    )

    for col, label, color in zip(SUBSTEP_COLS, SUBSTEP_LABELS, SUBSTEP_COLORS):
        vals = df_gc_on[col].to_numpy()
        ax.fill_between(steps, bottoms, bottoms + vals,
                        alpha=0.80, color=color, label=label)
        bottoms += vals

    ax.plot(steps, df_gc_on["step_ms"].to_numpy(), color=STEP_COLOR,
            linewidth=0.7, alpha=0.45, label="Total step (raw)")

    # Red dashed vertical lines at GC event step positions
    gc_steps = gc_df["step_idx"].tolist()
    for step in gc_steps:
        ax.axvline(step, color="#cc0000", linewidth=1.2,
                   linestyle="--", alpha=0.75, zorder=5)

    ax.set_ylim(0, y_max * 1.4)

    # Build legend with GC handle
    substep_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.80)
        for c in SUBSTEP_COLORS
    ]
    step_handle = mlines.Line2D([], [], color=STEP_COLOR, linewidth=0.7,
                                alpha=0.45, label="Total step (raw)")
    gc_handle = mlines.Line2D(
        [], [], color="#cc0000", linewidth=1.2, linestyle="--",
        label="GC gen-2 (long-lived\nobject sweep)",
    )
    ax.legend(
        handles=substep_handles + [step_handle, gc_handle],
        labels=SUBSTEP_LABELS + ["Total step (raw)", "GC gen-2 (long-lived\nobject sweep)"],
        fontsize=9, loc="upper right",
    )

    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel("Time (ms)", fontsize=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  2. breakdown_gc_on_annotated -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Breakdown (gc_off), flat baseline
# ---------------------------------------------------------------------------

def plot_breakdown_gc_off(df_gc_off: pd.DataFrame, out_path: Path,
                          meta: dict, y_max: float) -> None:
    steps   = df_gc_off["step_idx"].to_numpy()
    bottoms = np.zeros(len(df_gc_off))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.suptitle(
        f"Spike Analysis (GC off)",
        fontsize=15, fontweight="bold",
    )

    for col, label, color in zip(SUBSTEP_COLS, SUBSTEP_LABELS, SUBSTEP_COLORS):
        vals = df_gc_off[col].to_numpy()
        ax.fill_between(steps, bottoms, bottoms + vals,
                        alpha=0.80, color=color, label=label)
        bottoms += vals

    ax.plot(steps, df_gc_off["step_ms"].to_numpy(), color=STEP_COLOR,
            linewidth=0.7, alpha=0.45, label="Total step (raw)")

    ax.set_ylim(0, y_max * 1.4)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel("Time (ms)", fontsize=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  3. breakdown_gc_off          -> {out_path}")


# ---------------------------------------------------------------------------
# Combined 3-panel figure (gc_on / annotated / gc_off)
# ---------------------------------------------------------------------------

def plot_combined(df_gc_on: pd.DataFrame, gc_df: pd.DataFrame,
                  df_gc_off: pd.DataFrame, out_path: Path,
                  meta: dict, y_max: float) -> None:
    """Three vertically stacked subplots with a single shared legend at the top."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ylim = y_max * 1.4

    # Stacking order for combined plot: optimizer (bottom), backward, forward (top)
    _COMB_COLS   = ["backward_ms", "forward_ms", "optimizer_ms"]
    _COMB_LABELS = [ "Backward", "Forward", "Optimizer"]
    _COMB_COLORS = ["#DD8452", "#4C72B0", "#55A868"]

    # --- Panel 1: GC on breakdown ---
    steps_on = df_gc_on["step_idx"].to_numpy()
    bottoms = np.zeros(len(df_gc_on))
    fill_handles = []
    for col, label, color in zip(_COMB_COLS, _COMB_LABELS, _COMB_COLORS):
        vals = df_gc_on[col].to_numpy()
        h = ax1.fill_between(steps_on, bottoms, bottoms + vals,
                             alpha=0.80, color=color)
        fill_handles.append(h)
        bottoms += vals
    step_line, = ax1.plot(steps_on, df_gc_on["step_ms"].to_numpy(),
                          color=STEP_COLOR, linewidth=0.7, alpha=0.45)
    ax1.set_ylim(0, ylim)
    ax1.set_ylabel("Time (ms)", fontsize=14)
    ax1.set_title("Execution Time Breakdown (Automatic GC on)", fontsize=15)
    ax1.tick_params(axis="both", labelsize=13)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    # --- Panel 2: GC on annotated ---
    bottoms = np.zeros(len(df_gc_on))
    for col, color in zip(_COMB_COLS, _COMB_COLORS):
        vals = df_gc_on[col].to_numpy()
        ax2.fill_between(steps_on, bottoms, bottoms + vals,
                         alpha=0.80, color=color)
        bottoms += vals
    ax2.plot(steps_on, df_gc_on["step_ms"].to_numpy(),
             color=STEP_COLOR, linewidth=0.7, alpha=0.45)
    gc_line = None
    if not gc_df.empty:
        gc_steps = gc_df["step_idx"].tolist()
        for step in gc_steps:
            ax2.axvline(step, color="#cc0000", linewidth=1.2,
                        linestyle="--", alpha=0.75, zorder=5)
        gc_line = mlines.Line2D([], [], color="#cc0000", linewidth=1.2,
                                linestyle="--")
    ax2.set_ylim(0, ylim)
    ax2.set_ylabel("Time (ms)", fontsize=14)
    ax2.set_title("Breakdown with GC Gen-2 Annotated", fontsize=15)
    ax2.tick_params(axis="both", labelsize=13)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)

    # --- Panel 3: GC off ---
    if not df_gc_off.empty:
        steps_off = df_gc_off["step_idx"].to_numpy()
        bottoms = np.zeros(len(df_gc_off))
        for col, color in zip(_COMB_COLS, _COMB_COLORS):
            vals = df_gc_off[col].to_numpy()
            ax3.fill_between(steps_off, bottoms, bottoms + vals,
                             alpha=0.80, color=color)
            bottoms += vals
        ax3.plot(steps_off, df_gc_off["step_ms"].to_numpy(),
                 color=STEP_COLOR, linewidth=0.7, alpha=0.45)
    ax3.set_ylim(0, ylim)
    ax3.set_xlabel("Step", fontsize=14)
    ax3.set_ylabel("Time (ms)", fontsize=14)
    ax3.set_title("Execution Time Breakdown (GC off)", fontsize=15)
    ax3.tick_params(axis="both", labelsize=13)
    ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax3.set_axisbelow(True)

    # --- Global legend at top ---
    handles = list(fill_handles)
    labels  = list(_COMB_LABELS)
    if gc_line is not None:
        handles.append(gc_line)
        labels.append("GC-gen-2 event")
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=len(labels),
               fontsize=13, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  4. combined          -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_gc_on(gc_on_path: Path) -> None:
    print(f"\nProcessing: {gc_on_path.name}")

    gc_off_path    = find_gc_off_csv(gc_on_path)
    gc_events_path = find_gc_events_csv(gc_on_path)

    df_gc_on  = pd.read_csv(gc_on_path)
    df_gc_off = pd.read_csv(gc_off_path) if gc_off_path else pd.DataFrame()
    gc_df     = pd.read_csv(gc_events_path) if gc_events_path else pd.DataFrame()

    # Compute shared y_max across both runs for comparable scales
    all_step_ms = list(df_gc_on["step_ms"])
    if not df_gc_off.empty:
        all_step_ms += list(df_gc_off["step_ms"])
    y_max = float(np.max(all_step_ms)) if all_step_ms else 1.0

    out_dir = gc_on_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use gc_on stem but strip the _gc_on suffix for output naming
    base_stem = gc_on_path.stem.replace("_gc_on", "")
    meta = _parse_meta(base_stem)

    plot_breakdown_gc_on(df_gc_on, out_dir / f"{base_stem}_breakdown_gc_on.png", meta, y_max)

    if not gc_df.empty:
        plot_breakdown_gc_on_annotated(df_gc_on, gc_df, out_dir / f"{base_stem}_breakdown_gc_on_annotated.png", meta, y_max)
    else:
        print("  [skip] no GC events CSV — skipping annotated plot")

    if not df_gc_off.empty:
        plot_breakdown_gc_off(df_gc_off, out_dir / f"{base_stem}_breakdown_gc_off.png", meta, y_max)
    else:
        print("  [skip] no gc_off CSV found — skipping gc_off plot")

    # Combined 3-panel figure
    plot_combined(df_gc_on, gc_df, df_gc_off, out_dir / f"{base_stem}_combined.png", meta, y_max)


def main() -> None:
    script_dir = Path(__file__).parent.resolve()

    gc_on_paths = discover_gc_on_csvs(script_dir)
    if not gc_on_paths:
        print("No pna_spike_bs*_wk2_gc_on.csv found in spike/. Run start-pna-spike.sh first.")
        sys.exit(0)

    print(f"Auto-discovered {len(gc_on_paths)} gc_on CSV(s)")
    for gc_on_path in gc_on_paths:
        process_gc_on(gc_on_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
