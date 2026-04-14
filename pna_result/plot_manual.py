#!/usr/bin/env python3
"""
plot_manual.py — visualise GC-controlled (manual) training baseline.

One combined figure per batch size with two subplots:
  Left  (wide)  — Substep breakdown per step (stacked area)
  Right (slim)  — p90 overhead bar chart: Raw vs Manual-GC

Usage
-----
    python pna_result/plot_manual.py
"""
from __future__ import annotations

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
# Style
# ---------------------------------------------------------------------------

SUBSTEP_COLS   = ["forward_ms", "backward_ms", "optimizer_ms"]
SUBSTEP_LABELS = ["Forward", "Backward", "Optimizer"]
SUBSTEP_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
STEP_COLOR     = "#888888"

SIMPLE_COLOR   = "#E76F51"   # terracotta for raw (GC-on)
MANUAL_COLOR   = "#2A9D8F"   # teal for manual (GC-off)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bs(stem: str) -> str:
    m = re.search(r"bs(\d+)", stem)
    return m.group(1) if m else "?"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_manual(directory: Path) -> List[Path]:
    d = directory / "manual"
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.glob("pna_manual_gc_bs*_wk2.csv")
        if not p.name.endswith("_agg.csv")
    )


def _find_simple_match(manual_path: Path, directory: Path) -> Optional[Path]:
    """Find the matching simple CSV for a manual-gc file (same bs and wk)."""
    m = re.search(r"bs(\d+)_wk(\d+)", manual_path.name)
    if not m:
        return None
    bs, wk = m.group(1), m.group(2)
    simple_dir = directory / "simple"
    if not simple_dir.is_dir():
        return None
    matches = sorted(
        p for p in simple_dir.glob(f"pna_simple_bs{bs}_wk{wk}.csv")
        if not p.name.endswith("_agg.csv")
    )
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Standalone breakdown plot
# ---------------------------------------------------------------------------

def plot_breakdown(df: pd.DataFrame, out_path: Path, batch_size: str) -> None:
    steps   = df["step_idx"].to_numpy()
    bottoms = np.zeros(len(df))

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"Manual-GC Baseline — Execution Time Breakdown\n"
        f"batch size {batch_size}",
        fontsize=13, fontweight="bold",
    )

    for col, label, color in zip(SUBSTEP_COLS, SUBSTEP_LABELS, SUBSTEP_COLORS):
        vals = df[col].to_numpy()
        ax.fill_between(steps, bottoms, bottoms + vals,
                        alpha=0.80, color=color, label=label)
        bottoms += vals

    ax.plot(steps, df["step_ms"].to_numpy(), color=STEP_COLOR,
            linewidth=0.7, alpha=0.45, label="Total step (raw)")

    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlabel("Step")
    ax.set_ylabel("Time (ms)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  breakdown -> {out_path}")


# ---------------------------------------------------------------------------
# Combined plot: breakdown (left) + overhead bar (right)
# ---------------------------------------------------------------------------

def plot_combined(
    manual_df: pd.DataFrame,
    simple_df: Optional[pd.DataFrame],
    out_path: Path,
    batch_size: str,
) -> None:
    has_overhead = (simple_df is not None and "step_ms" in simple_df.columns)

    # Layout: breakdown gets ~75% width, overhead bar gets ~25%
    if has_overhead:
        fig, (ax_bd, ax_oh) = plt.subplots(
            1, 2, figsize=(12, 4),
            gridspec_kw={"width_ratios": [3, 1]},
        )
    else:
        fig, ax_bd = plt.subplots(figsize=(10, 4))
        ax_oh = None

    fig.suptitle(
        f"Manual-GC Baseline (batch size {batch_size})",
        fontsize=15, fontweight="bold", y=0.96,
    )

    # ---- Left: total step time ----
    steps   = manual_df["step_idx"].to_numpy()
    step_ms = manual_df["step_ms"].to_numpy()

    ax_bd.bar(steps, step_ms, width=1.0, color=STEP_COLOR, alpha=0.80)

    ax_bd.set_xlabel("Step", fontsize=14)
    ax_bd.set_ylabel("Time (ms)", fontsize=14)
    ax_bd.set_title("Per-Step Execution Time", fontsize=15)
    ax_bd.tick_params(axis="both", labelsize=13)
    ax_bd.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_bd.set_axisbelow(True)

    # ---- Right: overhead bar chart (p90) ----
    if has_overhead and ax_oh is not None:
        simple_p90 = float(simple_df["step_ms"].quantile(0.90))
        manual_p90 = float(manual_df["step_ms"].quantile(0.90))
        saving_pct = ((manual_p90 - simple_p90) / simple_p90 * 100.0
                      if simple_p90 > 0 else 0.0)

        vals   = [simple_p90, manual_p90]
        colors = [SIMPLE_COLOR, MANUAL_COLOR]

        bars = ax_oh.bar([0, 1], vals, width=0.55, color=colors,
                         alpha=0.85, zorder=3)

        y_max = max(vals)

        # Saving annotation on manual-gc bar (negative = faster)
        ax_oh.text(bars[1].get_x() + bars[1].get_width() / 2,
                   bars[1].get_height() + y_max * 0.01,
                   f"{saving_pct:.1f}%",
                   ha="center", va="bottom", fontsize=11,
                   color=MANUAL_COLOR, fontweight="bold")

        ax_oh.set_xticks([0, 1])
        ax_oh.set_xticklabels(["Raw", "Manual-GC"], fontsize=13)
        ax_oh.set_ylabel("p90 Step Latency (ms)", fontsize=14)
        ax_oh.set_title("GC Overhead", fontsize=15)
        ax_oh.tick_params(axis="both", labelsize=13)
        ax_oh.set_ylim(0, y_max * 1.22)
        ax_oh.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax_oh.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Pancake (donut) chart — average step time breakdown
# ---------------------------------------------------------------------------

def plot_pancake(manual_df: pd.DataFrame, out_path: Path,
                 batch_size: str) -> None:
    fwd_mean  = float(manual_df["forward_ms"].mean())
    bwd_mean  = float(manual_df["backward_ms"].mean())
    opt_mean  = float(manual_df["optimizer_ms"].mean())
    step_mean = float(manual_df["step_ms"].mean())
    other     = max(0.0, step_mean - fwd_mean - bwd_mean - opt_mean)

    values = [fwd_mean, bwd_mean, opt_mean, other]
    labels = ["Forward", "Backward", "Optimizer", "Other"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#888888"]

    # Drop zero slices
    nonzero = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0.05]
    if not nonzero:
        print(f"  pancake -> [skip] no data")
        return
    vals, lbls, cols = zip(*nonzero)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(
        f"PNA Manual-GC — Average Step Time Breakdown\n"
        f"batch size {batch_size}",
        fontsize=13, fontweight="bold",
    )

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

    ax.text(0, 0, f"mean step\n{step_mean:.1f} ms",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="#333333")

    ax.legend(
        wedges,
        [f"{l}  ({v:.1f} ms)" for l, v in zip(lbls, vals)],
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=2, fontsize=9, frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  pancake -> {out_path}")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_file(manual_path: Path, simple_path: Optional[Path],
                 out_dir: Path) -> None:
    bs = _parse_bs(manual_path.stem)
    stem = re.sub(r"_agg$", "", manual_path.stem)

    print(f"\nProcessing: {manual_path.name}  (bs={bs})")

    manual_df = pd.read_csv(manual_path)
    required = {"step_idx", "step_ms"}
    if required - set(manual_df.columns):
        print(f"  [skip] missing columns in manual CSV")
        return

    simple_df = None
    if simple_path and simple_path.exists():
        simple_df = pd.read_csv(simple_path)

    has_substeps = {"forward_ms", "backward_ms", "optimizer_ms"}.issubset(manual_df.columns)
    if has_substeps:
        plot_breakdown(manual_df, out_dir / f"{stem}_breakdown.png", bs)
        plot_pancake(manual_df, out_dir / f"{stem}_pancake.png", bs)
    plot_combined(manual_df, simple_df, out_dir / f"{stem}.png", bs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).parent.resolve()
    out_dir = script_dir / "manual" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_csvs = _discover_manual(script_dir)
    if not manual_csvs:
        print(
            f"No pna_manual_gc_bs*_wk2.csv found in {script_dir / 'manual'}.\n"
            "Run start-pna-manual-gc.sh first."
        )
        sys.exit(0)

    print(f"Auto-discovered {len(manual_csvs)} manual-gc CSV(s)")

    for manual_path in manual_csvs:
        simple_path = _find_simple_match(manual_path, script_dir)
        process_file(manual_path, simple_path, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
