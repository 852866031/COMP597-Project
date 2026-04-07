#!/usr/bin/env python3
"""
plot_overhead.py — measurement overhead comparison bar chart.

Compares average p95 step latency across four workloads:
  1. baseline      — simple trainer, no instrumentation    (simple/)
  2. gc-off        — spike trainer, GC disabled run        (spike/)
  3. util-measure  — utils trainer, hw utilisation sampling (utils/)
  4. carbon-measure— carbon trainer, CodeCarbon task tracking (carbon/)

Output
------
  pna_result/plots/overhead.png

Usage
-----
    python pna_result/plot_overhead.py

    # Override individual CSV paths:
    python pna_result/plot_overhead.py \\
        --simple  pna_result/simple/pna_simple_bs256.csv \\
        --gc-off  pna_result/spike/pna_spike_bs256_gc_off.csv \\
        --utils   pna_result/utils/pna_utils_bs256_steps.csv \\
        --carbon  pna_result/carbon/pna_carbon_bs256_step-steps.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BAR_COLORS = [
    "#888888",   # baseline    — grey
    "#4C72B0",   # gc-manual   — blue
    "#2A9D8F",   # util        — teal
    "#E76F51",   # carbon      — terracotta
]
BAR_LABELS = [
    "simple",
    "baseline\n(gc-manual)",
    "util-measure",
    "carbon-measure",
]

P95 = 0.95


# ---------------------------------------------------------------------------
# P95 helpers
# ---------------------------------------------------------------------------

def _p95_ms(series: pd.Series) -> float:
    """Return the 95th-percentile value of a numeric series (in ms)."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(P95))


def _avg_p95_ms(series: pd.Series, window: int = 50) -> float:
    """
    Average of a rolling p95 — gives a stable single-number summary that
    is robust to one-off outliers while still capturing sustained latency.

    Strategy: split the series into non-overlapping windows of *window*
    steps, compute the p95 of each window, then average those.
    Falls back to a global p95 when there are fewer rows than *window*.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if clean.empty:
        return float("nan")
    if len(clean) < window:
        return float(clean.quantile(P95))

    n_windows = len(clean) // window
    p95_per_window = [
        float(clean.iloc[i * window: (i + 1) * window].quantile(P95))
        for i in range(n_windows)
    ]
    return float(np.mean(p95_per_window))


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def _discover(directory: Path, subdir: str, pattern: str) -> Optional[Path]:
    d = directory / subdir
    if not d.is_dir():
        return None
    matches = sorted(d.glob(pattern))
    return matches[0] if matches else None


def discover_simple(directory: Path) -> Optional[Path]:
    return _discover(directory, "simple", "pna_simple_bs*.csv")


def discover_gc_manual(directory: Path) -> Optional[Path]:
    return _discover(directory, "manual", "pna_manual_gc_bs*.csv")


def discover_utils(directory: Path) -> Optional[Path]:
    # exclude _agg files
    d = directory / "utils"
    if not d.is_dir():
        return None
    matches = sorted(
        p for p in d.glob("pna_utils_bs*_steps.csv")
        if "_agg" not in p.name
    )
    return matches[0] if matches else None


def discover_carbon(directory: Path) -> Optional[Path]:
    return _discover(directory, "carbon", "pna_carbon_bs*_step-steps.csv")


# ---------------------------------------------------------------------------
# Per-workload latency extraction
# ---------------------------------------------------------------------------

def load_simple_p95(path: Path) -> float:
    df = pd.read_csv(path)
    if "step_ms" not in df.columns:
        raise ValueError(f"'step_ms' column not found in {path}")
    return _avg_p95_ms(df["step_ms"])


def load_gc_manual_p95(path: Path) -> float:
    df = pd.read_csv(path)
    if "step_ms" not in df.columns:
        raise ValueError(f"'step_ms' column not found in {path}")
    return _avg_p95_ms(df["step_ms"])


def load_utils_p95(path: Path) -> float:
    df = pd.read_csv(path)
    if "step_ms" not in df.columns:
        raise ValueError(f"'step_ms' column not found in {path}")
    return _avg_p95_ms(df["step_ms"])


def load_carbon_p95(path: Path) -> float:
    """Carbon CSV stores duration in seconds; convert to ms."""
    df = pd.read_csv(path)
    if "duration" not in df.columns:
        raise ValueError(f"'duration' column not found in {path}")
    # Only keep step rows (task_name matches e{n}_step_{n})
    mask = df["task_name"].astype(str).str.match(r"e\d+_step_\d+")
    step_rows = df[mask]
    if step_rows.empty:
        raise ValueError(f"No step rows found in {path}")
    step_ms = pd.to_numeric(step_rows["duration"], errors="coerce") * 1000.0
    return _avg_p95_ms(step_ms)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_overhead(values: list, labels: list, colors: list,
                  out_path: Path, batch_size: str) -> None:
    valid   = [(v, l, c) for v, l, c in zip(values, labels, colors)
               if v is not None and not (isinstance(v, float) and np.isnan(v))]
    vals    = [x[0] for x in valid]
    lbls    = [x[1] for x in valid]
    cols    = [x[2] for x in valid]

    x = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"PNA — Measurement Overhead Comparison\n"
        f"avg p95 step latency · batch size {batch_size}",
        fontsize=13, fontweight="bold",
    )

    bars = ax.bar(x, vals, width=0.55, color=cols, alpha=0.85, zorder=3)

    # The second bar (index 1) is the baseline for overhead calculations
    baseline = vals[1] if len(vals) > 1 else None

    # Value labels on top of each bar; overhead % on bars 2+ (index >= 2)
    for i, (bar, val) in enumerate(zip(bars, vals)):
        top = bar.get_height() + max(vals) * 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top,
            f"{val:.1f} ms",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
        if i >= 2 and baseline is not None and baseline > 0:
            overhead_pct = (val - baseline) / baseline * 100.0
            sign = "+" if overhead_pct >= 0 else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top + max(vals) * 0.055,
                f"{sign}{overhead_pct:.1f}%",
                ha="center", va="bottom", fontsize=9,
                color="#E63946", fontweight="bold",
            )

    # Baseline reference line at bar 2 height
    if baseline is not None and len(vals) > 2:
        ax.axhline(y=baseline, color=cols[1], linestyle="--",
                   linewidth=1.2, alpha=0.7, zorder=2,
                   label=f"baseline ({baseline:.1f} ms)")
        ax.legend(fontsize=9, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=10)
    ax.set_ylabel("Avg p95 Step Latency (ms)")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot measurement overhead comparison across PNA workloads."
    )
    parser.add_argument("--simple", metavar="CSV",
                        help="Path to simple pna_simple_bs*.csv")
    parser.add_argument("--gc-manual", metavar="CSV", dest="gc_manual",
                        help="Path to manual pna_manual_gc_bs*.csv")
    parser.add_argument("--utils", metavar="CSV",
                        help="Path to utils pna_utils_bs*_steps.csv")
    parser.add_argument("--carbon", metavar="CSV",
                        help="Path to carbon pna_carbon_bs*_step-steps.csv")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    # Resolve paths — use CLI override if given, otherwise auto-discover
    paths = {
        "simple":    Path(args.simple).resolve()    if args.simple    else discover_simple(script_dir),
        "gc_manual": Path(args.gc_manual).resolve() if args.gc_manual else discover_gc_manual(script_dir),
        "utils":     Path(args.utils).resolve()     if args.utils     else discover_utils(script_dir),
        "carbon":    Path(args.carbon).resolve()    if args.carbon    else discover_carbon(script_dir),
    }

    loaders = {
        "simple":    load_simple_p95,
        "gc_manual": load_gc_manual_p95,
        "utils":     load_utils_p95,
        "carbon":    load_carbon_p95,
    }

    print("Sources:")
    values = []
    batch_size = "?"
    for key, path in paths.items():
        if path is None or not path.exists():
            print(f"  {key:8s} -> [not found]")
            values.append(float("nan"))
        else:
            try:
                v = loaders[key](path)
                print(f"  {key:8s} -> {path.name}  (avg p95 = {v:.2f} ms)")
                values.append(v)
                # Try to extract batch size from any file name
                import re
                m = re.search(r"bs(\d+)", path.name)
                if m and batch_size == "?":
                    batch_size = m.group(1)
            except Exception as exc:
                print(f"  {key:8s} -> [error] {exc}")
                values.append(float("nan"))

    out_path = script_dir / "plots" / "overhead.png"
    plot_overhead(values, BAR_LABELS, BAR_COLORS, out_path, batch_size)
    print("\nDone.")


if __name__ == "__main__":
    main()
