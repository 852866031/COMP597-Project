#!/usr/bin/env python3
"""Run all PNA plotting scripts."""
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()

for name in ["plot_simple.py", "plot_spike.py", "plot_utils.py", "plot_carbon.py", "plot_batchsize.py", "plot_workers.py", "plot_overhead.py"]:
    script = script_dir / name
    if not script.exists():
        print(f"[skip] {name} not found")
        continue
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print('='*60)
    result = subprocess.run([sys.executable, str(script)])
    if result.returncode != 0:
        print(f"[warn] {name} exited with code {result.returncode}")

print("\nAll plots done.")
