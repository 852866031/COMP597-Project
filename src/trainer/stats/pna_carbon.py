# src/trainer/stats/pna_carbon.py
"""
PNACarbonStats — energy and carbon emission stats for the PNA workload.

Uses CodeCarbon's OfflineEmissionsTracker in task mode.  A single task
accumulates energy across consecutive training steps; it is closed (and a new
one immediately opened) at the first step boundary where the window has lasted
at least 500 ms.  This ensures CodeCarbon's power-sampling thread has enough
time to take at least one reading per window.

One CSV is written to output_dir/carbon/ when log_stats() is called:

  pna_carbon_bs<N>_wk<M>_steps.csv
      One row per training step.  Energy columns are empty ("") for steps that
      did not close a measurement window.  Non-empty rows hold the energy
      accumulated since the previous measurement; divide by window_steps in
      post-processing to obtain a per-step average.

      Columns:
        epoch            — training epoch
        step_idx         — global step counter
        step_ms          — CUDA-synced wall-clock step duration (ms)
        energy_consumed  — accumulated kWh for this window, or ""
        cpu_energy       — kWh attributed to CPU, or ""
        gpu_energy       — kWh attributed to GPU, or ""
        ram_energy       — kWh attributed to RAM, or ""
        emissions        — kg CO₂ eq. for this window, or ""

Configuration is read from conf.trainer_stats_configs.codecarbon.*
(run_num, project_name, output_dir).

Design note — why deferred merge
---------------------------------
CodeCarbon's BaseOutput.task_out() is invoked **once** when tracker.stop() is
called (not after each stop_task()).  Reading energy inline in stop_step() after
stop_task() therefore always returns None.

The fix:
  1. Record (task_name, end_step_idx) in _window_closings whenever a window
     is closed via stop_task().
  2. stop_train() closes any open window then calls tracker.stop(), which
     fires task_out() and populates _mem_output.tasks with all tasks.
  3. log_stats() calls _merge_energy() which matches task_name → _StepRow
     and fills in the energy fields before writing the CSV.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
from codecarbon import OfflineEmissionsTracker
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData, TaskEmissionsData
import codecarbon.core.cpu

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as stats_utils

# Force CodeCarbon to use constant CPU TDP (not live psutil sampling).
codecarbon.core.cpu.is_psutil_available = lambda: False

logger = logging.getLogger(__name__)

trainer_stats_name = "pna_carbon"

# Minimum wall-clock nanoseconds a measurement window must span (500 ms).
_SAMPLE_INTERVAL_NS: int = 500_000_000


def _flt(v) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# In-memory CodeCarbon output handler
# ---------------------------------------------------------------------------

class _InMemoryTaskOutput(BaseOutput):
    """Collects all task emissions data when tracker.stop() fires task_out().

    task_out() is called ONCE at tracker.stop() with the full list of all
    tasks accumulated during the run.  We store all of them so that
    _merge_energy() can look up any task by name.
    """

    def __init__(self) -> None:
        self.tasks: List[dict] = []

    def out(self, total: EmissionsData, delta: EmissionsData) -> None:
        pass

    def live_out(self, total: EmissionsData, delta: EmissionsData) -> None:
        pass

    def task_out(self, data: List[TaskEmissionsData], experiment_name: str) -> None:
        # data contains ALL tasks accumulated during the run.
        self.tasks = [dict(d.values) for d in data]


# ---------------------------------------------------------------------------
# Per-step data row
# ---------------------------------------------------------------------------

@dataclass
class _StepRow:
    epoch:           int
    step_idx:        int
    step_ms:         float
    forward_ms:      float
    backward_ms:     float
    optimizer_ms:    float
    energy_consumed: Optional[float]  # kWh; None = window not closed this step
    cpu_energy:      Optional[float]
    gpu_energy:      Optional[float]
    ram_energy:      Optional[float]
    emissions:       Optional[float]  # kgCO2eq


def _to_csv_dict(row: _StepRow) -> dict:
    return {k: ("" if v is None else v) for k, v in asdict(row).items()}


# ---------------------------------------------------------------------------
# Factory (auto-discovery entry point)
# ---------------------------------------------------------------------------

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning("PNACarbonStats: no device provided; falling back to default")
        device = torch.get_default_device()

    cc = conf.trainer_stats_configs.codecarbon
    try:
        batch_size = int(conf.model_configs.pna.batch_size)
    except AttributeError:
        batch_size = int(conf.batch_size)

    try:
        num_workers = int(conf.model_configs.pna.num_workers)
    except AttributeError:
        num_workers = 0

    return PNACarbonStats(
        device=device,
        run_num=int(cc.run_num),
        project_name=str(cc.project_name),
        output_dir=str(cc.output_dir),
        batch_size=batch_size,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Main stats class
# ---------------------------------------------------------------------------

class PNACarbonStats(base.TrainerStats):
    """Energy and carbon stats using CodeCarbon with ≥500 ms measurement windows.

    Parameters
    ----------
    device :       PyTorch device (CUDA sync + GPU index lookup).
    run_num :      Integer run identifier embedded in file names.
    project_name : Human-readable label passed to CodeCarbon.
    output_dir :   Root results directory; files go to output_dir/carbon/.
    batch_size :   Embedded in output file names.
    num_workers :  Embedded in output file names.
    """

    def __init__(
        self,
        device: torch.device,
        run_num: int,
        project_name: str,
        output_dir: str,
        batch_size: int = 0,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.device       = device
        self.run_num      = run_num
        self.project_name = project_name
        self.output_dir   = output_dir
        self.batch_size   = batch_size
        self.num_workers  = num_workers

        os.makedirs(output_dir, exist_ok=True)
        carbon_dir = os.path.join(output_dir, "carbon")
        os.makedirs(carbon_dir, exist_ok=True)

        gpu_id = device.index if (device.type == "cuda" and device.index is not None) else 0
        stem   = f"pna_carbon_bs{batch_size}_wk{num_workers}"

        self._steps_csv_path = os.path.join(carbon_dir, f"{stem}_steps.csv")

        # In-memory handler — populated once at tracker.stop().
        self._mem_output = _InMemoryTaskOutput()

        self._step_tracker = OfflineEmissionsTracker(
            project_name=project_name,
            experiment_name="steps",
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[self._mem_output],
            allow_multiple_runs=True,
            api_call_interval=-1,   # continuous background sampling
            gpu_ids=[gpu_id],
            log_level="warning",
        )

        # Counters / state
        self._step_idx: int      = 0
        self._current_epoch: int = 0
        self._recording: bool    = False

        # Measurement window state
        self._measuring: bool    = False
        self._task_name: str     = ""
        self._task_start_ns: int = 0

        # Deferred merge: each closed window records (task_name, end_step_idx).
        # _merge_energy() fills energy into the matching _StepRow after
        # tracker.stop() has populated _mem_output.tasks.
        self._window_closings: List[Tuple[str, int]] = []

        # CUDA-synced step and substep timers
        self._t_step      = stats_utils.RunningTimer()
        self._t_forward   = stats_utils.RunningTimer()
        self._t_backward  = stats_utils.RunningTimer()
        self._t_optimizer = stats_utils.RunningTimer()

        # Accumulated per-step rows (written to CSV by log_stats)
        self._rows: List[_StepRow] = []

        # Start the CodeCarbon background sampling thread.
        self._step_tracker.start()

        logger.info("PNACarbonStats: steps CSV -> %s",
                    os.path.abspath(self._steps_csv_path))

    # ------------------------------------------------------------------
    # CUDA sync helper
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    # TrainerStats interface
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        # Force-close any open measurement window at the epoch boundary,
        # even if the 500 ms interval has not been reached yet.
        if self._measuring and self._recording:
            self._window_closings.append((self._task_name, self._step_idx))
            self._step_tracker.stop_task(task_name=self._task_name)
            self._measuring = False
        self._current_epoch = epoch

    def start_train(self) -> None:
        self._recording = True

    def stop_train(self) -> None:
        """Close any open window, stop the tracker, and trigger task_out()."""
        self._recording = False
        if self._measuring:
            # Record this window closing before stop_task().
            self._window_closings.append((self._task_name, self._step_idx))
            self._step_tracker.stop_task(task_name=self._task_name)
            self._measuring = False
        # tracker.stop() fires task_out() on all output handlers, which
        # populates self._mem_output.tasks with all accumulated task data.
        self._step_tracker.stop()

    # --- step ----------------------------------------------------------

    def start_step(self) -> None:
        self._step_idx += 1
        self._sync()
        self._t_step.start()
        # Open a new measurement window if none is currently active.
        if self._recording and not self._measuring:
            self._task_name     = f"e{self._current_epoch}_step_{self._step_idx}"
            self._task_start_ns = time.perf_counter_ns()
            self._step_tracker.start_task(task_name=self._task_name)
            self._measuring = True

    def stop_step(self) -> None:
        self._sync()
        self._t_step.stop()
        if not self._recording or not self._measuring:
            return
        elapsed = time.perf_counter_ns() - self._task_start_ns
        if elapsed >= _SAMPLE_INTERVAL_NS:
            # Record which step closed this window — energy filled in later.
            self._window_closings.append((self._task_name, self._step_idx))
            self._step_tracker.stop_task(task_name=self._task_name)
            self._measuring = False
            # Do NOT read energy here; task_out() hasn't fired yet.

    # --- forward -------------------------------------------------------

    def start_forward(self) -> None:
        self._sync()
        self._t_forward.start()

    def stop_forward(self) -> None:
        self._sync()
        self._t_forward.stop()

    # --- backward ------------------------------------------------------

    def start_backward(self) -> None:
        self._sync()
        self._t_backward.start()

    def stop_backward(self) -> None:
        self._sync()
        self._t_backward.stop()

    # --- optimizer -----------------------------------------------------

    def start_optimizer_step(self) -> None:
        self._sync()
        self._t_optimizer.start()

    def stop_optimizer_step(self) -> None:
        self._sync()
        self._t_optimizer.stop()

    # --- checkpoint (no-op) -------------------------------------------

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def log_step(self) -> None:
        """Queue a per-step row.  Energy is always None here; filled by _merge_energy()."""
        if not self._recording:
            return
        self._rows.append(_StepRow(
            epoch           = self._current_epoch,
            step_idx        = self._step_idx,
            step_ms         = self._t_step.get_last()      / 1e6,
            forward_ms      = self._t_forward.get_last()   / 1e6,
            backward_ms     = self._t_backward.get_last()  / 1e6,
            optimizer_ms    = self._t_optimizer.get_last() / 1e6,
            energy_consumed = None,
            cpu_energy      = None,
            gpu_energy      = None,
            ram_energy      = None,
            emissions       = None,
        ))

    def log_stats(self) -> None:
        """Merge energy data (now available) and write the CSV."""
        self._merge_energy()
        self._write_steps_csv()

    # ------------------------------------------------------------------
    # Deferred energy merge
    # ------------------------------------------------------------------

    def _merge_energy(self) -> None:
        """Fill energy fields into _StepRow objects after tracker.stop().

        _mem_output.tasks is a list of dicts, each with a 'task_name' key,
        populated once by task_out() at tracker.stop().  We match each entry
        in _window_closings to both its task dict and the _StepRow whose
        step_idx equals end_step_idx.
        """
        if not self._mem_output.tasks:
            logger.warning(
                "PNACarbonStats: _mem_output.tasks is empty after tracker.stop(); "
                "no energy data will appear in the CSV."
            )
            return

        # Build lookup: task_name -> task dict
        task_map: dict = {}
        for t in self._mem_output.tasks:
            name = str(t.get("task_name", ""))
            if name:
                task_map[name] = t

        # Build lookup: step_idx -> _StepRow
        step_row_map = {r.step_idx: r for r in self._rows}

        merged = 0
        for task_name, end_step_idx in self._window_closings:
            task = task_map.get(task_name)
            row  = step_row_map.get(end_step_idx)
            if task is None:
                logger.debug("PNACarbonStats: no task data for task_name=%r", task_name)
                continue
            if row is None:
                logger.debug(
                    "PNACarbonStats: no row for end_step_idx=%d (task=%r)",
                    end_step_idx, task_name,
                )
                continue
            row.energy_consumed = _flt(task.get("energy_consumed"))
            row.cpu_energy      = _flt(task.get("cpu_energy"))
            row.gpu_energy      = _flt(task.get("gpu_energy"))
            row.ram_energy      = _flt(task.get("ram_energy"))
            row.emissions       = _flt(task.get("emissions"))
            merged += 1

        logger.info(
            "PNACarbonStats: merged energy for %d / %d windows",
            merged, len(self._window_closings),
        )

    # ------------------------------------------------------------------
    # CSV writer
    # ------------------------------------------------------------------

    def _write_steps_csv(self) -> None:
        if not self._rows:
            logger.warning("PNACarbonStats: no rows collected; skipping CSV")
            return
        fieldnames = list(asdict(self._rows[0]).keys())
        with open(self._steps_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(_to_csv_dict(row))
        logger.info(
            "PNACarbonStats: wrote %d step rows to %s",
            len(self._rows), os.path.abspath(self._steps_csv_path),
        )
