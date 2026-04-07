# src/trainer/stats/pna_carbon.py
"""
PNACarbonStats — energy and carbon emission stats for the PNA workload.

Uses CodeCarbon's OfflineEmissionsTracker in task mode to measure energy
consumption (kWh) and CO₂-equivalent emissions (kg CO₂ eq.) per training
step and per substep (forward / backward / optimizer).

Two task CSV files are written to output_dir/carbon/ when stop_train() is
called:

  pna_carbon_bs<N>_step-steps.csv
      One row per training step.  Key columns:
        task_name        — "e{epoch}_step_{idx}"
        duration         — wall-clock seconds
        emissions        — kg CO₂ eq.
        energy_consumed  — total kWh
        cpu_energy       — kWh attributed to CPU
        gpu_energy       — kWh attributed to GPU
        ram_energy       — kWh attributed to RAM

  pna_carbon_bs<N>_substep-substeps.csv
      One row per substep call (forward / backward / optimizer).  Same
      columns; task_name is "e{epoch}_fwd_{idx}", "e{epoch}_bwd_{idx}",
      or "e{epoch}_opt_{idx}".

These two files are sufficient to produce the six target plots:

  1. total_energy_per_step      — energy_consumed from step CSV
  2. energy_breakdown_substep   — energy_consumed per substep type, by step
  3. energy_breakdown_hardware  — cpu_energy / gpu_energy / ram_energy from step CSV
  4. carbon_per_step            — emissions from step CSV
  5. carbon_breakdown_substep   — emissions per substep type, by step
  6. carbon_breakdown_hardware  — hardware emissions derived from energy share ×
                                  carbon intensity (emissions / energy_consumed)

Configuration is read from conf.trainer_stats_configs.codecarbon.*
(run_num, project_name, output_dir) — the same CLI flags as all other
PNA stats classes.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import List

import torch
from codecarbon import OfflineEmissionsTracker
import codecarbon.core.cpu

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as stats_utils
from src.trainer.stats.codecarbon import SimpleFileOutput

# Force CodeCarbon to use constant CPU TDP (not psutil live sampling) for
# reproducibility across runs.
codecarbon.core.cpu.is_psutil_available = lambda: False

logger = logging.getLogger(__name__)

trainer_stats_name = "pna_carbon"

# Minimum nanoseconds between CodeCarbon-tracked steps.
# Steps inside the interval pass through all hooks as timing-only no-ops for
# CodeCarbon (timers still run on every step for accurate step_ms).
_SAMPLE_INTERVAL_NS: int = 1000_000_000  # 500 ms


@dataclass
class _TimingRow:
    """One row in the timing CSV — step-level wall-clock time only."""
    epoch:    int
    step_idx: int
    step_ms:  float


# ---------------------------------------------------------------------------
# Factory (auto-discovery entry point)
# ---------------------------------------------------------------------------

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning(
            "PNACarbonStats: no device provided; falling back to default device"
        )
        device = torch.get_default_device()

    cc = conf.trainer_stats_configs.codecarbon
    try:
        batch_size = int(conf.model_configs.pna.batch_size)
    except AttributeError:
        batch_size = int(conf.batch_size)

    return PNACarbonStats(
        device=device,
        run_num=int(cc.run_num),
        project_name=str(cc.project_name),
        output_dir=str(cc.output_dir),
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Main stats class
# ---------------------------------------------------------------------------

class PNACarbonStats(base.TrainerStats):
    """Energy and carbon emission stats using CodeCarbon task-mode tracking.

    Two OfflineEmissionsTracker instances run in task mode throughout
    training:

    * ``_step_tracker``    — one task per training step
    * ``_substep_tracker`` — one task per substep (forward / backward /
                             optimizer)

    Task names encode the epoch and step index in the form
    ``"e{epoch}_{phase}_{step_idx}"`` so the plotting script can
    reconstruct the time axis without a separate metadata file.

    Energy and emission data per hardware component (cpu_energy,
    gpu_energy, ram_energy) are provided directly by CodeCarbon and
    appear as columns in both output CSVs.

    Parameters
    ----------
    device:
        PyTorch device used for CUDA synchronisation and GPU index lookup.
    run_num:
        Integer run identifier embedded in file names and CSV rows.
    project_name:
        Human-readable label passed to CodeCarbon.
    output_dir:
        Root results directory.  Files are written to output_dir/carbon/.
    batch_size:
        Batch size embedded in output file names.
    """

    def __init__(
        self,
        device: torch.device,
        run_num: int,
        project_name: str,
        output_dir: str,
        batch_size: int = 0,
    ) -> None:
        super().__init__()

        self.device       = device
        self.run_num      = run_num
        self.project_name = project_name
        self.output_dir   = output_dir
        self.batch_size   = batch_size

        os.makedirs(output_dir, exist_ok=True)
        carbon_dir = os.path.join(output_dir, "carbon")
        os.makedirs(carbon_dir, exist_ok=True)

        gpu_id = device.index if (device.type == "cuda" and device.index is not None) else 0
        stem   = f"pna_carbon_bs{batch_size}"

        # ------------------------------------------------------------------
        # Step tracker — one CodeCarbon task per training step.
        # SimpleFileOutput.task_out() appends "-steps" to the stem, producing:
        #   pna_carbon_bs<N>_step-steps.csv  (task rows — what we use)
        #   pna_carbon_bs<N>_step.csv        (cumulative summary — ignored)
        # ------------------------------------------------------------------
        self._step_tracker = OfflineEmissionsTracker(
            project_name=project_name,
            experiment_name="steps",
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[
                SimpleFileOutput(
                    output_file_name=f"{stem}_step.csv",
                    output_dir=carbon_dir,
                )
            ],
            allow_multiple_runs=True,
            api_call_interval=-1,   # continuous sampling (no periodic API flush)
            gpu_ids=[gpu_id],
            log_level="warning",
        )

        # ------------------------------------------------------------------
        # Substep tracker — one task per forward / backward / optimizer call.
        # Produces:
        #   pna_carbon_bs<N>_substep-substeps.csv  (task rows — what we use)
        #   pna_carbon_bs<N>_substep.csv            (cumulative summary — ignored)
        # ------------------------------------------------------------------
        self._substep_tracker = OfflineEmissionsTracker(
            project_name=project_name,
            experiment_name="substeps",
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[
                SimpleFileOutput(
                    output_file_name=f"{stem}_substep.csv",
                    output_dir=carbon_dir,
                )
            ],
            allow_multiple_runs=True,
            api_call_interval=-1,
            gpu_ids=[gpu_id],
            log_level="warning",
        )

        # Step / epoch counters
        self._step_idx: int       = 0
        self._current_epoch: int  = 0

        # Guard flag: True only during measured epochs (between start_train and
        # stop_train).  The warmup epoch still calls start_forward / stop_forward
        # etc. via step(), so we must suppress start_task / stop_task calls there
        # to avoid the "_active_task_emissions_at_start was None" CodeCarbon error.
        self._recording: bool = False

        # Interval gate — controls CodeCarbon task start/stop only.
        # Timers run on every step; CodeCarbon only fires when the interval
        # has elapsed so start_task/stop_task overhead is amortised.
        self._last_sample_ns: int = 0
        self._sampling_this_step: bool = False

        # CUDA-synced wall-clock step timer — records full step_ms including any
        # CodeCarbon start_task / stop_task overhead on sampled steps.
        self._t_step = stats_utils.RunningTimer()

        # Accumulated timing rows written by log_stats().
        self._timing_rows: List[_TimingRow] = []
        self._timing_csv_path = os.path.join(carbon_dir, f"{stem}_timing.csv")

        # Start both task-mode trackers.  Tasks are started/stopped inside the
        # training loop; the background threads just keep sampling power.
        # The warmup epoch is excluded by the _recording guard above.
        self._substep_tracker.start()
        self._step_tracker.start()

        logger.info(
            "PNACarbonStats: step CSV    -> %s",
            os.path.abspath(os.path.join(carbon_dir, f"{stem}_step-steps.csv")),
        )
        logger.info(
            "PNACarbonStats: substep CSV -> %s",
            os.path.abspath(os.path.join(carbon_dir, f"{stem}_substep-substeps.csv")),
        )

    # ------------------------------------------------------------------
    # CUDA sync helper
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    # Task name helpers
    # ------------------------------------------------------------------

    def _step_task(self) -> str:
        """Task name for the current step: "e{epoch}_step_{idx}"."""
        return f"e{self._current_epoch}_step_{self._step_idx}"

    def _substep_task(self, phase: str) -> str:
        """Task name for a substep: "e{epoch}_{fwd|bwd|opt}_{idx}"."""
        return f"e{self._current_epoch}_{phase}_{self._step_idx}"

    # ------------------------------------------------------------------
    # TrainerStats interface
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Called by the trainer at the start of each measured epoch."""
        self._current_epoch = epoch

    def start_train(self) -> None:
        """Mark the start of the measured training phase."""
        self._recording = True

    def stop_train(self) -> None:
        """Stop recording and flush both trackers; this writes the CSVs."""
        self._recording = False
        self._step_tracker.stop()
        self._substep_tracker.stop()

    # --- step ----------------------------------------------------------

    def start_step(self) -> None:
        self._step_idx += 1
        self._sampling_this_step = False
        if not self._recording:
            return
        # Check interval gate — only open a CodeCarbon task for this step if
        # enough wall-clock time has elapsed since the last sampled step.
        now_ns = time.perf_counter_ns()
        if now_ns - self._last_sample_ns >= _SAMPLE_INTERVAL_NS:
            self._last_sample_ns = now_ns
            self._sampling_this_step = True
        # Always time the step so step_ms is available for every measured step.
        # Timer starts before start_task (when sampled) so CodeCarbon overhead
        # is included inside the timing window.
        self._sync()
        self._t_step.start()
        if self._sampling_this_step:
            self._step_tracker.start_task(task_name=self._step_task())

    def stop_step(self) -> None:
        if not self._recording:
            return
        self._sync()
        if self._sampling_this_step:
            self._step_tracker.stop_task(task_name=self._step_task())
        self._t_step.stop()

    # --- forward -------------------------------------------------------

    def start_forward(self) -> None:
        if not self._sampling_this_step:
            return
        self._sync()
        self._substep_tracker.start_task(task_name=self._substep_task("fwd"))

    def stop_forward(self) -> None:
        if not self._sampling_this_step:
            return
        self._sync()
        self._substep_tracker.stop_task(task_name=self._substep_task("fwd"))

    # --- backward ------------------------------------------------------

    def start_backward(self) -> None:
        if not self._sampling_this_step:
            return
        self._sync()
        self._substep_tracker.start_task(task_name=self._substep_task("bwd"))

    def stop_backward(self) -> None:
        if not self._sampling_this_step:
            return
        self._sync()
        self._substep_tracker.stop_task(task_name=self._substep_task("bwd"))

    # --- optimizer -----------------------------------------------------

    def start_optimizer_step(self) -> None:
        if not self._sampling_this_step:
            return
        self._sync()
        self._substep_tracker.start_task(task_name=self._substep_task("opt"))

    def stop_optimizer_step(self) -> None:
        if not self._sampling_this_step:
            return
        self._sync()
        self._substep_tracker.stop_task(task_name=self._substep_task("opt"))

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
        """Append one timing row per step — called after stop_step()."""
        if not self._recording:
            return
        self._timing_rows.append(_TimingRow(
            epoch    = self._current_epoch,
            step_idx = self._step_idx,
            step_ms  = self._t_step.get_last() / 1e6,
        ))

    def log_stats(self) -> None:
        """Write the timing CSV (CodeCarbon CSVs are written by stop_train())."""
        self._write_timing_csv()

    # ------------------------------------------------------------------
    # CSV helper
    # ------------------------------------------------------------------

    def _write_timing_csv(self) -> None:
        if not self._timing_rows:
            logger.warning("PNACarbonStats: no timing rows; skipping timing CSV")
            return
        fieldnames = list(asdict(self._timing_rows[0]).keys())
        with open(self._timing_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._timing_rows:
                writer.writerow(asdict(row))
        logger.info(
            "PNACarbonStats: wrote %d timing rows to %s",
            len(self._timing_rows), os.path.abspath(self._timing_csv_path),
        )
