from __future__ import annotations

import csv
import gc
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils

logger = logging.getLogger(__name__)

trainer_stats_name = "pna_spike"


# ---------------------------------------------------------------------------
# Factory (auto-discovery entry point)
# ---------------------------------------------------------------------------

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning(
            "PNATrainerSpikeStats: no device provided; falling back to default device"
        )
        device = torch.get_default_device()

    cc = conf.trainer_stats_configs.codecarbon
    try:
        batch_size = int(conf.model_configs.pna.batch_size)
    except AttributeError:
        batch_size = int(conf.batch_size)

    return PNATrainerSpikeStats(
        device=device,
        run_num=int(cc.run_num),
        project_name=str(cc.project_name),
        output_dir=str(cc.output_dir),
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Per-step data row
# ---------------------------------------------------------------------------

@dataclass
class _StepRow:
    run_num:      int
    project_name: str
    step_idx:     int
    timestamp_s:  float
    step_ms:      float
    forward_ms:   float
    backward_ms:  float
    optimizer_ms: float
    loss:         float


# ---------------------------------------------------------------------------
# Main stats class
# ---------------------------------------------------------------------------

class PNATrainerSpikeStats(base.TrainerStats):
    """CUDA-synchronised timing stats for GC spike analysis.

    Designed to be used with PNASpikeTrainer which runs a gc_on epoch followed
    by a gc_off epoch.  Rows from each run are collected into separate steps
    CSVs.

    Parameters
    ----------
    device:
        PyTorch device on which training runs.
    run_num:
        Integer run identifier.
    project_name:
        Human-readable label written into each CSV row.
    output_dir:
        Directory where CSV files are saved.  Created if it does not exist.
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
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "spike"), exist_ok=True)

        # Current run type — set by set_run_type() before each epoch
        self._run_type: str = "gc_on"

        # Step counter — reset to 0 by set_run_type()
        self._step_idx: int = 0

        # Timers (wall-clock, nanosecond precision)
        self._t_step      = utils.RunningTimer()
        self._t_forward   = utils.RunningTimer()
        self._t_backward  = utils.RunningTimer()
        self._t_optimizer = utils.RunningTimer()

        # Last loss seen (updated by log_loss)
        self._last_loss: float = float("nan")

        # Accumulated per-step rows — one list per run
        self._rows_gc_on:     List[_StepRow] = []
        self._rows_gc_off:    List[_StepRow] = []
        self._rows_gc_manual: List[_StepRow] = []

        # GC event log — populated by the gc callback installed in start_train()
        self._gc_events: List[dict] = []
        self._gc_callback = None
        # Pending start times keyed by generation
        self._gc_start_ns: Dict[int, int] = {}

        # Output file paths
        self._gc_on_csv_path      = os.path.join(output_dir, "spike", f"pna_spike_bs{batch_size}_gc_on.csv")
        self._gc_off_csv_path     = os.path.join(output_dir, "spike", f"pna_spike_bs{batch_size}_gc_off.csv")
        self._gc_manual_csv_path  = os.path.join(output_dir, "spike", f"pna_spike_bs{batch_size}_gc_manual.csv")
        self._gc_events_csv_path  = os.path.join(output_dir, "spike", f"pna_spike_bs{batch_size}_gc_events.csv")

        logger.info(
            "PNATrainerSpikeStats: gc_on CSV  -> %s",
            os.path.abspath(self._gc_on_csv_path),
        )
        logger.info(
            "PNATrainerSpikeStats: gc_off CSV -> %s",
            os.path.abspath(self._gc_off_csv_path),
        )
        logger.info(
            "PNATrainerSpikeStats: GC events CSV -> %s",
            os.path.abspath(self._gc_events_csv_path),
        )

    # ------------------------------------------------------------------
    # Run-type control
    # ------------------------------------------------------------------

    def set_run_type(self, run_type: str) -> None:
        """Set the current run type and reset the step counter.

        Parameters
        ----------
        run_type:
            One of ``"gc_on"``, ``"gc_off"``, or ``"gc_manual"``.
        """
        self._run_type = run_type
        self._step_idx = 0

    # ------------------------------------------------------------------
    # CUDA sync helper
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    # TrainerStats interface
    # ------------------------------------------------------------------

    def start_train(self) -> None:
        """Install a gc.callback to record gen-2 collections (gc_on only)."""
        if self._run_type != "gc_on":
            return

        def _gc_hook(phase: str, info: dict) -> None:
            if phase == "start":
                self._gc_start_ns[info["generation"]] = time.perf_counter_ns()
            elif phase == "stop":
                if info["generation"] != 2:
                    self._gc_start_ns.pop(info["generation"], None)
                    return
                start_ns = self._gc_start_ns.pop(2, None)
                duration_ms = (
                    (time.perf_counter_ns() - start_ns) / 1e6
                    if start_ns is not None else float("nan")
                )
                self._gc_events.append({
                    "step_idx":      self._step_idx,
                    "generation":    2,
                    "collected":     info["collected"],
                    "uncollectable": info["uncollectable"],
                    "duration_ms":   duration_ms,
                })

        self._gc_callback = _gc_hook
        gc.callbacks.append(self._gc_callback)

    def stop_train(self) -> None:
        """Remove the gc callback installed in start_train."""
        if self._gc_callback is not None and self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)
            self._gc_callback = None

    # --- step ----------------------------------------------------------

    def start_step(self) -> None:
        self._sync()
        self._step_idx += 1
        self._t_step.start()

    def stop_step(self) -> None:
        self._sync()
        self._t_step.stop()

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
        """Cache the current step's loss for inclusion in the CSV row."""
        try:
            self._last_loss = float(loss.detach().cpu())
        except Exception:
            self._last_loss = float("nan")

    def log_step(self) -> None:
        """Snapshot timers after each step and queue a CSV row."""
        step_ms      = self._t_step.get_last()      / 1e6
        forward_ms   = self._t_forward.get_last()   / 1e6
        backward_ms  = self._t_backward.get_last()  / 1e6
        optimizer_ms = self._t_optimizer.get_last() / 1e6

        row = _StepRow(
            run_num      = self.run_num,
            project_name = self.project_name,
            step_idx     = self._step_idx,
            timestamp_s  = time.time(),
            step_ms      = step_ms,
            forward_ms   = forward_ms,
            backward_ms  = backward_ms,
            optimizer_ms = optimizer_ms,
            loss         = self._last_loss,
        )

        if self._run_type == "gc_on":
            self._rows_gc_on.append(row)
        elif self._run_type == "gc_manual":
            self._rows_gc_manual.append(row)
        else:
            self._rows_gc_off.append(row)

        logger.debug(
            "[%s] step %d | step=%.2f ms | fwd=%.2f ms | bwd=%.2f ms | opt=%.2f ms | loss=%.6f",
            self._run_type, self._step_idx,
            step_ms, forward_ms, backward_ms, optimizer_ms, self._last_loss,
        )

    def log_stats(self) -> None:
        """Write all CSV files once training is complete."""
        self._write_csv(self._rows_gc_on,     self._gc_on_csv_path)
        self._write_csv(self._rows_gc_off,    self._gc_off_csv_path)
        self._write_csv(self._rows_gc_manual, self._gc_manual_csv_path)
        self._write_gc_csv()

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _write_csv(self, rows: List[_StepRow], path: str) -> None:
        """Write rows to a CSV file (overwrite mode). Skips gracefully if rows is empty."""
        if not rows:
            logger.warning(
                "PNATrainerSpikeStats: no step rows to write; skipping %s", path
            )
            return

        fieldnames = list(asdict(rows[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

        logger.info(
            "PNATrainerSpikeStats: wrote %d step rows to %s",
            len(rows),
            os.path.abspath(path),
        )

    def _write_gc_csv(self) -> None:
        """Write one row per gen-2 GC event recorded during the gc_on epoch."""
        if not self._gc_events:
            logger.info("PNATrainerSpikeStats: no GC events recorded; skipping GC events CSV")
            return

        fieldnames = ["step_idx", "generation", "collected", "uncollectable", "duration_ms"]
        with open(self._gc_events_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in self._gc_events:
                writer.writerow({k: event[k] for k in fieldnames})

        logger.info(
            "PNATrainerSpikeStats: wrote %d GC events to %s",
            len(self._gc_events),
            os.path.abspath(self._gc_events_csv_path),
        )
