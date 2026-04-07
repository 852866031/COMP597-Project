# src/trainer/stats/pna_simple.py
"""
PNATrainerSimpleStats — lightweight timing stats for the PNA workload.

Measures wall-clock time (CUDA-synchronised) for each training step and its
three substeps (forward, backward, optimizer).  Two CSV files are written to
output_dir at the end of training:

  simple/result.csv       — one row per step
  simple/result_agg.csv   — mean + quantiles per timer

Configuration is read from conf.trainer_stats_configs.codecarbon.*
(run_num, project_name, output_dir) so the same CLI flags drive all stats
outputs.  The device is provided via the `device` kwarg passed by model_init.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils

logger = logging.getLogger(__name__)

trainer_stats_name = "pna_simple"


# ---------------------------------------------------------------------------
# Factory (auto-discovery entry point)
# ---------------------------------------------------------------------------

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning(
            "PNATrainerSimpleStats: no device provided; falling back to default device"
        )
        device = torch.get_default_device()

    cc = conf.trainer_stats_configs.codecarbon
    # Read the PNA-specific batch size so it can be embedded in file names.
    # Falls back to the global conf.batch_size if the model config is unavailable.
    try:
        batch_size = int(conf.model_configs.pna.batch_size)
    except AttributeError:
        batch_size = int(conf.batch_size)

    try:
        num_workers = int(conf.model_configs.pna.num_workers)
    except AttributeError:
        num_workers = 0

    return PNATrainerSimpleStats(
        device=device,
        run_num=int(cc.run_num),
        project_name=str(cc.project_name),
        output_dir=str(cc.output_dir),
        batch_size=batch_size,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Per-step data row
# ---------------------------------------------------------------------------

@dataclass
class _StepRow:
    run_num:          int
    project_name:     str
    step_idx:         int
    timestamp_s:      float   # wall-clock time at step end
    step_ms:          float
    forward_ms:       float
    backward_ms:      float
    optimizer_ms:     float
    loss:             float


# ---------------------------------------------------------------------------
# Main stats class
# ---------------------------------------------------------------------------

class PNATrainerSimpleStats(base.TrainerStats):
    """CUDA-synchronised timing stats for the PNA trainer.

    Writes per-step timings and end-of-training aggregate statistics to CSV
    files inside output_dir.

    Parameters
    ----------
    device:
        PyTorch device on which training runs.  Used only for CUDA
        synchronisation; CPU training is also supported (sync becomes a no-op).
    run_num:
        Integer run identifier written into file names and CSV rows so that
        multiple experiments can share the same output directory.
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
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.device       = device
        self.run_num      = run_num
        self.project_name = project_name
        self.output_dir   = output_dir
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "simple"), exist_ok=True)

        # Step counter
        self._step_idx: int = 0

        # Timers (wall-clock, nanosecond precision)
        self._t_step      = utils.RunningTimer()
        self._t_forward   = utils.RunningTimer()
        self._t_backward  = utils.RunningTimer()
        self._t_optimizer = utils.RunningTimer()

        # Last loss seen (updated by log_loss)
        self._last_loss: float = float("nan")

        # Accumulated per-step rows (flushed to CSV in log_stats)
        self._rows: List[_StepRow] = []

        # Derive output file paths
        self._steps_csv_path = os.path.join(output_dir, "simple", f"pna_simple_bs{batch_size}_wk{num_workers}.csv")
        self._agg_csv_path   = os.path.join(output_dir, "simple", f"pna_simple_bs{batch_size}_wk{num_workers}_agg.csv")

        logger.info(
            "PNATrainerSimpleStats: steps CSV -> %s",
            os.path.abspath(self._steps_csv_path),
        )
        logger.info(
            "PNATrainerSimpleStats: aggregate CSV -> %s",
            os.path.abspath(self._agg_csv_path),
        )

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
        pass

    def stop_train(self) -> None:
        pass

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
        # Convert ns -> ms for readability
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
        self._rows.append(row)

        logger.debug(
            "step %d | step=%.2f ms | fwd=%.2f ms | bwd=%.2f ms | opt=%.2f ms | loss=%.6f",
            self._step_idx, step_ms, forward_ms, backward_ms, optimizer_ms, self._last_loss,
        )

    def log_stats(self) -> None:
        """Write all CSV files once training is complete."""
        self._write_steps_csv()
        self._write_agg_csv()

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _write_steps_csv(self) -> None:
        """Write one row per training step to the steps CSV."""
        if not self._rows:
            logger.warning("PNATrainerSimpleStats: no step rows collected; skipping steps CSV")
            return

        fieldnames = list(asdict(self._rows[0]).keys())
        with open(self._steps_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(asdict(row))

        logger.info(
            "PNATrainerSimpleStats: wrote %d step rows to %s",
            len(self._rows),
            os.path.abspath(self._steps_csv_path),
        )

    def _write_agg_csv(self) -> None:
        """Write per-metric mean and quantiles to the aggregate CSV."""
        if not self._rows:
            logger.warning("PNATrainerSimpleStats: no step rows; skipping aggregate CSV")
            return

        import torch as _torch

        # Metric name -> list of values across all steps
        metrics = {
            "step_ms":      [r.step_ms      for r in self._rows],
            "forward_ms":   [r.forward_ms   for r in self._rows],
            "backward_ms":  [r.backward_ms  for r in self._rows],
            "optimizer_ms": [r.optimizer_ms for r in self._rows],
            "loss":         [r.loss         for r in self._rows],
        }

        quantile_levels = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]

        fieldnames = (
            ["run_num", "project_name", "metric", "n", "mean"]
            + [f"q{q}" for q in quantile_levels]
        )

        with open(self._agg_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metric_name, values in metrics.items():
                t = _torch.tensor(values, dtype=_torch.float64)
                # Filter out NaNs (e.g. loss may be nan if log_loss was never called)
                t = t[~t.isnan()]
                if t.numel() == 0:
                    continue
                qs = _torch.quantile(
                    t,
                    _torch.tensor(quantile_levels, dtype=_torch.float64),
                    interpolation="nearest",
                )
                row: dict = {
                    "run_num":      self.run_num,
                    "project_name": self.project_name,
                    "metric":       metric_name,
                    "n":            int(t.numel()),
                    "mean":         float(t.mean()),
                }
                for q_level, q_val in zip(quantile_levels, qs.tolist()):
                    row[f"q{q_level}"] = float(q_val)
                writer.writerow(row)

        logger.info(
            "PNATrainerSimpleStats: wrote aggregate stats to %s",
            os.path.abspath(self._agg_csv_path),
        )
