# src/trainer/stats/pna_manual_gc.py
"""
PNAManualGCStats — same timing measurements as PNATrainerSimpleStats but
output to output_dir/manual/.

Used with the PNA measurement trainer (--trainer pna) which suppresses
automatic GC and forces gen-2 collections between epochs.  The separate
output directory keeps these results cleanly isolated from the plain
pna_simple baseline so that overhead comparisons are unambiguous.

Two CSV files are written to output_dir/manual/ at the end of training:

  manual/pna_manual_gc_bs<N>.csv      — one row per step
  manual/pna_manual_gc_bs<N>_agg.csv  — mean + quantiles per timer
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

trainer_stats_name = "pna_manual_gc"


# ---------------------------------------------------------------------------
# Factory (auto-discovery entry point)
# ---------------------------------------------------------------------------

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning(
            "PNAManualGCStats: no device provided; falling back to default device"
        )
        device = torch.get_default_device()

    cc = conf.trainer_stats_configs.codecarbon
    try:
        batch_size = int(conf.model_configs.pna.batch_size)
    except AttributeError:
        batch_size = int(conf.batch_size)

    return PNAManualGCStats(
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
    run_num:          int
    project_name:     str
    step_idx:         int
    timestamp_s:      float
    step_ms:          float
    forward_ms:       float
    backward_ms:      float
    optimizer_ms:     float
    loss:             float


# ---------------------------------------------------------------------------
# Main stats class
# ---------------------------------------------------------------------------

class PNAManualGCStats(base.TrainerStats):
    """CUDA-synchronised timing stats for the manual-GC PNA trainer.

    Identical measurement logic to PNATrainerSimpleStats; only the output
    subdirectory and file-name prefix differ (manual/ and pna_manual_gc_bs<N>).

    Parameters
    ----------
    device:
        PyTorch device used for CUDA synchronisation.
    run_num:
        Integer run identifier embedded in file names and CSV rows.
    project_name:
        Human-readable label written into each CSV row.
    output_dir:
        Root results directory.  Files are written to output_dir/manual/.
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
        manual_dir = os.path.join(output_dir, "manual")
        os.makedirs(manual_dir, exist_ok=True)

        self._step_idx: int = 0

        self._t_step      = utils.RunningTimer()
        self._t_forward   = utils.RunningTimer()
        self._t_backward  = utils.RunningTimer()
        self._t_optimizer = utils.RunningTimer()

        self._last_loss: float = float("nan")
        self._rows: List[_StepRow] = []

        stem = f"pna_manual_gc_bs{batch_size}"
        self._steps_csv_path = os.path.join(manual_dir, f"{stem}.csv")
        self._agg_csv_path   = os.path.join(manual_dir, f"{stem}_agg.csv")

        logger.info("PNAManualGCStats: steps CSV -> %s", os.path.abspath(self._steps_csv_path))
        logger.info("PNAManualGCStats: agg CSV   -> %s", os.path.abspath(self._agg_csv_path))

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

    def start_step(self) -> None:
        self._sync()
        self._step_idx += 1
        self._t_step.start()

    def stop_step(self) -> None:
        self._sync()
        self._t_step.stop()

    def start_forward(self) -> None:
        self._sync()
        self._t_forward.start()

    def stop_forward(self) -> None:
        self._sync()
        self._t_forward.stop()

    def start_backward(self) -> None:
        self._sync()
        self._t_backward.start()

    def stop_backward(self) -> None:
        self._sync()
        self._t_backward.stop()

    def start_optimizer_step(self) -> None:
        self._sync()
        self._t_optimizer.start()

    def stop_optimizer_step(self) -> None:
        self._sync()
        self._t_optimizer.stop()

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        try:
            self._last_loss = float(loss.detach().cpu())
        except Exception:
            self._last_loss = float("nan")

    def log_step(self) -> None:
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
        self._write_steps_csv()
        self._write_agg_csv()

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _write_steps_csv(self) -> None:
        if not self._rows:
            logger.warning("PNAManualGCStats: no step rows collected; skipping steps CSV")
            return

        fieldnames = list(asdict(self._rows[0]).keys())
        with open(self._steps_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(asdict(row))

        logger.info(
            "PNAManualGCStats: wrote %d step rows to %s",
            len(self._rows), os.path.abspath(self._steps_csv_path),
        )

    def _write_agg_csv(self) -> None:
        if not self._rows:
            logger.warning("PNAManualGCStats: no step rows; skipping aggregate CSV")
            return

        import torch as _torch

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
                t = t[~t.isnan()]
                if t.numel() == 0:
                    continue
                qs = _torch.quantile(
                    t,
                    _torch.tensor(quantile_levels, dtype=_torch.float64),
                    interpolation="nearest",
                )
                row_dict: dict = {
                    "run_num":      self.run_num,
                    "project_name": self.project_name,
                    "metric":       metric_name,
                    "n":            int(t.numel()),
                    "mean":         float(t.mean()),
                }
                for q_level, q_val in zip(quantile_levels, qs.tolist()):
                    row_dict[f"q{q_level}"] = float(q_val)
                writer.writerow(row_dict)

        logger.info(
            "PNAManualGCStats: wrote aggregate stats to %s",
            os.path.abspath(self._agg_csv_path),
        )
