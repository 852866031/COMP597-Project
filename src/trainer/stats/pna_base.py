# src/trainer/stats/pna_base.py
"""
PNABaseStats — minimal per-step timing stats for the PNA workload.

Records only the CUDA-synchronised wall-clock duration of each training step.
No substep breakdown, no utilisation, no energy — just step_ms per row.

One CSV is written to output_dir/base/ at the end of training:

  pna_base_bs<N>_wk<M>.csv
      Columns: epoch, step_idx, step_ms
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import List

import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils

logger = logging.getLogger(__name__)

trainer_stats_name = "pna_base"


# ---------------------------------------------------------------------------
# Factory (auto-discovery entry point)
# ---------------------------------------------------------------------------

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning("PNABaseStats: no device provided; falling back to default")
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

    return PNABaseStats(
        device=device,
        output_dir=str(cc.output_dir),
        batch_size=batch_size,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Per-step data row
# ---------------------------------------------------------------------------

@dataclass
class _StepRow:
    epoch:    int
    step_idx: int
    step_ms:  float


# ---------------------------------------------------------------------------
# Main stats class
# ---------------------------------------------------------------------------

class PNABaseStats(base.TrainerStats):
    """Minimal per-step timing. Records only step_ms."""

    def __init__(
        self,
        device: torch.device,
        output_dir: str,
        batch_size: int = 0,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.device = device

        base_dir = os.path.join(output_dir, "base")
        os.makedirs(base_dir, exist_ok=True)

        stem = f"pna_base_bs{batch_size}_wk{num_workers}"
        self._csv_path = os.path.join(base_dir, f"{stem}.csv")

        self._step_idx: int = 0
        self._current_epoch: int = 0
        self._t_step = utils.RunningTimer()
        self._rows: List[_StepRow] = []

        logger.info("PNABaseStats: CSV -> %s", os.path.abspath(self._csv_path))

    # --- helpers ---

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # --- TrainerStats interface ---

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    def start_train(self) -> None:
        pass

    def stop_train(self) -> None:
        pass

    def start_step(self) -> None:
        self._step_idx += 1
        self._sync()
        self._t_step.start()

    def stop_step(self) -> None:
        self._sync()
        self._t_step.stop()

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def log_step(self) -> None:
        self._rows.append(_StepRow(
            epoch=self._current_epoch,
            step_idx=self._step_idx,
            step_ms=self._t_step.get_last() / 1e6,
        ))

    def log_stats(self) -> None:
        if not self._rows:
            logger.warning("PNABaseStats: no rows; skipping CSV")
            return
        fieldnames = list(asdict(self._rows[0]).keys())
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(asdict(row))
        logger.info("PNABaseStats: wrote %d rows to %s",
                    len(self._rows), os.path.abspath(self._csv_path))
