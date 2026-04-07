# src/trainer/stats/pna_utils.py
"""
PNAUtilsStats — hardware utilisation stats for the PNA workload.

Timing + GPU/CPU/RAM utilisation are recorded per step.
Utilisation is sampled at most once every 500 ms; steps that fall inside
a sampling gap get empty fields for those columns.

Two CSV files are written to output_dir/carbon/ at the end of training:

  pna_utils_bs<N>_steps.csv      — one row per training step
  pna_utils_bs<N>_steps_agg.csv  — mean + quantiles per metric

Configuration is read from conf.trainer_stats_configs.codecarbon.*
(run_num, project_name, output_dir).
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils

# Optional deps — gracefully degrade if unavailable
try:
    import pynvml as _pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

trainer_stats_name = "pna_utils"

# Minimum nanoseconds between utilisation samples (500 ms)
_SAMPLE_INTERVAL_NS: int = 250_000_000


# ---------------------------------------------------------------------------
# Factory (auto-discovery entry point)
# ---------------------------------------------------------------------------

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning(
            "PNAUtilsStats: no device provided; falling back to default device"
        )
        device = torch.get_default_device()

    cc = conf.trainer_stats_configs.codecarbon
    try:
        batch_size = int(conf.model_configs.pna.batch_size)
    except AttributeError:
        batch_size = int(conf.batch_size)

    return PNAUtilsStats(
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
    epoch:            int
    step_idx:         int
    timestamp_s:      float
    step_ms:          float
    forward_ms:       float
    backward_ms:      float
    optimizer_ms:     float
    loss:             float
    fwd_gpu_util:     Optional[float]  # % GPU util sampled at stop_forward
    fwd_cpu_util:     Optional[float]  # % CPU util sampled at stop_forward
    fwd_ram_used_mb:  Optional[float]  # RAM MiB       sampled at stop_forward
    bwd_gpu_util:     Optional[float]  # % GPU util sampled at stop_backward
    bwd_cpu_util:     Optional[float]  # % CPU util sampled at stop_backward
    bwd_ram_used_mb:  Optional[float]  # RAM MiB       sampled at stop_backward


# ---------------------------------------------------------------------------
# CSV serialisation helper
# ---------------------------------------------------------------------------

def _to_csv_dict(row) -> dict:
    """Convert a dataclass row to a CSV-ready dict (None → empty string)."""
    return {k: ("" if v is None else v) for k, v in asdict(row).items()}


# ---------------------------------------------------------------------------
# Main stats class
# ---------------------------------------------------------------------------

class PNAUtilsStats(base.TrainerStats):
    """Hardware utilisation stats for the PNA trainer.

    GPU utilisation is sampled via pynvml; CPU and RAM via psutil.
    Sampling is gated by a 500 ms wall-clock interval so that fast steps
    only pay the sampling cost occasionally.

    Parameters
    ----------
    device:
        PyTorch device used for CUDA synchronisation and GPU handle lookup.
    run_num:
        Integer run identifier embedded in file names and CSV rows.
    project_name:
        Human-readable label written into each CSV row.
    output_dir:
        Root results directory.  Files are written to output_dir/carbon/.
    batch_size:
        Batch size embedded in the output file names.
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
        carbon_dir = os.path.join(output_dir, "utils")
        os.makedirs(carbon_dir, exist_ok=True)

        # Step and epoch counters
        self._step_idx: int = 0
        self._current_epoch: int = 0

        # Timers (wall-clock, nanosecond precision)
        self._t_step      = utils.RunningTimer()
        self._t_forward   = utils.RunningTimer()
        self._t_backward  = utils.RunningTimer()
        self._t_optimizer = utils.RunningTimer()

        # Last loss seen (updated by log_loss)
        self._last_loss: float = float("nan")

        # Accumulated per-step rows
        self._rows: List[_StepRow] = []

        # Output file paths
        stem = f"pna_utils_bs{batch_size}"
        self._steps_csv_path     = os.path.join(carbon_dir, f"{stem}_steps.csv")
        self._steps_agg_csv_path = os.path.join(carbon_dir, f"{stem}_steps_agg.csv")

        # ---------- utilisation sampling state — independent per phase ----------
        # Each phase has its own 500 ms gate so forward and backward can both
        # fire in the same step without interfering with each other.
        self._last_fwd_sample_ts_ns: int = 0
        self._last_bwd_sample_ts_ns: int = 0
        self._fwd_util: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None)
        self._bwd_util: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None)

        # pynvml handle (None if CUDA unavailable or pynvml not installed)
        self._nvml_handle = None
        self._nvml_available = False
        if _PYNVML_AVAILABLE and device.type == "cuda":
            try:
                _pynvml.nvmlInit()
                gpu_index = device.index if device.index is not None else 0
                self._nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self._nvml_available = True
                logger.info("PNAUtilsStats: pynvml initialised for GPU %d", gpu_index)
            except Exception as exc:
                logger.warning("PNAUtilsStats: pynvml init failed: %s", exc)

        # Per-process psutil handle for CPU measurement.
        # A single persistent Process object is required so that consecutive
        # cpu_percent(interval=None) calls have a valid prior-measurement
        # baseline (the first call on a brand-new object always returns 0.0).
        self._psutil_proc = None
        if _PSUTIL_AVAILABLE:
            try:
                self._psutil_proc = _psutil.Process(os.getpid())
                # Warm-up call so the first real sample isn't always 0.0
                self._psutil_proc.cpu_percent(interval=None)
            except Exception as exc:
                logger.warning("PNAUtilsStats: psutil Process init failed: %s", exc)

        if not _PSUTIL_AVAILABLE:
            logger.warning("PNAUtilsStats: psutil not installed; CPU/RAM util will be empty")

        logger.info("PNAUtilsStats: steps CSV     -> %s", os.path.abspath(self._steps_csv_path))
        logger.info("PNAUtilsStats: steps agg CSV -> %s", os.path.abspath(self._steps_agg_csv_path))

    # ------------------------------------------------------------------
    # CUDA sync helper
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    # Utilisation sampling
    # ------------------------------------------------------------------

    def _sample_hw(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Read GPU/CPU/RAM utilisation unconditionally. Called only when gate allows."""
        gpu_util: Optional[float] = None
        if self._nvml_available and self._nvml_handle is not None:
            try:
                rates = _pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util = float(rates.gpu)
            except Exception as exc:
                logger.debug("PNAUtilsStats: GPU util sample failed: %s", exc)

        cpu_util: Optional[float] = None
        if _PSUTIL_AVAILABLE and self._psutil_proc is not None:
            try:
                # cpu_percent(interval=None) measures usage since the last call
                # on this Process object — i.e. only this process, on its
                # allocated core(s).  Values can exceed 100 % if multiple
                # threads are active; with --cpus-per-task=1 that is unlikely.
                cpu_util = float(self._psutil_proc.cpu_percent(interval=None))
            except Exception as exc:
                logger.debug("PNAUtilsStats: CPU util sample failed: %s", exc)

        ram_used_mb: Optional[float] = None
        if _PSUTIL_AVAILABLE:
            try:
                ram_used_mb = float(_psutil.virtual_memory().used) / (1024.0 * 1024.0)
            except Exception as exc:
                logger.debug("PNAUtilsStats: RAM util sample failed: %s", exc)

        return (gpu_util, cpu_util, ram_used_mb)

    def _maybe_sample_fwd(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Sample for the forward phase if 500 ms have elapsed since the last forward sample."""
        now_ns = time.perf_counter_ns()
        if now_ns - self._last_fwd_sample_ts_ns < _SAMPLE_INTERVAL_NS:
            return (None, None, None)
        self._last_fwd_sample_ts_ns = now_ns
        return self._sample_hw()

    def _maybe_sample_bwd(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Sample for the backward phase if 500 ms have elapsed since the last backward sample."""
        now_ns = time.perf_counter_ns()
        if now_ns - self._last_bwd_sample_ts_ns < _SAMPLE_INTERVAL_NS:
            return (None, None, None)
        self._last_bwd_sample_ts_ns = now_ns
        return self._sample_hw()

    # ------------------------------------------------------------------
    # TrainerStats interface
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Called by the trainer at the start of each measured epoch."""
        self._current_epoch = epoch

    def start_train(self) -> None:
        pass

    def stop_train(self) -> None:
        pass

    # --- step ----------------------------------------------------------

    def start_step(self) -> None:
        self._fwd_util = (None, None, None)
        self._bwd_util = (None, None, None)
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
        sample = self._maybe_sample_fwd()
        if sample != (None, None, None):
            self._fwd_util = sample
        

    # --- backward ------------------------------------------------------

    def start_backward(self) -> None:
        self._sync()
        self._t_backward.start()

    def stop_backward(self) -> None:
        self._sync()
        self._t_backward.stop()
        sample = self._maybe_sample_bwd()
        if sample != (None, None, None):
            self._bwd_util = sample
        

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
        try:
            self._last_loss = float(loss.detach().cpu())
        except Exception:
            self._last_loss = float("nan")

    def log_step(self) -> None:
        step_ms      = self._t_step.get_last()      / 1e6
        forward_ms   = self._t_forward.get_last()   / 1e6
        backward_ms  = self._t_backward.get_last()  / 1e6
        optimizer_ms = self._t_optimizer.get_last() / 1e6

        fwd_gpu, fwd_cpu, fwd_ram = self._fwd_util
        bwd_gpu, bwd_cpu, bwd_ram = self._bwd_util

        row = _StepRow(
            run_num          = self.run_num,
            project_name     = self.project_name,
            epoch            = self._current_epoch,
            step_idx         = self._step_idx,
            timestamp_s      = time.time(),
            step_ms          = step_ms,
            forward_ms       = forward_ms,
            backward_ms      = backward_ms,
            optimizer_ms     = optimizer_ms,
            loss             = self._last_loss,
            fwd_gpu_util     = fwd_gpu,
            fwd_cpu_util     = fwd_cpu,
            fwd_ram_used_mb  = fwd_ram,
            bwd_gpu_util     = bwd_gpu,
            bwd_cpu_util     = bwd_cpu,
            bwd_ram_used_mb  = bwd_ram,
        )
        self._rows.append(row)

        logger.debug(
            "step %d | step=%.2f ms | fwd=%.2f ms | bwd=%.2f ms | opt=%.2f ms"
            " | loss=%.6f | fwd_gpu=%s%% bwd_gpu=%s%%",
            self._step_idx, step_ms, forward_ms, backward_ms, optimizer_ms,
            self._last_loss,
            f"{fwd_gpu:.1f}" if fwd_gpu is not None else "—",
            f"{bwd_gpu:.1f}" if bwd_gpu is not None else "—",
        )

    def log_stats(self) -> None:
        self._write_steps_csv()
        self._write_agg_csv()

        if self._nvml_available:
            try:
                _pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _write_steps_csv(self) -> None:
        if not self._rows:
            logger.warning("PNAUtilsStats: no step rows collected; skipping steps CSV")
            return

        fieldnames = list(asdict(self._rows[0]).keys())
        with open(self._steps_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(_to_csv_dict(row))

        logger.info(
            "PNAUtilsStats: wrote %d step rows to %s",
            len(self._rows), os.path.abspath(self._steps_csv_path),
        )

    def _write_agg_csv(self) -> None:
        if not self._rows:
            logger.warning("PNAUtilsStats: no step rows; skipping aggregate CSV")
            return

        import torch as _torch

        metrics = {
            "step_ms":      [r.step_ms      for r in self._rows],
            "forward_ms":   [r.forward_ms   for r in self._rows],
            "backward_ms":  [r.backward_ms  for r in self._rows],
            "optimizer_ms": [r.optimizer_ms for r in self._rows],
            "loss":         [r.loss         for r in self._rows],
        }

        for col, attr in [
            ("fwd_gpu_util",    "fwd_gpu_util"),
            ("fwd_cpu_util",    "fwd_cpu_util"),
            ("fwd_ram_used_mb", "fwd_ram_used_mb"),
            ("bwd_gpu_util",    "bwd_gpu_util"),
            ("bwd_cpu_util",    "bwd_cpu_util"),
            ("bwd_ram_used_mb", "bwd_ram_used_mb"),
        ]:
            vals = [getattr(r, attr) for r in self._rows if getattr(r, attr) is not None]
            if vals:
                metrics[col] = vals

        quantile_levels = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]

        fieldnames = (
            ["run_num", "project_name", "metric", "n", "mean"]
            + [f"q{q}" for q in quantile_levels]
        )

        with open(self._steps_agg_csv_path, "w", newline="") as f:
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
            "PNAUtilsStats: wrote aggregate stats to %s",
            os.path.abspath(self._steps_agg_csv_path),
        )
