# src/trainer/pna.py
"""
PNATrainer — trainer for the energy consumption measurement workload.

Key difference from PNASimpleTrainer: automatic Python GC is disabled for the
duration of training and a full gen-2 collection is manually triggered
between epochs.  This prevents GC pauses from contaminating per-step timing
and energy measurements.

Selected via  --trainer pna_carbon  in the launch script.
"""
from __future__ import annotations


from typing import Any, Dict, Optional

import tqdm.auto

from src.trainer.pna_simple import PNATrainer


class PNAMeasurementTrainer(PNATrainer):
    """PNA trainer dedicated to energy consumption measurement.

    Automatic GC is suppressed during training so that gen-2 collection
    pauses cannot fire inside a measured step.  Instead, ``gc.collect(2)``
    is called explicitly between epochs, keeping the pause outside any
    measurement window.

    Energy instrumentation (CodeCarbon, RAPL, NVML, etc.) will be layered
    into ``start_train`` / ``stop_train`` in the stats class without
    touching this trainer.
    """

    def train(self, model_kwargs: Optional[Dict[str, Any]] = None) -> None:
        import gc
        if model_kwargs is None:
            model_kwargs = {}

        self.model.train()

        # ------------------------------------------------------------------
        # Disable automatic GC for the entire training run.
        # A full gen-2 collection is forced manually between epochs so that
        # accumulated cyclic garbage is cleared at a controlled, unmetered
        # point rather than mid-step.
        # ------------------------------------------------------------------
        gc.disable()

        # Warmup epoch (no recording)
        warmup_bar = tqdm.auto.tqdm(range(self.num_batches), desc="warmup")
        self._run_epoch(epoch=0, model_kwargs=model_kwargs,
                        record=False, progress_bar=warmup_bar)
        warmup_bar.close()
        gc.collect(2)

        # Measured epochs
        progress_bar = tqdm.auto.tqdm(
            range(self.num_batches * self.conf.epochs), desc="loss: N/A"
        )

        self.stats.start_train()

        for epoch in range(1, self.conf.epochs + 1):
            # Force a full gen-2 collection BEFORE the epoch starts.
            # gc.collect(2) collects all three generations, clearing any
            # cyclic garbage that built up during the previous epoch so it
            # cannot fire unpredictably inside a measured step.
            gc.collect(2)

            # Notify stats of the current epoch so it can tag each step row.
            if hasattr(self.stats, "set_epoch"):
                self.stats.set_epoch(epoch)

            self._run_epoch(epoch=epoch, model_kwargs=model_kwargs,
                            record=True, progress_bar=progress_bar)

        self.stats.stop_train()
        progress_bar.close()

        # Restore automatic GC now that measurements are complete.
        gc.enable()

        self.stats.log_stats()

    def _run_epoch(self, epoch: int, model_kwargs: Dict[str, Any],
                   record: bool, progress_bar) -> None:
        """Run one epoch.  If *record* is False stats calls are skipped (warmup)."""
        import gc
        for i, batch in enumerate(self.loader):
            if (i%25) == 0:
                gc.collect(2)
            if record:
                self.stats.start_step()

            loss, descr = self.step(i, epoch, batch, model_kwargs)

            if record:
                self.stats.stop_step()

                if self.enable_checkpointing and self.should_save_checkpoint(i):
                    self.stats.start_save_checkpoint()
                    self.save_checkpoint(i)
                    self.stats.stop_save_checkpoint()

                self.stats.log_loss(loss)
                self.stats.log_step()

            if descr is not None:
                progress_bar.set_description(descr)
            progress_bar.update(1)