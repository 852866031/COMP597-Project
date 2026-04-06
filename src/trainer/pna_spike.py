# src/trainer/pna_spike.py
"""
PNASpikeTrainer — runs a warmup epoch, then a gc-on epoch, then a gc-off epoch
so that GC-induced timing spikes can be confirmed and attributed.

Inherits from PNATrainer and overrides only the train() method.
"""
from __future__ import annotations

import gc as _gc
from typing import Dict, Optional

import tqdm.auto

from src.trainer.pna_simple import PNATrainer


class PNASpikeTrainer(PNATrainer):
    """PNA trainer that records a GC-on run followed by a GC-off run.

    The ``stats`` object must be a ``PNATrainerSpikeStats`` instance (or any
    object implementing ``set_run_type(str)`` in addition to the standard
    ``TrainerStats`` interface).
    """

    def train(self, model_kwargs: Optional[Dict] = None) -> None:
        if model_kwargs is None:
            model_kwargs = {}

        self.model.train()

        # --- 1. Warmup (no recording) ---
        warmup_bar = tqdm.auto.tqdm(range(self.num_batches), desc="warmup")
        self._run_epoch(epoch=0, model_kwargs=model_kwargs,
                        record=False, progress_bar=warmup_bar)
        warmup_bar.close()

        # --- 2. GC-on run (all configured epochs) ---
        self.stats.set_run_type("gc_on")
        _gc.enable()
        self.stats.start_train()   # installs GC callback
        gc_on_bar = tqdm.auto.tqdm(
            range(self.num_batches * self.conf.epochs), desc="recording (GC on)"
        )
        for epoch in range(1, self.conf.epochs + 1):
            self._run_epoch(epoch=epoch, model_kwargs=model_kwargs,
                            record=True, progress_bar=gc_on_bar)
        gc_on_bar.close()
        self.stats.stop_train()    # removes GC callback

        # --- 3. GC-off run (all configured epochs) ---
        _gc.disable()
        self.stats.set_run_type("gc_off")
        gc_off_bar = tqdm.auto.tqdm(
            range(self.num_batches * self.conf.epochs), desc="recording (GC off)"
        )
        for epoch in range(1, self.conf.epochs + 1):
            self._run_epoch(epoch=epoch, model_kwargs=model_kwargs,
                            record=True, progress_bar=gc_off_bar)
        gc_off_bar.close()
        _gc.enable()

        self.stats.log_stats()
