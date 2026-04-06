from typing import Any, Dict, Optional, Tuple
import src.trainer.base as base
import src.trainer.stats.base as stats_base
import src.config as config
import tqdm.auto
import numpy as np
import gc as _gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
from torch_geometric.nn.models import PNA as _PNA

class PNATrainer(base.Trainer):
    def __init__(self,
                 loader : data.DataLoader,
                 dataset: torch.utils.data.Dataset,
                 model : nn.Module,
                 optimizer : optim.Optimizer,
                 lr_scheduler : optim.lr_scheduler.LRScheduler,
                 device : torch.device,
                 stats : stats_base.TrainerStats,
                 conf: config.Config):
        super().__init__(model, loader, device, stats)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.conf = conf
        self.TRAIN_mean, self.TRAIN_std = (
            self.mean(dataset).item(),
            self.std(dataset).item(),
        )
        self.num_batches = len(loader)

    def mean(self, dataset: torch.utils.data.Dataset):
        return np.mean([dataset.get(i).y for i in range(len(dataset))])

    def std(self,  dataset: torch.utils.data.Dataset):
        return np.std([dataset.get(i).y for i in range(len(dataset))])

    def checkpoint_dict(self, i: int) -> Dict[str, Any]:
        return {
            "step": i,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
        }

    def batch_size(self, x):
        return int(x.batch[-1] - x.batch[0] + 1)

    def forward(self, step: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        if self.conf.use3d:
            molecule_repr = self.model(z=batch.z, pos=batch.pos, batch=batch.batch)
        else:
            molecule_repr = self.model(
                x=batch.x.type(torch.float),
                batch=batch.batch,
                edge_index=batch.edge_index,
                batch_size=self.batch_size(batch),
            )
            molecule_repr = global_max_pool(molecule_repr, batch.batch)

        pred = molecule_repr.squeeze()

        B = pred.size()[0]

        # batch.y is a flat [B] tensor (one scalar target per molecule).
        # pred is also [B] after squeeze(), so both must be 1-D for L1Loss.
        y = batch.y.view(B)

        # normalize
        y = (y - self.TRAIN_mean) / self.TRAIN_std

        criterion = nn.L1Loss()

        return criterion(pred, y) # Our loss

    def backward(self, i: int, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()

    def optimizer_step(self, step: int, epoch : int) -> None:
        self.optimizer.step()

        self.lr_scheduler.step(epoch - 1 + step / self.num_batches)

    def process_batch(self, batch):
        return batch.to(self.device)

    def step(self, i : int, epoch : int, batch : Any, model_kwargs : Optional[Dict[str, Any]]) -> Tuple[torch.Tensor, Optional[str]]:
        if model_kwargs is None:
            model_kwargs = {}
        batch = self.process_batch(batch)

        self.stats.start_forward()
        loss = self.forward(i, batch, model_kwargs)
        self.stats.stop_forward()

        self.stats.start_backward()
        self.backward(i, loss)
        self.stats.stop_backward()

        self.stats.start_optimizer_step()
        self.optimizer_step(i, epoch)
        self.stats.stop_optimizer_step()

        return loss, f"loss: {loss:.4f}"


    def _run_epoch(self, epoch: int, model_kwargs: Dict[str, Any],
                   record: bool, progress_bar) -> None:
        """Run one epoch.  If *record* is False stats calls are skipped (warmup)."""
        for i, batch in enumerate(self.loader):
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

    def train(self, model_kwargs : Optional[Dict[str, Any]]) -> None:
        if model_kwargs is None:
            model_kwargs = {}
        _gc.enable()
        self.model.train()  # No eval ever

        warmup_bar = tqdm.auto.tqdm(range(self.num_batches), desc="warmup")
        self._run_epoch(epoch=0, model_kwargs=model_kwargs,
                        record=False, progress_bar=warmup_bar)
        warmup_bar.close()

        # ------------------------------------------------------------------
        # Measured epochs
        # ------------------------------------------------------------------
        progress_bar = tqdm.auto.tqdm(
            range(self.num_batches * self.conf.epochs), desc="loss: N/A"
        )

        self.stats.start_train()

        for epoch in range(1, self.conf.epochs + 1):
            self._run_epoch(epoch=epoch, model_kwargs=model_kwargs,
                            record=True, progress_bar=progress_bar)

        self.stats.stop_train()
        progress_bar.close()
        self.stats.log_stats()
