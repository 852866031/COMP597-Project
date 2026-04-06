from typing import Any, Dict, Optional
import src.trainer.base as base
import src.config as config
import src.trainer.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class SimpleTrainer(base.Trainer):
    """Trainer for a simple iteration.

    This trainer implements a simple iteration step for a single device.

    Parameters
    ----------
    loader
        A PyTorch dataloader that will be used to obtain the data at each step.
    model
        The model to train.
    optimizer
        The PyTorch optimizer used to update the models weights.
    lr_scheduler
        A learning rate scheduler configured to work with the provided 
        optimizer.
    device
        The device on which the input batches will be moved.
    stats
        An object to gather statistics during training.

    Attributes
    ----------
    loader : torch.utils.data.DataLoader
        The object used to load data during training.
    model : torch.nn.Module
        The model to train as provided to the constructor.
    optimizer : torch.optim.Optimizer
        The optimizer used during training as provided to the constructor.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler used during training as provided to the 
        constructor.
    device : torch.device
        The device used to move the input batches as provided to the 
        constructor.
    stats : src.trainer.stats.TrainerStats
        The `TrainerStats` object used to gather statistics.

    """

    def __init__(self, 
                 loader : data.DataLoader, 
                 model : nn.Module, 
                 optimizer : optim.Optimizer, 
                 lr_scheduler : optim.lr_scheduler.LRScheduler, 
                 device : torch.device, 
                 stats : stats.TrainerStats,
                 conf: Optional[config.Config] = None):
        super().__init__(model, loader, device, stats)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # TODO remove conf as it is unused.
        self.conf = conf

    def checkpoint_dict(self, i: int) -> Dict[str, Any]:
        super_dict = super().checkpoint_dict(i)
        super_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        super_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        return super_dict
    
    def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        self.optimizer.zero_grad() #Zero the gradients
        outputs = self.model(**batch, **model_kwargs)
        return outputs.loss
    
    def backward(self, i: int, loss: torch.Tensor) -> None:
        loss.backward()
    
    def optimizer_step(self, i: int) -> None:
        self.optimizer.step()
        self.lr_scheduler.step()
