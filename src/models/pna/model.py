# === import necessary modules ===
import random
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class
import src.trainer.stats as trainer_stats # Trainer statistics module

# === import necessary external modules ===
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch
import torch.utils.data as data
from types import SimpleNamespace as NS
from torch_geometric.nn.models import PNA as _PNA
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import numpy as np

model_name = "pna"


def create_model(sample, degree):
    out_channels = 1
    if hasattr(sample.y, "shape") and len(sample.y.shape) > 1:
        out_channels = sample.y.shape[-1]

    _, in_channels = sample.x.shape

    return NS(
        category="2d",
        model=_PNA(
            # Basic GCNN setup
            in_channels=in_channels, 
            out_channels=out_channels,
            hidden_channels=64,
            num_layers=64,
            # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html
            aggregators=['mean', 'min', 'max', 'std'],
            scalers=['identity', 'amplification', 'attenuation'],
            # Histogram of in-degrees of nodes in the training set, used by scalers to normalize
            deg=degree(),
        ),
    )


class PNAModel(nn.Module):
    """
    Class wrapper around torch_geometric.nn.models.PNA using the exact
    hyperparams from milabench's PNA() factory.

    Notes:
      - Expects node features `x` with shape [num_nodes, in_channels].
      - Returns per-node outputs; you typically apply a global pool outside
        (milabench does global_max_pool in the training script).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deg,  # in-degree histogram tensor (torch.long)
        hidden_channels: int = 64,
        num_layers: int = 64,
        aggregators=None,
        scalers=None,
        device=None,
    ):
        super().__init__()
        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["identity", "amplification", "attenuation"]

        self.net = _PNA(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
        )
        self.device = device

    @classmethod
    def from_sample(cls, sample, degree_fn, out_channels: int | None = None, device=None, **kwargs):
        """
        Build the model using the same inference logic as the original function:
          - in_channels from sample.x.shape[1]
          - out_channels from sample.y if it is multi-dim; otherwise 1
          - deg from degree_fn()
        """
        _, in_channels = sample.x.shape

        if out_channels is None:
            out_channels = 1
            if hasattr(sample, "y") and hasattr(sample.y, "shape") and len(sample.y.shape) > 1:
                out_channels = sample.y.shape[-1]

        deg = degree_fn()
        return cls(in_channels=in_channels, out_channels=out_channels, deg=deg, device=device, **kwargs)

    def forward(self, x, edge_index, batch=None, batch_size=None, **kwargs):
        """
        Match milabench training call style:
          model(x=float(batch.x), batch=batch.batch, edge_index=batch.edge_index, batch_size=...)
        PyG's PNA model accepts x, edge_index, batch, batch_size; we pass through.
        """
        return self.net(x=x, edge_index=edge_index, batch=batch, batch_size=batch_size)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

################################################################################
#################################   Helper   #################################
################################################################################
def train_degree(train_dataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        try:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
        except TypeError:
            pass

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        try:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        except TypeError:
            pass

    return deg

def mean(self):
    return np.mean([self.get(i).y for i in range(len(self))])

def std(self):
    return np.std([self.get(i).y for i in range(len(self))])

def process_dataset(conf: config.Config, dataset: data.Dataset):
    # Degree histogram
    deg = train_degree(dataset)

    # Sample for shape inference
    sample = dataset[0]

    # Target normalization
    train_mean = float(mean(dataset))
    train_std = float(std(dataset))

    # IMPORTANT: must be PyG DataLoader
    dataloader: data.DataLoader = DataLoader(
        dataset,
        batch_size=int(conf.batch_size),
        shuffle=True,
        num_workers=int(conf.num_workers),
    )

    return dataloader, sample, deg, train_mean, train_std



################################################################################
##################################    Init    ##################################
################################################################################

def model_init(conf: config.Config, train_dataset: torch.utils.data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initializes the PNA model and returns the appropriate trainer based on the configuration.

    Reads all hyperparameters from the already-parsed global config object
    (conf.model_configs.pna.*) — no re-parsing of sys.argv.

    Args:
        conf: The fully-parsed global configuration object.
        train_dataset: The dataset to use for training.
    Returns:
        (PNATrainer, None)
    """
    # Use the already-parsed PNA model config — no extra argparse needed.
    pna_conf = conf.model_configs.pna

    # Reproducibility
    random.seed(pna_conf.seed)
    np.random.seed(pna_conf.seed)
    torch.manual_seed(pna_conf.seed)
    torch.backends.cudnn.deterministic = True

    # Build degree histogram and a representative sample for shape inference
    degree_hist = train_degree(train_dataset)
    sample = next(iter(train_dataset))

    info = create_model(
        sample=sample,
        degree=lambda: degree_hist,
    )

    # Device: respect config, fall back to CPU when CUDA is unavailable
    device = (
        torch.device(f"cuda:{pna_conf.device_index}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = info.model.to(device)

    # Optimizer: use PNA-specific lr and weight_decay from config
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=pna_conf.lr,
        weight_decay=pna_conf.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pna_conf.epochs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=pna_conf.batch_size,
        shuffle=True,
        num_workers=pna_conf.num_workers,
    )

    stats = trainer_stats.init_from_conf(conf=conf, device=device)

    # Select trainer class based on --trainer flag (mirrors GPT2 pattern)
    trainer_type = str(conf.trainer).strip().lower()
    if trainer_type == "pna_spike":
        from src.trainer.pna_spike import PNASpikeTrainer
        TrainerClass = PNASpikeTrainer
        print("Using PNA spike trainer — separate GC-on and GC-off runs to confirm GC-induced timing spikes.")
    elif trainer_type == "pna":
        from src.trainer.pna_measurement import PNAMeasurementTrainer
        TrainerClass = PNAMeasurementTrainer
        print("Using PNA measurement trainer — GC is disabled during training and gen-2 collections are forced between epochs to keep GC pauses outside measurement windows.")
    else:
        # pna_simple (and any other value) → plain GC-on trainer
        from src.trainer.pna_simple import PNATrainer
        TrainerClass = PNATrainer
        print("Using simple PNA trainer — no GC suppression, no separate GC-on/GC-off runs.")

    print(f"Model initialized ({trainer_type}) — trainable parameters: {count_trainable_parameters(model):,}")

    pna_trainer = TrainerClass(
        loader=train_loader,
        model=model,
        dataset=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        stats=stats,
        conf=pna_conf,
    )
    return pna_trainer, None