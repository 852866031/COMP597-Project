from __future__ import annotations

import os
from src.config.util.base_config import _Arg, _BaseConfig

config_name = "pna"


class ModelConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()

        # ========================
        # Model behavior
        # ========================

        self._arg_model = _Arg(
            type=str,
            default="pna",
            help="GNN architecture key from bench.models.models",
        )

        self._arg_use3d = _Arg(
            action="store_true",
            default=False,
            help="Use 3D coordinates (pos,z) in forward pass",
        )

        self._arg_seed = _Arg(
            type=int,
            default=1234,
            metavar="S",
            help="Random seed",
        )

        self._arg_device_index = _Arg(
            type=int,
            default=0,
            help="GPU index passed to accelerator.fetch_device()",
        )

        # ========================
        # Dataset knobs
        # ========================

        self._arg_data_root = _Arg(
            type=str,
            default=os.getenv("MILABENCH_DIR_DATA", ""),
            help="Dataset root directory (parent of pcqm4m_v2/).",
        )

        self._arg_num_samples = _Arg(
            type=int,
            default=100000,
            help="Number of samples used from PCQM4Mv2Subset",
        )

        self._arg_split = _Arg(
            type=str,
            default="train",
            choices=["train", "val", "test", "holdout"],
            help="Dataset split to use",
        )

        # ========================
        # Training loop
        # ========================

        self._arg_batch_size = _Arg(
            type=int,
            default=256,
            metavar="N",
            help="Training batch size",
        )

        self._arg_epochs = _Arg(
            type=int,
            default=5,
            metavar="N",
            help="Number of training epochs",
        )

        self._arg_lr = _Arg(
            type=float,
            default=1e-6,
            metavar="LR",
            help="Learning rate",
        )

        self._arg_num_workers = _Arg(
            type=int,
            default=0,
            help="PyG DataLoader workers",
        )

        self._arg_weight_decay = _Arg(
            type=float,
            default=0.0,
            help="Adam weight_decay (default matches milabench: 0.0)",
        )

        self._arg_trainer_stats = _Arg(
            type=str,
            default="simple",
            help="Type of statistics to gather during training.",
        )

        # ========================
        # Checkpointing
        # ========================

        self._arg_enable_checkpointing = _Arg(
            action="store_true",
            default=False,
            help="Enable checkpointing during training",
        )

        self._arg_checkpoint_frequency = _Arg(
            type=int,
            default=1,
            help="Save checkpoint every N steps (only if enable_checkpointing)",
        )