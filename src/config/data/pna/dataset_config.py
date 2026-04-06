# src/config/data/pna_dataset/pna_dataset_config.py
from __future__ import annotations
from src.config.util.base_config import _Arg, _BaseConfig

config_name = "pna_dataset"

class DataConfig(_BaseConfig):
    def __init__(self) -> None:
        super().__init__()

        self._arg_pna_data_root = _Arg(
            type=str,
            default="/home/slurm/comp597/students/jchen213/pna/dataset",
            help="Dataset root directory (parent of pcqm4m_v2).",
        )

        self._arg_num_samples = _Arg(
            type=int,
            default=10000,
            help="Number of samples to use from PCQM4Mv2Subset.",
        )

        self._arg_split = _Arg(
            type=str,
            default="train",
            choices=["train", "val", "test", "holdout"],
            help="Which split to use for PCQM4Mv2Subset.",
        )

        self._arg_backend = _Arg(
            type=str,
            default="sqlite",
            choices=["sqlite"],
            help="Storage backend for PyG dataset.",
        )