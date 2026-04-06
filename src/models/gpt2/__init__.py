# === import necessary modules ===
from src.models.gpt2.gpt2 import gpt2_init
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class

# === import necessary external modules ===
from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "gpt2"

def init_model(conf : config.Config, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    return gpt2_init(conf, dataset)
