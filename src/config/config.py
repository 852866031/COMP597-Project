"""This file contains the configurations for the project.
"""
from src.config.util.base_config import _Arg, _BaseConfig
from src.config.logging_config import LoggingConfig
import src.config.models as models_config
import src.config.trainers as trainers_config
import src.config.trainer_stats as trainer_stats_config
import src.config.data as data_config

class Config(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self.logging = LoggingConfig()
        self._arg_batch_size = _Arg(type=int, help="Size of batches", default=4)
        self._arg_learning_rate = _Arg(type=float, help="The learning rate for training. It is used by the optimizer for all models.", default=1e-6)
        self._arg_model = _Arg(type=str, help="Which model to train.")
        self.model_configs = models_config.ModelConfigs()
        self._arg_trainer = _Arg(type=str, help="How to train the model", default="simple")
        self.trainer_configs = trainers_config.TrainerConfigs()
        self._arg_trainer_stats = _Arg(type=str, help="Type of statistics to gather. By default it is set to no-op, which ignores everything.", default="no-op")
        self.trainer_stats_configs = trainer_stats_config.TrainerStatsConfigs()
        self._arg_data = _Arg(type=str, help="Dataset module to use to load data.", default="dataset")
        self.data_configs = data_config.DataConfigs()

