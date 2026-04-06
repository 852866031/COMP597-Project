from src.config.util.base_config import _Arg, _BaseConfig

class ModelConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_tokenize_num_process = _Arg(type=int, help="Number of threads used to tokenize the dataset.", default=1)
