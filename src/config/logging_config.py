from src.config.util.base_config import _Arg, _BaseConfig
import logging

class DefaultLoggingConfig:

    def __init__(self) -> None:
        self.level : str = "WARNING"
        self.format : str = "[{levelname:.4}] : {asctime} : {module:<24.24} : {message}"
        self.datefmt : str = "%Y-%m-%dT%H:%M:%S" # ISO 8601
        self.style : str = '{'

class LoggingConfig(_BaseConfig):
    level : str
    filename : str
    filemode : str
    format : str
    datefmt : str
    style : str

    def __init__(self) -> None:
        self._default = DefaultLoggingConfig()
        self._arg_level = _Arg(type=str, help="Logging level.", default=self._default.level)
        self._arg_filename = _Arg(type=str, help="Filename to write logs to. If not provided, logs will go to stdout.", default=None)
        self._arg_filemode = _Arg(type=str, help="Filemode to use when filename is provided.", default='a')
        self._arg_format = _Arg(type=str, help="Format to use to output logs.", default=self._default.format)
        self._arg_datefmt = _Arg(type=str, help="Date format to use in format.", default=self._default.datefmt)
        self._arg_style = _Arg(type=str, help="Python string format used in the format argument.", default=self._default.style)

