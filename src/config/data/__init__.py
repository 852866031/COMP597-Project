import src.config.util.config_auto_discovery as config_auto_discovery

class DataConfigs(config_auto_discovery.ConfigAutoDiscovery):

    def __init__(self) -> None:
        super().__init__(__package__, list(__path__), "DataConfig", "config_name")

