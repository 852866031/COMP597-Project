import src.config.util.config_auto_discovery as config_auto_discovery

class TrainerStatsConfigs(config_auto_discovery.ConfigAutoDiscovery):

    def __init__(self) -> None:
        super().__init__(__package__, list(__path__), "TrainerStatsConfig", "config_name")

