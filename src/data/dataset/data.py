import datasets
import src.config as config
import torch.utils.data

data_load_name="dataset"

def load_data(conf : config.Config) -> torch.utils.data.Dataset:
    """Simple function to load a dataset based on the provided config object.
    """
    train_files = None
    if conf.data_configs.dataset.train_files is not None and conf.data_configs.dataset.train_files != "":
        train_files = {"train": conf.data_configs.dataset.train_files}
    return datasets.load_dataset(conf.data_configs.dataset.name, data_files=train_files, split=conf.data_configs.dataset.split, num_proc=conf.data_configs.dataset.load_num_proc)
