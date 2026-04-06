# Programming Guide

This document attemps to provide guidelines to add code in the repository. Make sure to read through it carefully to make it easier to add the code for your project.

## Auto-discovery

Auto-discovery is the feature that is used so that you can easily add your components to this code without having to change the provided code.

The idea behind auto-discovery is that we provide you with where to put your code and what it should contain, and if you do so, the provided code in this repository will be able to pick up your components without you modifying the provided code. Typically, auto-discovery will ask the following:

| Category | Requirement | Description |
| :--- | :---: | :--- |
| Location | Required | We will ask you to place your code in a specific location. You will either have to create a file or a directory in a specific location. |
| Name | Optional | Your code will have the option to provide a name to register your component. The auto-discovery module will always default to using the name of your directory/file if you do not provide a name. To provide a name, your module will need to have an attribute (a module variable) containing this name. The name of that attribute will depend on the component you are adding. |
| Attribute | Required | We will ask that your code provides a certain attribute that fulfills some requirements. It might be a function, or a class with a certain name. |

Here is a summary of what will be detailed in the sections below:

| Extensions | Location | Name | Attribute |
| :--- | :--- | :--- | :--- |
| Models | Dedicated subdirectory to `src/models/` | `model_name` | Function with signature `init_model(conf : src.config.Config, dataset : torch.utils.data.Dataset) -> Tuple[src.trainer.Trainer, Optional[Dict[str, Any]]]` |
| Data configuration | Dedicated subdirectory to `src/config/data/` | `config_name` | Class named `DataConfig` which extends `src.config.util.base_config._BaseConfig` with no input to the `__init__` method. |
| Models configuration | Dedicated subdirectory to `src/config/models/` | `config_name` | Class named `ModelConfig` which extends `src.config.util.base_config._BaseConfig` with no input to the `__init__` method. |
| Trainer stats configuration | Dedicated subdirectory to `src/config/trainer_stats/` | `config_name` | Class named `TrainerStatsConfig` which extends `src.config.util.base_config._BaseConfig` with no input to the `__init__` method. |
| Trainer configuration | Dedicated subdirectory to `src/config/trainers/` | `config_name` | Class named `TrainerConfig` which extends `src.config.util.base_config._BaseConfig` with no input to the `__init__` method. |
| Data load function | Dedicated subdirectory to `src/data/` | `data_load_name` | Function with signature `load_data(conf : src.config.Config) -> torch.utils.data.Dataset` |
| Trainer stats | Dedicated file in the `src/trainer/` directory | `trainer_stats_name` | Function with signature `contruct_trainer_stats(conf : src.config.Config, **kwargs) -> src.trainer.stats.TrainerStats` |

## Adding New Models

To add a model, you will need to create a dedicated directory under `src/models/` (give a name that matches your machine learning model). The directory you created will become a submodule of the models module. We recommend that you use a structure similar to below:

```
src/models/<model_name>
├── __init__.py
├── model.py
```

Where `__init__.py` will make sure a function with signature `init_model(conf : src.config.Config, dataset : torch.utils.data.Dataset) -> Tuple[src.trainer.Trainer, Optional[Dict[str, Any]]]` is available and optionally provides a name for your model by setting the variable `model_name` to the desired name. `model.py` contains the basics to initialize your model and creates a trainer for that model. Please take a look at the GPT2 example provided at `src/models/gpt2/`. Once this is done, you will be able to use it with `python3 launch.py --model <model_name>`. In simple steps:

1. Create directory `src/models/<model_name>`.
2. Create file `src/models/<model_name>/__init__.py`
3. Create file `src/models/<model_name>/model.py`
4. Add the following to `src/models/<model_name>/__init__.py`
   ```python
   # === import necessary modules ===
   import src.models.<model_name>.model
   import src.config as config # Configurations
   import src.trainer as trainer # Trainer base class

   # === import necessary external modules ===
   from typing import Any, Dict, Optional, Tuple
   import torch.utils.data as data

   model_name = "<model_name>"

   def init_model(conf : config.Config, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
       pass
   ```
5. Write your code in `src/models/<model_name>/model.py`.
6. Complete the implementation of `init_model(...)`

If your model needs specific configurations that should be provided by the config object provided to the init method, see the [relevant documentation](#additional-models-configurations).

## Adding Configurations

The root configuration object defining the structure is the `Config` class defined in `src/config/config.py`. Find below how to write a configuration object, and how to add it to the configuration structure.

### Defining a configuration object

The configuration objects in this project are designed to integrate with the `argparse` module provided by Python. This is done using two classes: `_Arg` and `_BaseConfig`, which are defined in `src/config/utils/base_config.py`. 

#### Using `_BaseConfig`

The `_BaseConfig` class provides two public methods: `add_arguments` and `parse_arguments`. 

The way it works is that any attributes of itself with a name like `_arg_<name>` (starting with `_arg_`) is treated as an argument and will be added to the parser by `add_arguments`, as an argument available with `--<prefix><name>`, where `<prefix>` is computed at runtime to allow for a structure and `<name>` is the desired argument. 

Then, `parse_arguments` takes the value of the flag `--<prefix><name>` and makes the value available in the config by creating an attribute named `<name>` set to the flag's value.

#### Using `_Arg`

The `_Arg` class simply encapsulates the inputs to the [`add_argument`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument) method of an [`ArgumentParser`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser) object from the `argparse` Python module, with the exception of the argument's name. You can extend the class to make more complex arguments. 

#### Example usage

Find below a simplified example:

```python
class SchoolConfig(_BaseConfig):
    def __init__(self):
        super().__init__()
        self._arg_name = _Arg(type=str, help="Name of the school", default="")
        self._arg_major = _Arg(type=str, help="Major studied at school", default="")

class StudentConfig(_BaseConfig):
    def __init__(self):
        super().__init__()
        self._arg_name = _Arg(type=str, help="Student's name")
        self._arg_age = _Arg(type=int, help="Student's age")
        self.school = SchoolConfig()
parser = argparse.ArgumentParser()
conf = StudentConfig()
conf.add_arguments(parser)
args, _ = parser.parse_known_args()
conf.parse_arguments(args)
```

The example above would allow using the flags:

* `--name`
* `--age`
* `--school.name`
* `--school.major`

Notice how the `school.` prefix would be automatically added to the sub-configuration. This allows more complex structures while maintaining flag name uniqueness. Unfortunately, it leads to lengthy flag names. 

### Additional data configurations

Maybe you will need an additional component under `src/data/` to load your dataset, and you would like to provide configuration options to your component. To do so, you should create a configuration class under `src/config/data`. Please follow the steps below:

1. Create the directory `src/config/data/<component_name>`
2. Create the file `src/config/data/<component_name>/config.py` where you implement a class named `DataConfig` that extends `_BaseConfig`. This class should contain all the configurations you would like to make available as arguments following the [steps](#defining-a-configuration-object) to create a configuration object.
3. Create the file `src/config/data/<component_name>/__init__.py` which will import the config class you just created. It can also define a variable `config_name` if you would like the prefix to be different from the directory name.

Your arguments should become available with the `--data_configs.<component_name>` prefix. You can check with `python3 launch.py --help` whether you can see your flags. 

### Additional models configurations

Maybe you will need to make your model configurable using arguments on the command line. If you wish to add configurations for your model, you should create a configuration class under `src/config/models`. Please follow the steps below:

1. Create the directory `src/config/models/<model_name>`
2. Create the file `src/config/models/<model_name>/config.py` where you implement a class named `ModelConfig` that extends `_BaseConfig`. This class should contain all the configurations you would like to make available as arguments following the [steps](#defining-a-configuration-object) to create a configuration object.
3. Create the file `src/config/models/<model_name>/__init__.py` which will import the config class you just created. It can also define a variable `config_name` if you would like the prefix to be different from the directory name.

Your arguments should become available with the `--model_configs.<model_name>` prefix. You can check with `python3 launch.py --help` whether you can see your flags. 

### Additional trainer stats configurations

You might want to collect data differently, which will involve [adding a trainer stats class](#measurements). If this added class needs to be configurable at runtime, you should make a dedicated configuration class for it under `src/config/trainer_stats`. Please follow the steps below: 

1. Create the directory `src/config/trainer_stats/<trainer_stats_name>`
2. Create the file `src/config/trainer_stats/<trainer_stats_name>/config.py` where you implement a class named `TrainerStatsConfig` that extends `_BaseConfig`. This class should contain all the configurations you would like to make available as arguments following the [steps](#defining-a-configuration-object) to create a configuration object.
3. Create the file `src/config/trainer_stats/<trainer_stats_name>/__init__.py` which will import the config class you just created. It can also define a variable `config_name` if you would like the prefix to be different from the directory name.

Your arguments should become available with the `--trainer_stats_configs.<trainer_stats_name>` prefix. You can check with `python3 launch.py --help` whether you can see your flags. 

### Additional trainers configurations

Finally, you might want/need [additional trainer classes](#trainers). If those classes require specific configurations, you can make them available by creating a configuraiton class for them under `src/config/trainers`. Please follow the steps below:

1. Create the directory `src/config/trainers/<trainer_name>`
2. Create the file `src/config/trainers/<trainer_name>/config.py` where you implement a class named `TrainerConfig` that extends `_BaseConfig`. This class should contain all the configurations you would like to make available as arguments following the [steps](#defining-a-configuration-object) to create a configuration object.
3. Create the file `src/config/trainers/<trainer_name>/__init__.py` which will import the config class you just created. It can also define a variable `config_name` if you would like the prefix to be different from the directory name.

Your arguments should become available with the `--trainer_configs.<trainer_name>` prefix. You can check with `python3 launch.py --help` whether you can see your flags. 

## Data

To add a new data loading function, you will need to create a dedicated directory under `src/data/`. Give it a name that relates to your dataset, or a specific library you are using to load the data for example. It will become a submodule of the data module. You could use a structure similar to the one below:

```
src/models/<name>
├── __init__.py
├── data.py
```

You can check the submodule at `src/data/dataset` for an example. You module should provide a function with the signature `def load_data(conf : config.Config) -> torch.utils.data.Dataset`. Additionally, you can configure the name that will be available to select this function when using the `--data` flag. By default, the name assigned will be the name of the directory, but you can change it if your module provides the `data_load_name` variable, in which case it will use the content of the variable as the name. Here is a step by step process:

1. Create directory `src/data/<name>`.
2. Create file `src/data/<name>/__init__.py`.
3. Create file `src/data/<name>/data.py`.
4. Add the following to `src/data/<name>/__init__.py`.
   ```python
   from src.data.<name>.data import *
   ```
5. Add the following to `src/data/<name>/data.py`.
   ```python
   import src.config as config
   import torch.utils.data

   data_load_name="<your_name>"

   def load_data(conf : config.Config) -> torch.utils.data.Dataset:
       # Your implementation here
       pass
   ```

## Trainers

The provided trainer might not suit your model, in which case you will need to add a new one. There is no auto-discovery here, as there is no factory to create a trainer. Your implementation under `src/models/<your_model>/` will simply import whatever trainer it needs. For the sake of organization though, the code for your trainer should be stored under `src/trainer/`. 

Before creating a new trainer, you should identify what it is that you need from it. For example, if the only reason preventing you from using the provided simple trainer under `src/trainer/simple.py` is the `process_batch` method, then you should probably use an implementation similar to this one:

```python
import src.trainer.simple as simple

class MyTrainer(simple.Simple):

    def __init__(...):
        # Your implementation
        pass

    def process_batch(self, i : int, batch : Any) -> Any:
        # Your implementation
        pass
```

This way, you will keep a lean implementation and avoid reimplementing code that was already available. 

## Measurements

Measurement techniques will likely vary for each model. As such, you might need to make your own tools or extensions to tools in order to collect meaningful data. 

Metrics can be collected during the training of a model using the `TrainerStats` class. If you need a specific way to perform measurements, you should create a class that extends the `TrainerStats` base class (or one of the already provided classes which extend it). To do so, create a python file under `src/trainer/stats/`, and give a name appropriate to what you are doing or the tools you are using. Here's what your file should look like:

```python
import src.config as config
import src.trainer.stats.base as base

trainer_stats_name="<your_name>"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    # Handle additional configurations here
    return YourTrainerStats(...)

class YourTrainerStats(base.TrainerStats):

    def __init__(...):
        pass

    # Override any method from base.TrainerStats that you need.

```

The function signature `def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats` is really important, as this is what will be used to construct the TrainerStats object. You can use the `trainer_stats_name` variable to configure the name by which it will be available using the `--trainer_stats` flag. If the variable is not provided, it should default to the name of the file (without the `.py` extension).

## Logging

The Python standard library provides a logging library. It can easily be used by adding the following code at the top of any Python file where logging is needed:

```python
import logging
logger = logging.getLogger(__name__)
```

The code provided already provides flags for basic configurations of the logging. Simply use the `--help` flag to print the available flags, which are of the format `--logging.*`. You can refer directly to the official [Python documentation](https://docs.python.org/3/library/logging.html#logging.basicConfig) for additional details about each flag. Each flag corresponds to an argument of the `logging.basicConfig(...)` function.
