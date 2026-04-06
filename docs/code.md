# Code Documentation

Most of this code is documented using doc strings. The [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style was followed as much as possible. Below, we give an overview of each core module in the repository. 

## Table of contents

<!-- no toc -->
1. [Table of contents](#table-of-contents)
2. [`config`](#config)
3. [`data`](#data)
4. [`models`](#models)
5. [`trainer`](#trainer)
   1. [`trainer.stats`](#trainerstats)
6. [`auto_discovery`](#auto_discovery)

## `config`

> PATH=`src/config`

This module contains the configuration classes used to configure experiments. A roughly one-to-one mapping is created between the configuration objects and the arguments that can be passed to the program. Some of the configuration objects are wrappers that automatically detect subconfigurations and add them to the configuration structure. 
### Automatic sub-configurations

> PATH=`src/config/data` <br>
> PATH=`src/config/models` <br>
> PATH=`src/config/trainer_stats` <br>
> PATH=`src/config/trainers`

All the paths above are directories where additional configuration classes can be added and they will be automatically added to the configuration structure at runtime. 

### `config.utils`

> PATH=`src/config/utils`

This submodule provides utilities to the configurations. There are three important classes:

| Class | Description |
| :--- | :--- |
| `_Arg` | The basic argument class that can be extended. It is used for automatic argument detection and parsing. |
| `_BaseConfig` | The basic configuration class. It integrates with the standard library module `argparse` to map arguments to configuration attributes. All configuration classes extend this class. |
| `ConfigAutoDiscovery` | This class is used to automatically detect additional configuration classes.  |

## `data`

> PATH=`src/data`

This module provides the means to load a dataset into memory. It is meant to obtain a [`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) object. It is one of those modules that integrates the [auto-discovery](#auto_discovery) feature. Submodules can be added to provide additional functions to load datasets, which will be automatically detected and imported at runtime. Please find details about adding dataset load functions in the [dedicated documentation](code_extensions.md#data).

The starter code provided contains the `dataset` submodule, which provides a small wrapper to Hugging Face's [datasets](https://huggingface.co/docs/datasets/index).

## `models`

> PATH=`src/models`

This modules provides the means to create/load a machine learning model and returns a trainer to train it. It is one of those modules that integrates the [auto-discovery](#auto_discovery) feature. Every submodule should provide the means the train one model such that the auto-discovery feature can register it. Please find details about adding models in the [dedicated documentation](code_extensions.md#models).

## `trainer`

> PATH=`src/trainer`

This modules provides the interface and classes to train machine learning models. The expectation is that class meant to train a model should extend the `Trainer` abstract class provided by at `src/trainer/base.py`. Please find details about adding trainers in the [dedicated documentation](code_extensions.md#trainers).

### `trainer.stats`

> PATH=`src/trainer/stats`

This module provides a set of classes that trainers can use to collect metrics during the training of machine learning models. This module also integrates the auto-discovery feature so that additional classes can be added. Any class meant to be used to collect metrics from a trainer should implement the extend the `TrainerStats` abstract class provided at `src/trainer/stats/base.py`. Please find details about adding models in the [dedicated documentation](code_extensions.md#measurements). Below are the already provided measurements classes:

| Class | Description |
| :--- | :--- |
| `NOOPTrainerStats` | Performs no measurements. Used as a default for transparent usage when no metrics are to be collected. |
| `SimpleTrainerStats` | Performs simple timing of the execution time of each step and substeps. |
| `CodeCarbonStats` | Performs time and energy measurements of the GPU during training. |

## `auto_discovery`

> PATH=`src/auto_discovery`

This module provides the means to import additional modules at runtime. This allows adding features without modifying any of the original files of the repository. It will allow a transparent merge of all the sub-projects that will be completed during the semester. Under no circumstances should this module be changed.
