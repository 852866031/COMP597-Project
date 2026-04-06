# === import necessary modules ===
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class
import src.trainer.stats as trainer_stats # Trainer statistics module

# === import necessary external modules ===
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers

"""
This file contains the code to train a GPT-2 model using Simple trainer (src/trainer/simple.py).
It is based on the GPT-2 model from HuggingFace Transformers.
https://huggingface.co/docs/transformers/en/model_doc/gpt2
"""


def init_gpt2_tokenizer():
    """
    Initializes the GPT-2 tokenizer. This tokenizer is found in the HuggingFace Transformers library.
    Returns:
        transformers.PreTrainedTokenizer: The GPT-2 tokenizer.
    """
    # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Tokenizer 
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")

    # GPT-2 does not have a padding token, so we set it to the end of sequence token. This is not necessary for all models, adjust as needed.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def process_dataset(conf: config.Config, tokenizer: transformers.PreTrainedTokenizer, dataset: data.Dataset) -> data.Dataset:
    """
    Processes the dataset for training.
    Args:
        conf (config.Config): The configuration object.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        dataset (data.Dataset): The dataset to process.
    Returns:
        data.Dataset: The processed dataset.
    """
    def tokenize(examples):
        return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt") 
    dataset = dataset.map(tokenize, batched=True, num_proc=conf.model_configs.gpt2.tokenize_num_process) # Tokenize the dataset
    dataset = dataset.remove_columns(column_names=["text", "url", "timestamp"]) # Remove unnecessary columns
    return dataset

def init_gpt2_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    """
    Initializes the optimizer for the GPT-2 model.
    Args:
        conf (config.Config): The configuration object.
        model (nn.Module): The GPT-2 model.
    Returns:
        optim.Optimizer: The initialized optimizer.
    """
    # This is a simple AdamW optimizer with weight decay. Choose different optimizers as needed.
    # Note: The learning rate is taken from the configuration object. Adjust it as needed for different models and training setups based on the loss function.
    return optim.AdamW(model.parameters(), lr=conf.learning_rate)

def pre_init_gpt2(conf: config.Config, dataset: data.Dataset) -> Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForLanguageModeling]:
    """
    Prepares the GPT-2 model, dataset, tokenizer and data collator for training.
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForLanguageModeling]: The GPT-2 model, dataset, tokenizer and data collator.
    """
    tokenizer = init_gpt2_tokenizer()
    dataset = process_dataset(conf, tokenizer, dataset)

    # Based on https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # GPT-2 is a causal language model, so mlm is set to False.

    # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config
    model_config = transformers.GPT2Config() 
    # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2LMHeadModel
    model = transformers.GPT2LMHeadModel(config=model_config) # choose a GPT-2 model architecture from the HuggingFace library

    # Set the padding token id to the tokenizer's padding token id
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, dataset, tokenizer, data_collator

################################################################################
#################################    Simple    #################################
################################################################################

def simple_trainer(conf : config.Config, model : transformers.GPT2LMHeadModel, dataset : data.Dataset, tokenizer : transformers.PreTrainedTokenizer, data_collator : transformers.DataCollatorForLanguageModeling) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for GPT-2 model. Uses the SimpleTrainer from src/trainer/simple.py.
    Args:
        conf (config.Config): The configuration object.
        model (transformers.GPT2LMHeadModel): The GPT-2 model to train.
        dataset (data.Dataset): The dataset to train on.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        data_collator (transformers.DataCollatorForLanguageModeling): The data collator to use.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The simple trainer and a dictionary with additional options.
    """
    loader = data.DataLoader(dataset, batch_size=conf.batch_size, collate_fn=data_collator) # DataLoader for batching the dataset
    model = model.cuda() # Move the model to GPU
    optimizer = init_gpt2_optim(conf, model) # Initialize the optimizer for GPT-2
    scheduler = transformers.get_scheduler( # Linear learning rate decay scheduler
        "linear", 
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader), 
    )

    # Return the SimpleTrainer with the initialized components
    return trainer.SimpleTrainer(loader=loader, model=model, optimizer=optimizer, lr_scheduler=scheduler, device=model.device, stats=trainer_stats.init_from_conf(conf=conf, device=model.device, num_train_steps=len(loader))), None

################################################################################
##################################    Init    ##################################
################################################################################

def gpt2_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initializes the GPT-2 model and returns the appropriate trainer based on the configuration.
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The initialized trainer and a dictionary with additional options.
    """
    model, dataset, tokenizer, data_collator = pre_init_gpt2(conf, dataset)
    # Note: Currently, only Simple trainer is implemented for GPT-2. Add more trainers as needed.
    if conf.trainer == "simple": 
        return simple_trainer(conf, model, dataset, tokenizer, data_collator)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")

