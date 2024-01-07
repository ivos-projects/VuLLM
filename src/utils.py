import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from datasets import concatenate_datasets, load_dataset
import datasets
import logging
from pathlib import Path
import json
import torch
import random


def get_max_length(model):
    max_length = None
    # Find maximum sequence length in the model configuration and save it in "max_length" if found
    for length_setting in [
        "n_positions",
        "max_position_embeddings",
        "seq_length",
    ]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    # Set "max_length" to 1024 (default value) if maximum sequence length is not found in the model configuration
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def random_classifier(weights, n):
    """
    Generate n random classifications based on the given weights.

    Parameters:
    - weights: A dictionary where keys are the classification labels
               and values are the corresponding weights.
    - n: The number of examples to classify.

    Returns:
    A list of n randomly classified examples.
    Example usage:
    weights = {'true': 0.1, 'false': 0.9}
    n_examples = 10
    result = random_classifier(weights, n_examples)
    """
    labels = list(weights.keys())
    probabilities = list(weights.values())
    # Normalize probabilities to sum to 1
    total_probability = sum(probabilities)
    probabilities = [p / total_probability for p in probabilities]
    # Generate n random classifications
    classifications = random.choices(labels, weights=probabilities, k=n)

    return classifications


def find_all_linear_names(model):
    """model should be a peft model"""
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


def create_prompt_formats(sample: dict, validation=False):
    system = "You are a helpful assitant that searches for vulnerabilites in code."
    user = (
        "Is the follwing code vulnerable or not? Answer with True or False: \n"
        f" {sample['input']}\n"
    )

    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user}[/INST] "
    if not validation:
        prompt += f"{str(bool(sample['output'])).strip()} </s>"
    sample["text"] = prompt
    return sample


def setup_logger(name: str, logdir: str, filename: str) -> logging.Logger:
    path = Path(logdir)
    if not path.exists():
        path.mkdir(parents=True)
    path = path / filename
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)  # add the file handler
    return logger


def read_config(config_path: str, modelname: str) -> dict:
    """Reads the config file and returns the config for the modelname"""
    with open(config_path) as f:
        config = json.load(f)
    assert modelname in config.keys(), f"{modelname} not in config"
    return config[modelname]


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_evaluation_dataset(split: str, prompt_template: bool) -> datasets.Dataset:
    """split is one of valid or test"""
    dataset = load_dataset("oscaraandersson/reveal")
    split_dataset = dataset[split]
    if prompt_template:
        split_dataset = split_dataset.map(
            create_prompt_formats,
            fn_kwargs={"validation": True},
            load_from_cache_file=False,
        )
    split_dataset = split_dataset.rename_column("output", "labels")
    return split_dataset


def get_dataset(
    tokenizer,
    truncated_train=True,
    balanced=True,
    max_samples=1000,
    max_length=1024,
    proomt_template=False,
):
    """returns a tokenized huggingface dataset for training
    columns must include 'input_ids', 'attention_mask', 'labels'
    """

    def tokenizer_function_truncated(examples, column_to_tokenize):
        return tokenizer(
            examples[column_to_tokenize],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    def tokenizer_function_nontrunc(examples, column_to_tokenize):
        return tokenizer(
            examples[column_to_tokenize],
            max_length=max_length,
            padding="max_length",
            truncation=False,
        )

    column_to_tokenize = "input"

    dataset = load_dataset("oscaraandersson/reveal")

    if proomt_template:
        dataset["train"] = dataset["train"].map(
            create_prompt_formats,
            fn_kwargs={"validation": False},
            load_from_cache_file=False,
        )
        dataset["valid"] = dataset["valid"].map(
            create_prompt_formats,
            fn_kwargs={"validation": True},
            load_from_cache_file=False,
        )
        column_to_tokenize = "text"

    dataset = dataset.rename_column("output", "labels")

    if truncated_train:
        train_dataset = dataset["train"].map(
            tokenizer_function_truncated,
            fn_kwargs={"column_to_tokenize": column_to_tokenize},
            batched=True,
            remove_columns=list(set(["input", column_to_tokenize])),
        )
    else:
        train_dataset = dataset["train"].map(
            tokenizer_function_nontrunc,
            fn_kwargs={"column_to_tokenize": column_to_tokenize},
            batched=True,
            remove_columns=list(set(["input", column_to_tokenize])),
        )

        train_indices = [
            i
            for i in range(len(train_dataset))
            if len(train_dataset[i]["input_ids"]) <= max_length
        ]
        train_dataset = train_dataset.select(train_indices)

    dataset["valid"] = dataset["valid"].map(
        tokenizer_function_truncated,
        fn_kwargs={"column_to_tokenize": column_to_tokenize},
        batched=True,
        remove_columns=list(set(["input"])),
    )

    if balanced:
        n_positives = sum(train_dataset["labels"])
        n_samples = min(n_positives, max_samples // 2)  # samples for each class
        positive_samples = train_dataset.filter(
            lambda example: example["labels"] == 1, keep_in_memory=True
        ).select(range(n_samples))
        negative_samples = train_dataset.filter(
            lambda example: example["labels"] == 0, keep_in_memory=True
        ).select(range(n_samples))
        dataset["train"] = concatenate_datasets([positive_samples, negative_samples])
    else:
        n_samples = min(len(train_dataset), max_samples)
        dataset["train"] = train_dataset.select(range(n_samples))
    return dataset
