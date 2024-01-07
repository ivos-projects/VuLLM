import argparse
import json

import torch
from peft import PeftModel
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForSequenceClassification,
)

from src.utils import get_evaluation_dataset, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    return args


def read_config(config_path, modelname) -> dict:
    """Reads the config file and returns the config for the modelname"""
    with open(config_path) as f:
        config = json.load(f)
    assert modelname in config.keys(), f"{modelname} not in config"
    return config[modelname]


def get_bnb_config() -> BitsAndBytesConfig:
    """Config for 4bit quantization"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_model(basemodel: str):
    bnb_config = get_bnb_config()
    model = LlamaForSequenceClassification.from_pretrained(
        basemodel,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.pad_token_id = model.config.eos_token_id
    return model


def get_tokenizer(basemodel: str):
    tokenizer = AutoTokenizer.from_pretrained(basemodel)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_adapter(model, adapter_path) -> PeftModel:
    return PeftModel.from_pretrained(model, adapter_path)


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config, args.modelname)
    logger = setup_logger(
        name="sequence inference",
        logdir=config["logdir"],
        filename="sequence_classification_inference.log",
    )

    logger.info("\n\nStarting sequence inference ==========================")
    for hyperparam, val in config.items():
        logger.info(f"{hyperparam}: {val}")

    # get model and tokenizer
    model = get_model(basemodel=config["basemodel"])
    tokenizer = get_tokenizer(basemodel=config["basemodel"])
    eval_dataset = get_evaluation_dataset(
        split=config["dataset"], prompt_template=config["proompt_template"]
    )

    # add adapter if specified
    if "adapter_path" in config.keys():
        model = load_adapter(model, adapter_path=config["adapter_path"])

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(eval_dataset))):
            input_ids = tokenizer.encode(
                eval_dataset[i]["input"],
                return_tensors="pt",
                truncation=True,
                max_length=config["max_length"],
            ).to("cuda")
            logits = model(input_ids).logits
            prediction = logits.argmax().item()
            y_true.append(eval_dataset["labels"][i])
            y_pred.append(prediction)

    logger.info("Classification Report:")
    logger.info(classification_report(y_true, y_pred, digits=4))
