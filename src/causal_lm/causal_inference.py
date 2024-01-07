""" 
Example run, modelname is the key for the params in the config.json file
python3 causal_inference.py \
    --modelname causal-100-balanced \
    --config ../../config.json \

"""

import numpy as np
import torch
from peft import (
    PeftModel,
)
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import argparse
import json

from src.utils import setup_logger, get_evaluation_dataset


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
    model = AutoModelForCausalLM.from_pretrained(
        basemodel, quantization_config=bnb_config, device_map="auto"
    )
    model.config.pad_token_id = model.config.eos_token_id
    return model


def get_tokenizer(basemodel: str):
    tokenizer = AutoTokenizer.from_pretrained(basemodel)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_adapter(model, adapter_path) -> PeftModel:
    return PeftModel.from_pretrained(model, adapter_path)


def get_logits(model, tokenizer, prompt, iteration) -> tuple[int, tuple[float, float]]:
    """Gets the logits for the prompt and returns the prediction and the
    softmax scores for the two classes"""
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config["max_length"],
    ).to("cuda")
    classes = [0, 1]
    class_names = ["False", "True"]
    false_token_id = tokenizer.encode(class_names[0])[1]
    true_token_id = tokenizer.encode(class_names[1])[1]
    with torch.no_grad():
        outputs = model(input_ids).logits
        softmax = outputs[0, -1, :].softmax(dim=0)
        false_logit = outputs[0, -1, false_token_id]
        true_logit = outputs[0, -1, true_token_id]
        prediction = classes[np.array([false_logit.item(), true_logit.item()]).argmax()]
    if iteration % 200 == 0:  # log every 200 iterations
        logger.info(
            "top 10 tokens:"
            f" {[tokenizer.decode(output) for output in outputs[0, -1, :].topk(10).indices]}"
        )
    return prediction, (
        softmax[false_token_id].item(),
        softmax[true_token_id].item(),
    )


if __name__ == "__main__":
    # setup logger and config
    args = parse_args()
    config = read_config(args.config, args.modelname)
    logger = setup_logger(
        name="causal inference",
        logdir=config["logdir"],
        filename="causal_inference.log",
    )

    logger.info("\n\nStarting causal inference ==========================")
    for hyperparam, val in config.items():
        logger.info(f"{hyperparam}: {val}")

    # get model, tokenizer and dataset
    model = get_model(config["basemodel"])
    tokenizer = get_tokenizer(config["basemodel"])
    eval_dataset = get_evaluation_dataset(
        split=config["dataset"], prompt_template=config["proompt_template"]
    )

    # load adapter if available
    if "adapter_path" in config.keys():
        model = load_adapter(model, adapter_path=config["adapter_path"])

    logger.info("sample train example: ")
    logger.info(f"{eval_dataset['text'][0]}")

    # get logits
    y_true, y_pred, scores = [], [], []
    for i in tqdm(range(len(eval_dataset))):
        sample = eval_dataset[i]
        prediction, score = get_logits(model, tokenizer, sample["text"], i)
        y_true.append(sample["labels"])
        y_pred.append(prediction)
        scores.append(score)

    # log results
    logger.info(f"\n{classification_report(y_true, y_pred)}")
