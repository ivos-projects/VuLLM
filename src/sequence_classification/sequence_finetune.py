import argparse
import os
from pathlib import Path

import evaluate
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    LlamaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.utils import (
    find_all_linear_names,
    get_dataset,
    get_quantization_config,
    read_config,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    return args


def get_model(basemodel: str):
    label2id = {"False": 0, "True": 1}
    id2label = {0: "False", 1: "True"}
    model = LlamaForSequenceClassification.from_pretrained(
        basemodel,
        quantization_config=get_quantization_config(),
        num_labels=2,
        device_map="auto",
        id2label=id2label,
        label2id=label2id,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = get_peft_config(model)
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False  # only used for generation
    return model


def get_tokenizer(basemodel: str):
    tokenizer = AutoTokenizer.from_pretrained(basemodel)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_peft_config(model):
    """Config for LoRA weights"""
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=find_all_linear_names(model),
        modules_to_save=["score"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    return peft_config


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def get_trainer(model, train_dataset, eval_dataset, config) -> Trainer:
    if config["max_eval_samples"]:  # for faster evalutaion
        eval_dataset = eval_dataset.select(range(config["max_samples"]))
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            report_to=config["report_to"],
            evaluation_strategy="epoch",
            run_name=config["run_name"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            warmup_steps=config["warmup_steps"],
            num_train_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            fp16=True,
            logging_steps=1,
            output_dir=config["adapter_path"],
            optim="paged_adamw_32bit",
            remove_unused_columns=False,  # this is important
        ),
    )
    return trainer


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "vullm"
    args = parse_args()
    config = read_config(args.config, args.modelname)
    logger = setup_logger(
        name="sequence finetune",
        logdir=config["logdir"],
        filename="sequence_classification_finetune.log",
    )

    logger.info("\n\nStarting sequence finetune ==========================")
    for hyperparam, val in config.items():
        logger.info(f"{hyperparam}: {val}")

    model = get_model(basemodel=config["basemodel"])
    tokenizer = get_tokenizer(basemodel=config["basemodel"])

    output_dir = Path(config["adapter_path"])
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    dataset = get_dataset(
        tokenizer,
        truncated_train=config["truncated_train"],
        balanced=config["balanced"],
        max_samples=config["max_samples"],
        max_length=config["max_length"],
        proomt_template=config["proompt_template"],
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["valid"]

    trainer = get_trainer(model, train_dataset, eval_dataset, config)
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # train the model
    train_result = trainer.train()
    metrics = train_result.metrics
    logger.info(f"metrics: {metrics}")

    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)

    print("Saving last checkpoint of the model...")
    trainer.model.save_pretrained(output_dir)
