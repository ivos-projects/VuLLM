""" 
Example run, modelname is the key for the params in the config.json file
python3 causal_inference.py \
    --modelname causal-100-balanced \
    --config ../../config.json \

"""

import argparse
import os
from pathlib import Path

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils import (
    find_all_linear_names,
    get_dataset,
    read_config,
    setup_logger,
    get_quantization_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    return args


def get_model(basemodel: str):
    model = AutoModelForCausalLM.from_pretrained(
        basemodel,
        quantization_config=get_quantization_config(),
        device_map="auto",
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
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=find_all_linear_names(model),
        modules_to_save=["lm_head"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return peft_config


# keep only the input_ids and attention_masks
def get_trainer(model, tokenizer, train_dataset, config) -> Trainer:
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=TrainingArguments(
            report_to="wandb",
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
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    return trainer


if __name__ == "__main__":
    args = parse_args()
    config = read_config(config_path=args.config, modelname=args.modelname)
    logger = setup_logger(
        name="causal finetune",
        logdir=config["logdir"],
        filename="causal_finetune.log",
    )

    logger.info("\n\nStarting causal finetune ==========================")
    for hyperparam, val in config.items():
        logger.info(f"{hyperparam}: {val}")
    os.environ["WANDB_PROJECT"] = "vullm"

    # configure the directory to store the adapter
    output_dir = Path(config["adapter_path"])
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # get the model and tokenizer
    model = get_model(config["basemodel"])
    tokenizer = get_tokenizer(config["basemodel"])

    # get the dataset
    dataset = get_dataset(
        tokenizer,
        truncated_train=config["truncated_train"],
        balanced=config["balanced"],
        max_samples=config["max_samples"],
        max_length=config["max_length"],
        proomt_template=config["proompt_template"],
    )
    train_dataset = dataset["train"].map(remove_columns=["labels"])
    assert 'input_ids' in train_dataset.column_names
    assert 'attention_mask' in train_dataset.column_names
    logger.info(f"sample train example: \n{tokenizer.decode(train_dataset['input_ids'][0])}")

    print("Training...")
    trainer = get_trainer(model, tokenizer, train_dataset, config)
    train_result = trainer.train()

    # save model
    trainer.save_state()
    trainer.model.save_pretrained(output_dir)
