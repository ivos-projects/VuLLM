import pandas as pd
from datasets import Dataset, ClassLabel, DatasetDict
from pathlib import Path

SEED = 42


def preprocess_reveal_dataset(
    csv_path: str, dataset_name="reveal", push_to_hub=False
) -> DatasetDict:
    """
    Preprocess the ReVeal dataset and return a DatasetDict object with the train,
    valid and test splits.
    """
    assert Path(csv_path).exists(), f"File {csv_path} does not exist."
    df = pd.read_csv(csv_path)
    df["vulnerable"] = df["label"].apply(lambda x: "True" if x == 1 else "False")

    dataset = Dataset.from_pandas(df[["functionSource", "vulnerable"]])
    dataset = dataset.rename_column("functionSource", "input")
    dataset = dataset.rename_column("vulnerable", "output")

    # Create the output as a ClassLabel column to perform the split
    new_features = dataset.features.copy()
    new_features["output"] = ClassLabel(num_classes=2, names=["False", "True"], id=None)
    dataset = dataset.cast(new_features)

    dataset = dataset.train_test_split(
        test_size=0.2, seed=SEED, stratify_by_column="output"
    )
    dev_set = dataset["test"].train_test_split(
        test_size=0.5, seed=SEED, stratify_by_column="output"
    )

    ds_splits = DatasetDict(
        {
            "train": dataset["train"],
            "valid": dev_set["train"],  # train in devset is the validation set
            "test": dev_set["test"],
        }
    )

    # must log in with huggingface-cli if push to hub
    if push_to_hub:
        ds_splits.push_to_hub(dataset_name)

    return ds_splits


if __name__ == "__main__":
    dataset = preprocess_reveal_dataset(
        csv_path="../data/reveal.csv", dataset_name="reveal", push_to_hub=False
    )
    print(dataset)
