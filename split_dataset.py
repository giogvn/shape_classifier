from pathlib import Path
import os
import splitfolders
from collections import namedtuple


def split_dirs_dataset_in_test_train_sets(
    input_folder: str,
    output: str = "train_test_datasets",
    ratios: tuple = (0.8, 0.1, 0.1),
) -> None:
    """Splits a dataset in train, test and validation sets.
    The dataset must be organized in folders, where each folder is a class.
    The output is a folder with the same structure as the input folder.
    If output is not provided, it is created.
    Args:
        input_folder (str): The path to the input folder.
        output (str, optional): The path to the output folder. Defaults to None.
        ratios(tuple, optional): The ratios for the train, validation and test sets.
            Defaults to (0.8, 0.1, 0.1).
    """
    splits = namedtuple("Splits", "train validation test")
    splits = splits(*ratios)
    splitfolders.ratio(
        input_folder,
        output=output,
        seed=42,
        ratio=(splits.train, splits.validation, splits.test),
    )
    return output


if __name__ == "__main__":
    input_folder = "imgs_dataset"
    output = "train_test_datasets"
    ratios = (0.8, 0.1, 0.1)
    print(
        f"Splitting dataset {input_folder} in train, validation and test sets with ratios {ratios}"
    )
    split_dirs_dataset_in_test_train_sets(input_folder, output, ratios=ratios)
    print("Done!")
