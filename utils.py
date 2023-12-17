import pandas as pd
from assemble_metadata import (
    CLASS_NAME,
    OBJ_ID,
    BACKGROUND,
    DATE,
    IMG_PATH,
    IMG_VIEW,
    IMG_TRANSFORM,
    INDOOR,
    THRESHOLD_METHOD,
)
from PIL import Image
from pathlib import Path
from sklearn.metrics import jaccard_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EVALUATION_HEADER = [
    CLASS_NAME,
    IMG_PATH,
    OBJ_ID,
    IMG_VIEW,
    IMG_TRANSFORM,
    INDOOR,
    THRESHOLD_METHOD,
    "jaccard_score",
    "f1_score",
]

ACCEPTED_IMG_TRANSFORATIONS = [
    "exp",
    "gray_gradient",
    "log",
    "mean_filter",
    "histogram_equalization",
]


def plot_mean_results_per_something(
    df: pd.DataFrame, groupby: str, metric: str, xlabel: str, ylabel: str, title: str
):
    mean_scores = df.groupby(groupby)[metric].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=groupby, y=metric, data=mean_scores)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def find_substring_and_replace(string: str, replacement: str = "") -> str:
    """Finds a substring in a string and replaces it with another string."""

    for substring in ACCEPTED_IMG_TRANSFORATIONS:
        substring = "_" + substring
        string = string.replace(substring, replacement)
        string = string.replace("_transform", replacement)

    return Path(string).name


class ThresholdAndFeretBoxEvaluator:
    def __init__(
        self,
        segmented_ground_truth_df: pd.DataFrame,
        threshold_methods_df: pd.DataFrame,
    ):
        self.segmented_ground_truth_df = segmented_ground_truth_df
        self.threshold_methods_df = threshold_methods_df

    def assemble_ground_truth_vs_segmented_imgs_df(
        self, method: str = "otsu_bin"
    ) -> pd.DataFrame:
        """Returns a dataframe with the ground truth and segmented images comparisons."""

        original = self.segmented_ground_truth_df.copy()
        segmented = self.threshold_methods_df[
            self.threshold_methods_df[THRESHOLD_METHOD] == method
        ].copy()
        original["image_name"] = original[IMG_PATH].apply(lambda x: Path(x).name)
        original["original_index"] = original.index
        segmented["image_name"] = segmented[IMG_PATH].apply(
            lambda x: find_substring_and_replace(x)
        )
        segmented["segmented_index"] = segmented.index
        df = pd.merge(
            segmented,
            original,
            on=["image_name"],
            suffixes=("", "_segmented"),
        )
        columns_to_keep = [
            CLASS_NAME,
            IMG_PATH,
            OBJ_ID,
            BACKGROUND,
            IMG_VIEW,
            IMG_TRANSFORM,
            THRESHOLD_METHOD,
            IMG_PATH + "_segmented",
            "original_index",
            "segmented_index",
        ]

        return df[columns_to_keep]

    def evaluate_segmentation(self, method: str = "otsu_bin") -> pd.DataFrame:
        """Returns a dataframe with the evaluation metrics for the given method."""

        df = self.assemble_ground_truth_vs_segmented_imgs_df(method=method)
        df["jaccard_score"] = df.apply(
            lambda row: self.get_jaccard_score(
                row["original_index"], row["segmented_index"]
            ),
            axis=1,
        )
        return df

    def get_jaccard_score(self, ground_truth_index: int, segmented_index: int):
        """Calculates the Jaccard score between the ground truth and the segmented image."""

        ground_truth_img = Image.open(
            self.segmented_ground_truth_df.iloc[ground_truth_index][IMG_PATH]
        ).convert("L")
        ground_truth_img = np.where(np.array(ground_truth_img) == 255, 1, 0)

        segmented_img = Image.open(
            self.threshold_methods_df.iloc[segmented_index][IMG_PATH]
        ).convert("L")

        segmented_img = np.where(np.array(segmented_img) == 255, 1, 0)

        ground_truth_array = ground_truth_img.flatten()
        segmented_array = segmented_img.flatten()
        return jaccard_score(ground_truth_array, segmented_array)
