import pandas as pd
import os
from pathlib import Path
from assemble_metadata import (
    CouldNotGetFileMetadataError,
    CLASS_NAME,
    IMG_PATH,
    IMG_VIEW,
    BACKGROUND,
    DATE,
    INDOOR,
    OBJ_ID,
)
from PIL import Image

NUMBER_OF_IMAGES = "Number of Images"
DATASET_SIZE = "Dataset Size"
IMAGES_RESOLUTIONS = "Image's Resolutions"
DESCRIPTION = "Description"


class InvalidPathError(Exception):
    """Raised when the path is not valid"""

    pass


def get_image_size_in_bytes(img_path: Path) -> int:
    """Gets the byte size of an image
    Args:
        img_path (Path): the path to the image file.

    Returns:
        int: the image size in bytes
    Raises:
        CouldNotGetFileMetadataError: if the image's byte size could not be retrieved.
    """

    try:
        with Image.open(img_path) as img:
            byte_size = os.path.getsize(img_path)
            if byte_size == 0:
                raise CouldNotGetFileMetadataError(
                    f"Could not get the byte size of {img_path}"
                )
            return byte_size

    except CouldNotGetFileMetadataError as e:
        print(f"Error: {e}")
        return None


def get_image_resolution(img_path: Path) -> tuple[int, int]:
    """Gets the resolution
    Args:
        img_path (Path): the path to the image file.

    Returns:
        tuple(int): the image's resolutuion
    Raises:
        CouldNotGetFileMetadataError: if the image's byte size could not be retrieved.
    """

    try:
        with Image.open(img_path) as img:
            resolution = img.size
            if not any(resolution):
                raise CouldNotGetFileMetadataError(
                    f"Could not get the resoltion of {img_path}"
                )
            return resolution

    except CouldNotGetFileMetadataError as e:
        print(f"Error: {e}")
        return None


def assemble_general_summary_table(
    metadata_path: Path = None, save: bool = False, df=None
) -> pd.DataFrame:
    """Assemble the general summary table from the metadata file
    Args:
        metadata_path (Path): the path to the metadata csv file.
    Returns:
        None

    Raises:
        InvalidPathError: if the metadata_path is not a csv file.
    """

    if df is None:
        if metadata_path.suffix != ".csv":
            raise InvalidPathError(f"{metadata_path} is not a csv file.")

        df = pd.read_csv(metadata_path)

    resolutions = {}
    total_size = 0
    total_files = df.shape[0]

    img_paths = df[IMG_PATH].unique()

    for img_path in img_paths:
        total_size += get_image_size_in_bytes(img_path)
        resolution = get_image_resolution(img_path)
        if resolution not in resolutions:
            resolutions[resolution] = True

    total_size = str(total_size / (1024 * 1024)) + " MB"

    resolution = list(resolutions.keys())
    resolution = ",".join(str(r).replace(",", "x") for r in resolution)
    descriptions = [
        "Number of classes",
        NUMBER_OF_IMAGES,
        DATASET_SIZE,
        IMAGES_RESOLUTIONS,
    ]

    classes = df[CLASS_NAME].unique()
    total_classes = len(classes)
    values = [total_classes, total_files, total_size, resolution]
    class_totals = {cls: df[df[CLASS_NAME] == cls].shape[0] for cls in classes}

    total_class = "Class "
    for cls in classes:
        total_class += cls
        descriptions.append(total_class)
        values.append(class_totals[cls])
        total_class = "Class "

    data = {DESCRIPTION: descriptions, "Value": values}
    df = pd.DataFrame(data)
    if save:
        df.to_csv("general_summary.csv", index=False)
    return df


def assemble_by_class_summary_table(metadata_path: Path) -> None:
    """Assemble one summary table by class from the metadata file
    Args:
        metadata_path (Path): the path to the metadata csv file.
    Returns:
        None

    Raises:
        InvalidPathError: if the metadata_path is not a csv file.
    """

    header = [
        CLASS_NAME,
        NUMBER_OF_IMAGES,
        DATASET_SIZE,
        IMAGES_RESOLUTIONS,
        BACKGROUND,
    ]

    data = {h: [] for h in header}

    if metadata_path.suffix != ".csv":
        raise InvalidPathError(f"{metadata_path} is not a csv file.")

    df = pd.read_csv(metadata_path)

    df_grouped = df.groupby(CLASS_NAME)

    for cls, group in df_grouped:
        cls_df = assemble_general_summary_table(df=group, save=False)
        cls_df.set_index(DESCRIPTION, inplace=True)
        cls_df = cls_df.transpose()
        data[CLASS_NAME].append(cls)
        data[NUMBER_OF_IMAGES].append(cls_df[NUMBER_OF_IMAGES].values[0])
        data[DATASET_SIZE].append(cls_df[DATASET_SIZE].values[0])
        data[IMAGES_RESOLUTIONS].append(cls_df[IMAGES_RESOLUTIONS].values[0])
        backgrounds = group[BACKGROUND].unique()
        data[BACKGROUND].append(",".join(backgrounds))

    df = pd.DataFrame(data)
    df.to_csv("summary_by_class.csv", index=False)


assemble_general_summary_table(Path("metadata.csv"), save=True)
assemble_by_class_summary_table(Path("metadata.csv"))
