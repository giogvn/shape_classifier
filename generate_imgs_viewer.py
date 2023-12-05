from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from assemble_metadata import IMG_PATH, CLASS_NAME, IMG_TRANSFORM


def create_dataset_view(
    df: pd.DataFrame,
    cell_width: int,
    cell_height: int,
    class_name: str = CLASS_NAME,
) -> Image:
    """Creates a view of the dataset
    Args:
        metadata_path (Path): the path to the metadata csv file.
    Returns:
        img: a PIL Image object.

    Raises:
        InvalidPathError: if the metadata_path is not a csv file."""

    img_classes = df[class_name].unique()
    img_paths = []
    for img_class in img_classes:
        img_paths.extend(df[df[CLASS_NAME] == img_class][IMG_PATH].values)
    n_cols = 15
    n_rows = len(img_paths) // n_cols + (1 if len(img_paths) % n_cols != 0 else 0)

    full_img = Image.new("RGB", (n_cols * cell_width, n_rows * cell_height))

    for i, path in enumerate(img_paths):
        img = Image.open(path)
        if img.width > img.height:
            img = img.transpose(method=Image.Transpose.ROTATE_270)

        prop = img.width / img.height

        coluna = i % n_cols
        linha = i // n_cols

        x = coluna * cell_width
        y = linha * cell_height

        new_width = cell_width
        new_height = int(cell_width / prop)
        img = img.resize((new_width, new_height))
        full_img.paste(img, (x, y))

    return full_img


if __name__ == "__main__":
    cell_width = 100
    cell_height = 150
    metadata_path = Path("full_dataset_metadata.csv")
    df = pd.read_csv(metadata_path)
    for transform in df[IMG_TRANSFORM].unique():
        transf_df = df[df[IMG_TRANSFORM] == transform]
        view = create_dataset_view(transf_df, cell_width, cell_height)
        view.save(transform + "_transform_dataset_view.png")
