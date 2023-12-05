from pathlib import Path
import pandas as pd
from PIL import Image
from datetime import datetime
import warnings


CLASS_NAME = "class_name"
OBJ_ID = "obj_id"
BACKGROUND = "background"
DATE = "date"
IMG_PATH = "img_path"
IMG_VIEW = "img_view"
IMG_TRANSFORM = "img_transformation"
INDOOR = "indoor"

METADATA_HEADER = [
    CLASS_NAME,
    IMG_PATH,
    OBJ_ID,
    BACKGROUND,
    DATE,
    IMG_VIEW,
    IMG_TRANSFORM,
    INDOOR,
]
ACCEPTED_IMG_VIEWS = [
    "front",
    "back",
    "side",
    "isometric",
    "superior",
    "open_inside",
    "open_out",
    "open_inside_isometric",
    "open_outside_isometric",
]
ACCEPTED_IMG_TRANSFORATIONS = [
    "exp",
    "gray_gradient",
    "log",
    "mean_filter",
    "histogram_equalization",
]
INDOOR = True
ACCEPTED_IMG_BACKGROUNDS = ["dark", "light"]
ACCEPTED_IMG_FORMATS = [".jpg", ".png"]
INVALID_IMG_NAME_ERROR = f"""The image name must have the following format:\n
    objId_[background_color]_bg_[img_view]_[view_rank|...].[img_format].\n
    background_color must be one of {ACCEPTED_IMG_BACKGROUNDS}\n
    img_view must be one of {ACCEPTED_IMG_VIEWS}\n
    view_rank must be a number greater than 0\n
    img_format must be one of {ACCEPTED_IMG_FORMATS}
    """


class InvalidDirectoryStructureError(Exception):
    """Raised when the directory structure is not valid"""

    pass


class CouldNotGetFileMetadataError(Exception):
    """Raised when the file metadata could not be retrieved"""

    pass


class InvalidImageNameError(Exception):
    f"""The image name must have the following format:\n
    objId_[background_color]_bg_[img_view]_[view_rank|...].[img_format].\n
    background_color must be one of {ACCEPTED_IMG_BACKGROUNDS}\n
    img_view must be one of {ACCEPTED_IMG_VIEWS}\n
    view_rank must be a number greater than 0\n
    img_format must be one of {ACCEPTED_IMG_FORMATS}
    """
    pass


def get_image_creation_date(img_path: Path) -> str:
    """Returns the creation date of an image.
    Args:
        image_path (Path): the img file path.
    Returns:
    str: the image creation date extracted from its metadata.

    Raises:
        CouldNotGetFileMetadataError: if the image metadata could not be retrieved"""
    with Image.open(img_path) as img:
        try:
            creation_date = img._getexif().get(
                36867
            )  # 36867 corresponds to DateTimeOriginal in EXIF data
            creation_date = datetime.strptime(creation_date, "%Y:%m:%d %H:%M:%S")
        except:
            warnings.warn(
                "Could not get the image creation date. Using the current date instead",
                category=Warning,
            )
            creation_date = datetime.now().strftime("%Y:%m:%d %H:%M:%S")

        return creation_date


def get_img_metadata(img_path_name: str) -> tuple[str, str, str, str, str, bool]:
    """Returns the metadata of an image.
    Args:
        img_path_name (str): the img file name.

    Returns:
        tuple[str, str, str, str, str, bool]: A tuple containing the image's
        object id, image view, image transformation background and indoor
        information of an image.

    Raises:
        InvalidImageNameError: if the img_path name does not have the expected format:
        objId_[background_color]_bg_[img_view]_[view_rank|...].[img_format].
        See InvalidImageNameError for more details.
    """

    metadata = img_path_name.split("_dark_bg_")

    if len(metadata) != 2:
        metadata = img_path_name.split("_light_bg_")
        if len(metadata) != 2:
            raise InvalidImageNameError(INVALID_IMG_NAME_ERROR)
        else:
            background = "light"
    else:
        background = "dark"

    obj_id = metadata[0]
    img_view = None
    for view in ACCEPTED_IMG_VIEWS:
        if view in metadata[1]:
            img_view = view
            break

    img_transform = "original"
    for transform in ACCEPTED_IMG_TRANSFORATIONS:
        if transform in metadata[1]:
            img_transform = transform
            break

    if not img_view:
        raise InvalidImageNameError(INVALID_IMG_NAME_ERROR)

    return obj_id, img_view, img_transform, background, INDOOR


def assemble_metadata(
    base_path: Path, out: Path = "metadata.csv", header: list[str] = METADATA_HEADER
) -> None:
    """
    Assembles metadata.csv file from the images from which path's are rooted in
    base_path and have depth equal to 2 relatively to it. Every deeper or shallower
    image will be ignored.

    Args:
        base_path (Path): the root where the images classes directories are.
    Returns:
        None.

    Raises:
        InvalidDirectoryStructureError: If the directory structure does not have
        any image with depth equal to 2. The base path argument must have at least
        one internal directory in which some image is located.
    Example:
        The following directory structure is valid:
        base_path/
        │
        ├── class1/
        │   ├── img1.png
        │   └── img2.png
        │
        └── class2/
            ├── img1.jpg
            └── img2.jpg

        The following directory structure is NOT valid:
        base_path/
        ├── img1.png
        ├── img2.jpg
        └──  img3.jpg
    """

    data = {attr: [] for attr in header}
    found_img = False
    for path in base_path.iterdir():
        if path.is_dir():
            for img_path in path.iterdir():
                if img_path.is_file() and img_path.suffix in [".jpg", ".png"]:
                    print(img_path)
                    found_img = True
                    date = get_image_creation_date(img_path)
                    (
                        obj_id,
                        img_view,
                        img_transform,
                        background,
                        indoor,
                    ) = get_img_metadata(str(img_path.name))
                    data["img_path"].append(img_path)
                    data["date"].append(date)
                    data["class_name"].append(path.name)
                    data["obj_id"].append(obj_id)
                    data["background"].append(background)
                    data["img_view"].append(img_view)
                    data["img_transformation"].append(img_transform)
                    data["indoor"].append(indoor)

    if not found_img:
        raise InvalidDirectoryStructureError(
            f"Invalid directory structure. It should have at least one image in a directory with relative depth equal to 2.."
        )

    df = pd.DataFrame(data)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    assemble_metadata(Path("full_dataset"), out="full_dataset_metadata.csv")
