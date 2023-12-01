from pathlib import Path
from PIL import Image
import shutil


def assemble_full_dataset(inputs: list, output: str = "full_dataset") -> None:
    """Assembles a full dataset from multiple datasets.
    Args:
        inputs (list): A list of paths to the datasets.
        output (str, optional): The path to the output folder. Defaults to "full_dataset".
    """
    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)
    for input in inputs:
        input = Path(input)
        for path in input.iterdir():
            if path.is_dir():
                for img_path in path.iterdir():
                    if img_path.is_file() and img_path.suffix in [".jpg", ".png"]:
                        img_class = str(img_path.parent.name)
                        output_path = output / img_class
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        img_name = img_path.name

                        img = Image.open(img_path)
                        img.save(output_path / img_name)


def zip_folders(inputs: list) -> None:
    """Zips the folders in the inputs list.
    Args:
        inputs (list): A list of paths to the folders.
    """
    for input in inputs:
        path = Path(input)
        if path.is_dir():
            subdir_name = path.name
            shutil.make_archive(subdir_name, "zip", path)


if __name__ == "__main__":
    inputs = ["imgs_dataset", "transformed_imgs_dataset"]
    output = "full_dataset"
    """print(f"Assembling full dataset from {inputs}")
    assemble_full_dataset(inputs, output)"""
    inputs = [i for i in Path(output).iterdir() if i.is_dir()]
    print(f"Zipping {inputs}")
    zip_folders(inputs)
    print("Done!")
