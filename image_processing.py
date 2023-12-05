from scipy.signal import convolve2d
from PIL import Image
from skimage import io, exposure
from pathlib import Path
from assemble_metadata import IMG_PATH, OBJ_ID, BACKGROUND, IMG_VIEW, IMG_TRANSFORM
import pandas as pd
import cv2 as cv
import numpy as np
import os, time

THRESHOLDEN_IMG = "thresholden_image"
DETECTED_OBJECTS = "detected_objects"
ELAPSED_TIME = "elapsed_time"


class ImageTheresholder:
    def apply_adaptive_methods(df: pd.DataFrame) -> pd.DataFrame:
        """Applies adaptive thresholding methods to the images contained in the
        paths of a dataframe.
        Args:
            df(pd.DataFrame): A dataframe containing the metadata images
              including their path.
        Returns:
            out(pd.DataFrame): A dataframe containing the thresholded images
        """

        thresholded_imgs = []
        paths = []
        methods = []
        detected_objects = []
        elapsed_times = []
        methods_unique = [
            "mean_of_neighbourhood",
            "gaussian_weighted_sum_of_neighbourhood",
        ]
        for method_name in methods_unique:
            for path in df[IMG_PATH].values:
                print(f"Thresholding image {path} with the method {method_name}")
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                thresholden_img, elapsed_time = ImageTheresholder.apply_thresholding(
                    img, method_name
                )
                # obj_count = ImageTheresholder.count_objects(thresholden_img)
                paths.append(path)
                thresholded_imgs.append(thresholden_img)
                methods.append(method_name)
                detected_objects.append(0)
                elapsed_times.append(elapsed_time)

        return pd.DataFrame(
            {
                IMG_PATH: thresholded_imgs,
                IMG_TRANSFORM: methods,
                THRESHOLDEN_IMG: thresholded_imgs,
                DETECTED_OBJECTS: detected_objects,
                ELAPSED_TIME: elapsed_times,
            }
        )

    # TODO: add type hints for imgs
    def apply_thresholding(img, method: str):
        """Applies a thresholding method to an image.
        Args:
            img(PIL.Image): A PIL Image object.
            method(str): The name of the thresholding method to be applied.
        Returns:
            out(PIL.Image): A PIL Image object representing the thresholded
                image.
        """

        if method == "mean_of_neighbourhood":
            method = cv.ADAPTIVE_THRESH_MEAN_C
        elif method == "gaussian_weighted_sum_of_neighbourhood":
            method = cv.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            raise ValueError("Invalid method name")

        start = time.perf_counter()
        out = cv.adaptiveThreshold(img, 255, method, cv.THRESH_BINARY, 11, 2)
        end = time.perf_counter()

        return out, end - start


class ImageProcessor:
    def __init__(
        self, base_path: str, metadata_path: str = "full_dataset_metadata.csv"
    ):
        self.base_path = Path(base_path)
        self.metadata_df = pd.read_csv(Path(metadata_path))

    def _to_grayscale(self, img: Image) -> Image:
        """Converts an RGB image to grayscale.
        Args:
            img(PIL): RGB PIL Image object.
        Returns:
            out(Image): A PIL Image object in grayscale (L mode)"""

        if img.mode != "L":
            img = img.convert("L")

        return img

    def mean_filter(self, img_path: str, kernel_size: int = 30) -> Image:
        """Applies the mean filter to the image using a convolution operation.
        Args:
           img_path(str): the file path to the image to be filtered located in
            self.base_path.
           kernel_size(int) : the kernel dimension with which the image will be
            convolved with.
        Returns:
            convolved(Image): A PIL Image object representing the result of the
                convolution self.img * kernel(kernel_size)"""

        img = Image.open(img_path)
        if kernel_size == 0 or type(kernel_size) != int:
            raise ValueError("Kernel size must be a positive integer")
        shape = (kernel_size, kernel_size)
        filter = np.zeros(shape)
        filter.fill(1 / (kernel_size**2))

        img = self._to_grayscale(img)

        print(f"Filtering image {img_path} with the mean filter")

        out = convolve2d(img, filter, mode="same")

        return Image.fromarray(out.astype("uint8"), "L")

    def gray_gradient_transform(self, img_path: str) -> Image:
        """Adds a gray gradient array to every row of an image. This gradient
        is a range from 0 to 255 with a constant step such that less is added to
        the left most pixels of the images up to maximum values addition to the
        right most ones.
        Args:
           img_path(str) : the file path to the image to be transformed located in
            self.base_path.
        Returns:
            convolved(Image): A PIL Image object representing the result of the
                convolution self.img * kernel(kernel_size)

        """
        img = Image.open(img_path)
        img = self._to_grayscale(img)
        img = np.array(img, dtype="float64")
        gradient_step = 255 / img.shape[1]
        gradient = np.array(np.arange(0.0, 255.0, gradient_step))
        print(f"Transforming image {img_path} with gray gradient")
        img += gradient
        round_T = np.vectorize(lambda pixel: pixel if pixel <= 255 else 255)
        img = round_T(img)
        img_name = (
            img_path.name.split(".")[0] + "_gray_gradient_transform" + img_path.suffix
        )

        return Image.fromarray(img.astype("uint8"), "L"), img_name

    def log_transform(self, img_path: str) -> Image:
        log_T = np.vectorize(lambda pixel: np.log(pixel + 1))
        img = Image.open(img_path)
        img = self._to_grayscale(img)
        c = 255 / (np.log(1 + np.max(img)))
        out = c * log_T(img)
        print(f"Transforming image {img_path} with the base 2 log")
        img_name = img_path.name.split(".")[0] + "_log_transform" + img_path.suffix
        return Image.fromarray(out.astype("uint8"), "L"), img_name

    def exp_transform(self, img_path: str) -> Image:
        print(f"Transforming image {img_path} with the exponential")
        exp_T = np.vectorize(lambda pixel: np.exp(pixel))
        img = Image.open(img_path)
        img = self._to_grayscale(img)
        img = np.array(img, dtype="float64") / 255.0
        out = exp_T(img) - 1
        out /= np.max(out)
        out *= 255
        img_name = img_path.name.split(".")[0] + "_exp_transform" + img_path.suffix
        return Image.fromarray(out.astype("uint8"), "L"), img_name

    def histogram_normalization(self, img_path: str) -> Image:
        print(f"Applying the histogram normalization to the image {img_path}")
        img = Image.open(img_path)
        img = self._to_grayscale(img)
        out = exposure.equalize_hist(np.array(img, dtype="float64"))
        img_name = (
            img_path.name.split(".")[0] + "_histogram_equalization" + img_path.suffix
        )
        return Image.fromarray((out * 255).astype("uint8"), "L"), img_name

    def apply_filter_to_multiple_images(
        self,
        filter: callable = mean_filter,
        output_dir: str = "transformed_imgs_dataset",
    ) -> None:
        """Applies a filter to multiple images in the object's base path"""

        ints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for path in self.base_path.iterdir():
            if path.is_dir():
                for img_path in path.iterdir():
                    if img_path.is_file() and img_path.suffix in [".jpg", ".png"]:
                        try:
                            found = int(img_path.name[-5]) in ints
                        except:
                            found = False

                        if not found:
                            os.remove(img_path)
                            continue
                        img_class = str(img_path.parent.name)
                        output_path = Path(output_dir) / img_class
                        if not output_path.exists():
                            output_path.mkdir()
                        filtered_img, img_name = filter(img_path)
                        filtered_img.save(output_path / img_name)

    def adaptive_threshold(self, obj_id: str):
        df = self.metadata_df[self.metadata_df[OBJ_ID] == obj_id]
        threshold_imgs, best_method = ImageTheresholder.find_best_adaptive_method(
            df[IMG_PATH].values
        )
        return threshold_imgs, best_method


if __name__ == "__main__":
    base_path = "imgs_dataset"
    img_processor = ImageProcessor(base_path)
    """img_processor.apply_filter_to_multiple_images(filter=img_processor.mean_filter)
    img_processor.apply_filter_to_multiple_images(
        filter=img_processor.gray_gradient_transform
    )
    img_processor.apply_filter_to_multiple_images(filter=img_processor.log_transform)

    img_processor.apply_filter_to_multiple_images(filter=img_processor.exp_transform)
    img_processor.apply_filter_to_multiple_images(
        filter=img_processor.histogram_normalization, output_dir="normalized_dataset"
    )"""
    df = pd.read_csv("full_dataset_metadata.csv")
    ImageTheresholder.apply_adaptive_methods(df)
