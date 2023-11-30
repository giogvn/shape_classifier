from scipy.signal import convolve2d
from PIL import Image
import numpy as np
from pathlib import Path
import os


class ImageProcessor:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

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
        exp_T = np.vectorize(lambda pixel: np.exp(pixel))
        img = Image.open(img_path)
        img = self._to_grayscale(img)
        c = 255 / (np.log(1 + np.exp(img)))
        out = c * exp_T(img)
        print(f"Transforming image {img_path} with the exponential")
        img_name = img_path.name.split(".")[0] + "_exp_transform" + img_path.suffix
        return Image.fromarray(out.astype("uint8"), "L"), img_name

    def apply_filter_to_multiple_images(
        self,
        filter: callable = mean_filter,
        output_dir: str = "transformed_imgs_dataset",
    ) -> None:
        """Applies a filter to multiple images in the object's base path"""

        ints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        found_img = False
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


if __name__ == "__main__":
    base_path = "imgs_dataset"
    img_processor = ImageProcessor(base_path)
    # img_processor.apply_filter_to_multiple_images(filter=img_processor.mean_filter)
    img_processor.apply_filter_to_multiple_images(
        filter=img_processor.gray_gradient_transform
    )

    img_processor.apply_filter_to_multiple_images(filter=img_processor.log_transform)
    img_processor.apply_filter_to_multiple_images(filter=img_processor.exp_transform)
