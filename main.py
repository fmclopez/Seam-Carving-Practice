import typing
import cv2

import imageio
import numpy as np
from matplotlib import pyplot as plt

from base import SeamCarving
from implementation import SeamCarvingImplementation


def read_uint8_image(path) -> np.ndarray:
    #print(cv2.imread(path))
    return np.array(imageio.imread(path))


def read_boolean_mask(path) -> np.ndarray:
    return np.array(imageio.imread(path)).mean(axis=2) > 127


class InputImages:
    FILE_PATH = "sample_image.jpg"  # Don't edit the sample images
    RGB_UINT8 = read_uint8_image(FILE_PATH)
    RGB_FLOAT64 = (RGB_UINT8 / 255).astype(np.float64)
    GRAY_FLOAT64 = 0.299 * RGB_FLOAT64[:, :, 0] + 0.587 * RGB_FLOAT64[:, :, 1] + 0.1114 * RGB_FLOAT64[:, :, 2]
    GRAY_UINT8 = (GRAY_FLOAT64 * 255).astype(np.uint8)

class CarvingValues:
    DO_NOTHING = 0
    ADD_SEAMS = int(min(InputImages.RGB_FLOAT64.shape[:2]) * 0.25)
    REMOVE_SEAMS = -ADD_SEAMS


class RetainMasks:
    NONE = None
    FILE_PATH = "sample_retain.jpg"  # Don't edit the sample images
    BINARY_MASK = read_boolean_mask(FILE_PATH)


class RemoveMasks:
    NONE = None
    FILE_PATH = "sample_remove.jpg"  # Don't edit the sample images
    BINARY_MASK = read_boolean_mask(FILE_PATH)


# Your implementations
SEAM_CARVING_IMPLEMENTATION = SeamCarvingImplementation()


def present(
    title: str,
    image: typing.Union[str, np.ndarray] = InputImages.RGB_FLOAT64,
    #image: typing.Union[str, np.ndarray] = InputImages.GRAY_FLOAT64,
    vertical_seams: int = CarvingValues.DO_NOTHING,
    horizontal_seams: int = CarvingValues.DO_NOTHING,
    retain_mask: typing.Union[str, np.ndarray] = RemoveMasks.NONE,
    remove_mask: typing.Union[str, np.ndarray] = RemoveMasks.NONE,
    seam_carver: SeamCarving = SEAM_CARVING_IMPLEMENTATION
):
    if isinstance(image, str):
        image = read_uint8_image(image) / 255
    if isinstance(retain_mask, str):
        retain_mask = read_boolean_mask(retain_mask)
    if isinstance(remove_mask, str):
        remove_mask = read_boolean_mask(remove_mask)

    figure = image.copy()
    figure_shape = figure.shape[:2]

    if retain_mask is not None:
        if retain_mask.shape[:2] != figure_shape:
            retain_mask = retain_mask.astype(float)
            retain_mask = cv2.resize(retain_mask, dsize=figure_shape, interpolation=cv2.INTER_CUBIC)
            retain_mask = retain_mask > 0.5
        RETAIN_MASK_COLOR = np.array([0.1, 0.9, 0.1])  # Green
        figure[retain_mask] = (figure[retain_mask] + RETAIN_MASK_COLOR[np.newaxis, :]) / 2

    if remove_mask is not None:
        if remove_mask.shape[:2] != figure_shape:
            remove_mask = remove_mask.astype(float)
            remove_mask = cv2.resize(remove_mask, dsize=figure_shape, interpolation=cv2.INTER_CUBIC)
            remove_mask = remove_mask > 0.5
        REMOVE_MASK_COLOR = np.array([0.9, 0.1, 0.1])  # Red
        figure[remove_mask] = (figure[remove_mask] + REMOVE_MASK_COLOR[np.newaxis, :]) / 2

    output = seam_carver(
        image=image,
        vertical_seams=vertical_seams,
        horizontal_seams=horizontal_seams,
        retain_mask=retain_mask,
        remove_mask=remove_mask,
    )

    cv2.imwrite("output.png", output)

    fig, axs = plt.subplots(1, 2)

    ax: plt.Axes = axs[0]
    ax.imshow(figure, cmap="gray")
    ax.set_title("Input Image")

    ax: plt.Axes = axs[1]
    ax.imshow(output, cmap="gray")
    ax.set_title("Output Image")

    fig.suptitle(title)

    plt.show()


if __name__ == "__main__":

    # SCORE +1 can process RGB (H×W×3) uint8 [0,255] and float64 [0,1] images
    output = present("RGB uint8", image=InputImages.RGB_UINT8)
    output = present("RGB float64", image=InputImages.RGB_FLOAT64)

    # SCORE +1 can process Gray (H×W) uint8 [0,255] and float64 [0,1] images
    output = present("Gray uint8", image=InputImages.GRAY_UINT8)
    output = present("Gray float64", image=InputImages.GRAY_FLOAT64)

    # SCORE +1 can process image from file path
    output = present("File Path", image=InputImages.FILE_PATH)

    # SCORE +1 can remove vertical seams (decrease width)
    output = present("Vertical Seam Removal", vertical_seams=CarvingValues.REMOVE_SEAMS)

    # SCORE +1 can remove horizontal seams (decrease height)
    output = present("Horizontal Seam Removal", horizontal_seams=CarvingValues.REMOVE_SEAMS)

    # SCORE +1 can add vertical seams (increase width)
    output = present("Vertical Seam Creation", vertical_seams=CarvingValues.ADD_SEAMS)

    # SCORE +1 can add horizontal seams (increase height)
    output = present("Horizontal Seam Creation", horizontal_seams=CarvingValues.ADD_SEAMS)

    # SCORE +1 can use remove masks (masked regions is prioritzed when removing seams)
    #print("rmask before imple.")
    #print(RemoveMasks.BINARY_MASK)
    output = present(
        "Remove-Priority Mask",
        vertical_seams=CarvingValues.REMOVE_SEAMS,
        horizontal_seams=CarvingValues.REMOVE_SEAMS,
        remove_mask=RemoveMasks.BINARY_MASK
    )

    # SCORE +1 can use retain masks (masked regions will not be removed)
    output = present(
        "Retain-Priority Mask",
        vertical_seams=CarvingValues.REMOVE_SEAMS,
        horizontal_seams=CarvingValues.REMOVE_SEAMS,
        retain_mask=RetainMasks.BINARY_MASK,
    )

    # SCORE +1 can read images as binary masks
    output = present(
        "Masks from Files",
        vertical_seams=CarvingValues.REMOVE_SEAMS,
        horizontal_seams=CarvingValues.REMOVE_SEAMS,
        retain_mask=RetainMasks.FILE_PATH,
        remove_mask=RemoveMasks.FILE_PATH
    )

# Note: When checking, I'll use a different set of images
