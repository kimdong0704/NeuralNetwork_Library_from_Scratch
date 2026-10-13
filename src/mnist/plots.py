import matplotlib.pyplot as plt
import numpy as np

from .data import IMAGE_SIZE

CORRECT_COLOR = "black"
WRONG_COLOR = "#d62728"


def _image_grid(count: int, columns: int | None, height: float) -> np.ndarray:
    """A grid of axes with axis lines hidden, one row unless `columns` is given."""
    columns = min(columns or count, count)
    rows = -(-count // columns)
    _, axes = plt.subplots(rows, columns, figsize=(2 * columns, height * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    return np.array(axes.flat)


def _show(ax, image: np.ndarray) -> None:
    # flattened network inputs are reshaped back to a square image
    ax.imshow(np.asarray(image).reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray_r")


def plot_images(images: np.ndarray, labels: np.ndarray, columns: int | None = None) -> None:
    """Each image with its label as the title."""
    for ax, image, label in zip(_image_grid(len(images), columns, 2.5), images, labels):
        _show(ax, image)
        ax.set_title(f"label: {label}")
    plt.show()


def plot_predictions(images: np.ndarray, labels: np.ndarray, predictions: np.ndarray, columns: int | None = None) -> None:
    """Each image with its label and the network's prediction; wrong predictions are titled in red."""
    for ax, image, label, predicted in zip(_image_grid(len(images), columns, 2.8), images, labels, predictions):
        _show(ax, image)
        color = CORRECT_COLOR if predicted == label else WRONG_COLOR
        ax.set_title(f"label: {label}\npredicted: {predicted}", color=color)
    plt.tight_layout()
    plt.show()
