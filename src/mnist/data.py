from pathlib import Path

import numpy as np

# every MNIST image is IMAGE_SIZE x IMAGE_SIZE pixels
IMAGE_SIZE = 28

# idx headers: labels have magic number and count (8 bytes); images add rows and columns (16 bytes)
LABEL_HEADER_BYTES = 8
IMAGE_HEADER_BYTES = 16


def read_labels(path: str | Path) -> np.ndarray:
    """The digit (0-9) of every image in an idx1 label file, as a uint8 array."""
    data = Path(path).read_bytes()
    return np.frombuffer(data, dtype=np.uint8, offset=LABEL_HEADER_BYTES)


def read_images(path: str | Path) -> np.ndarray:
    """Every image in an idx3 image file, shape (count, IMAGE_SIZE, IMAGE_SIZE), pixels 0-255 as uint8."""
    data = Path(path).read_bytes()
    pixels = np.frombuffer(data, dtype=np.uint8, offset=IMAGE_HEADER_BYTES)
    return pixels.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
