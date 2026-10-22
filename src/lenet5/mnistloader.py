"""
    Python port of MNISTLoader.java: reads MNIST; images come back padded to 32x32, pixel/255.
	CONVERTED FROM JAVA TO PYTHON W/ AI ASSISTANCE
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mnist import read_images, read_labels

from .model_config import IMAGE_SIZE, INPUT_SIZE, PADDING

# pixels are stored as bytes 0-255
MAX_PIXEL = 255.0


def pad_image(image28: np.ndarray) -> np.ndarray:
	# center the 28x28 image in the 32x32 array (2 pixels padding on each side)
	image32 = np.zeros(image28.shape[:-2] + (INPUT_SIZE, INPUT_SIZE))
	image32[..., PADDING:PADDING + IMAGE_SIZE, PADDING:PADDING + IMAGE_SIZE] = image28
	return image32


class MNISTLoader:
	def __init__(self, train_images_path: str | Path, train_labels_path: str | Path,
				test_images_path: str | Path, test_labels_path: str | Path):
		# kept as bytes (a quarter of the memory of Java's doubles); scaled to 0-1 when an image is read
		self.train_images = read_images(train_images_path)  # (count, 28, 28)
		self.train_labels = read_labels(train_labels_path).astype(int)
		self.test_images = read_images(test_images_path)    # (count, 28, 28)
		self.test_labels = read_labels(test_labels_path).astype(int)

	def pad_image(self, image28: np.ndarray) -> np.ndarray:
		return pad_image(image28)

	def get_train_image(self, index: int) -> np.ndarray:
		return pad_image(self.train_images[index] / MAX_PIXEL)

	def get_train_label(self, index: int) -> int:
		return int(self.train_labels[index])

	def get_test_image(self, index: int) -> np.ndarray:
		return pad_image(self.test_images[index] / MAX_PIXEL)

	def get_test_label(self, index: int) -> int:
		return int(self.test_labels[index])

	def get_num_train_images(self) -> int:
		return len(self.train_images)

	def get_num_test_images(self) -> int:
		return len(self.test_images)

	# a subset of training data (useful for quick testing)
	def get_train_image_subset(self, start_index: int, count: int) -> np.ndarray:
		return pad_image(self.train_images[start_index:start_index + count] / MAX_PIXEL)

	def get_train_label_subset(self, start_index: int, count: int) -> np.ndarray:
		return self.train_labels[start_index:start_index + count].copy()

	# a simple visualization of an image
	@staticmethod
	def print_image(image: np.ndarray) -> None:
		for row in image:
			print("".join("##" if value > 0.5 else ".." if value > 0.2 else "  " for value in row))


@dataclass
class MNISTData:
	images: np.ndarray  # already padded to 32x32
	labels: np.ndarray

	@property
	def num_images(self) -> int:
		return len(self.images)


def load_data(images_path: str | Path, labels_path: str | Path) -> MNISTData:
	# raw 28x28 images scaled to 0-1, then padded to 32x32
	images = pad_image(read_images(images_path) / MAX_PIXEL)
	return MNISTData(images, read_labels(labels_path).astype(int))
