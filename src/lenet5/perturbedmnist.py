"""
    Python port of PerturbedMNIST.java: shifted and noisy copies of MNIST test images,
	identical on every run and for every model (the same images as the Java version).
	CONVERTED FROM JAVA TO PYTHON W/ AI ASSISTANCE

	A shift of d moves the digit d pixels left or right AND d pixels up or down
	(the direction is fixed per test image). Up to d = 4 no part of any digit
	leaves the central 28x28 region, so the CNN and the FFNet see the same digit.

	Every function takes one 32x32 image with an int test_index, or a stack (count, 32, 32)
	with an array of test indices, so a whole test set can be perturbed at once.
"""
import numpy as np

from .javarandom import JavaRandom
from .model_config import IMAGE_SIZE, PADDING

# new Random(SHIFT_SEED + testIndex) and new Random(NOISE_SEED + testIndex), as in the Java file
SHIFT_SEED = 1000
NOISE_SEED = 5000


def shifted(image: np.ndarray, test_index: int | np.ndarray, d: int) -> np.ndarray:
	"""The image moved d pixels horizontally and d pixels vertically; vacated pixels are 0."""
	random = JavaRandom(SHIFT_SEED + np.atleast_1d(test_index))
	dx = np.where(random.next_boolean(), d, -d)
	dy = np.where(random.next_boolean(), d, -d)

	if image.ndim == 2:
		return shift(image, int(dy[0]), int(dx[0]))

	# each image has one of four directions: shift every group at once
	out = np.empty_like(image)
	for group_dy in (d, -d):
		for group_dx in (d, -d):
			group = (dy == group_dy) & (dx == group_dx)
			out[group] = shift(image[group], group_dy, group_dx)
	return out


def noisy(image: np.ndarray, test_index: int | np.ndarray, sigma: float) -> np.ndarray:
	"""The image plus Gaussian noise with standard deviation sigma, clipped to [0, 1]."""
	random = JavaRandom(NOISE_SEED + np.atleast_1d(test_index))
	rows, cols = image.shape[-2:]

	# one Gaussian per pixel, row by row, like the Java loops
	noise = random.next_gaussians(rows * cols).reshape(image.shape)
	return np.minimum(1.0, np.maximum(0.0, image + sigma * noise))


def shift(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
	"""Moves the picture dy rows down and dx columns right (negative values: up / left)."""
	rows, cols = image.shape[-2:]
	out = np.zeros_like(image)

	# out[i, j] = image[i - dy, j - dx] wherever that pixel exists
	out[..., max(dy, 0):rows + min(dy, 0), max(dx, 0):cols + min(dx, 0)] = \
		image[..., max(-dy, 0):rows - max(dy, 0), max(-dx, 0):cols - max(dx, 0)]
	return out


def to_ffnet_input(image32: np.ndarray) -> np.ndarray:
	"""Removes the 2-pixel padding of a 32x32 image and flattens it row by row into 784 values."""
	return image32[PADDING:PADDING + IMAGE_SIZE, PADDING:PADDING + IMAGE_SIZE].reshape(IMAGE_SIZE * IMAGE_SIZE)
