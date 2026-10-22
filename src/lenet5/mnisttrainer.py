"""
    Python port of MNISTTrainer.java: trains LeNet5 on MNIST.
	CONVERTED FROM JAVA TO PYTHON W/ AI ASSISTANCE

	Usage (from src/):   python -m lenet5.mnisttrainer [epochs=2] [seed=1]

	Train on the first 50,000 training images one image at a time, watch progress on the last
	10,000 training images (the VALIDATION set), and evaluate the TEST set once, at the end.
	Training loss and accuracy come from the forward pass inside train(), so there is no second
	forward pass per example. Progress is also written to learning_curve.csv for plotting.
	At the end: a dead-unit count, the shifted and noisy test sets from perturbedmnist, and
	(with a display, or in a notebook) the NetworkVisualizer for the first training image.

	The learning rate is in lenet5.training_config and the initial weight range in lenet5.model_config:
	change them there for the Lab 4 experiments.
"""
import sys
import time
from pathlib import Path

import numpy as np

# run directly (e.g. VS Code's Run button): put src/ on the path so the packages import
if not __package__:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lenet5.lenet5 import LeNet5
from lenet5.mnistloader import MNISTLoader
from lenet5.model_config import OUTPUT_UNITS
from lenet5.networkvisualizer import NetworkVisualizer, has_display
from lenet5.padded_input import forward, layers_after_pad
from lenet5.perturbedmnist import noisy, shifted
from lenet5.training_config import LEARNING_RATE
from numpy_utils import COSTS

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mnist"
LEARNING_CURVE_PATH = Path("learning_curve.csv")

NUM_TRAIN = 50000                     # images 0 .. 49,999
VAL_START, VAL_END = 50000, 60000     # images 50,000 .. 59,999
REPORT_INTERVAL = 5000
TARGET_ACCURACY = 0.97
SHIFTS = [0, 1, 2, 3, 4]
SIGMAS = [0.1, 0.2, 0.3, 0.4, 0.5]

DEAD_UNIT_IMAGES = 1000

DEFAULT_EPOCHS = 2
DEFAULT_SEED = 1


def train(net: LeNet5, image32: np.ndarray, label: int) -> np.ndarray:
	# forward, backward with dC/dp, then one step; returns the forward pass's output (before this update)
	output = forward(net, image32)
	deltas = COSTS.CROSS_ENTROPY.derivative(output, one_hot(label))
	for layer in reversed(layers_after_pad(net)):
		deltas = layer.backward(deltas)
	net.step(LEARNING_RATE)
	return output


def one_hot(label: int) -> np.ndarray:
	return np.eye(OUTPUT_UNITS)[[label]]


def compute_loss(output: np.ndarray, label: int) -> float:
	return COSTS.CROSS_ENTROPY.function(output, one_hot(label))


def get_prediction(output: np.ndarray) -> int:
	return int(np.argmax(output))


def main(epochs: int = DEFAULT_EPOCHS, seed: int = DEFAULT_SEED) -> LeNet5:
	print("=== LeNet-5 MNIST Trainer ===\n")
	mnist = MNISTLoader(
		DATA_DIR / "train-images-idx3-ubyte",
		DATA_DIR / "train-labels-idx1-ubyte",
		DATA_DIR / "t10k-images-idx3-ubyte",
		DATA_DIR / "t10k-labels-idx1-ubyte"
	)
	net = LeNet5(seed=seed)
	print(f"Seed {seed}, {epochs} epochs, {NUM_TRAIN} training / {VAL_END - VAL_START} validation images")
	print(f"Validation accuracy before training: {100 * evaluate_validation(net, mnist):.2f}%\n")

	seen = 0
	first_above_target = -1
	with open(LEARNING_CURVE_PATH, "w") as csv:
		csv.write("examples_seen,mean_train_loss,train_accuracy,validation_accuracy\n")
		print(" examples   mean train loss   train acc   validation acc   time")
		for epoch in range(epochs):
			loss_sum = 0.0
			correct = 0
			start = time.perf_counter()
			for i in range(NUM_TRAIN):
				label = mnist.get_train_label(i)
				output = train(net, mnist.get_train_image(i), label)
				# loss and prediction of the forward pass inside train(), i.e. BEFORE this update
				loss_sum += compute_loss(output, label)
				if get_prediction(output) == label:
					correct += 1
				seen += 1
				if seen % REPORT_INTERVAL == 0:
					val = evaluate_validation(net, mnist)
					print(f"{seen:9d}   {loss_sum / REPORT_INTERVAL:15.4f}   {100.0 * correct / REPORT_INTERVAL:8.2f}%   "
						f"{100 * val:13.2f}%   {time.perf_counter() - start:5.1f}s")
					csv.write(f"{seen},{loss_sum / REPORT_INTERVAL:.6f},{correct / REPORT_INTERVAL:.4f},{val:.4f}\n")
					csv.flush()
					if first_above_target < 0 and val >= TARGET_ACCURACY:
						first_above_target = seen
					loss_sum = 0.0
					correct = 0

	if first_above_target > 0:
		print(f"\nValidation accuracy first reached 97% after {first_above_target} training examples.")
	else:
		print("\nValidation accuracy did not reach 97%.")

	# ---- the test set, evaluated once ------------------------------------------------
	print(f"TEST accuracy (reported once): {100 * evaluate_test(net, mnist, 0, 0.0):.2f}%")

	dead_unit_report(net, mnist)

	print("\nShifted test digits (d pixels horizontally and d pixels vertically):")
	for d in SHIFTS:
		print(f"  d = {d}: {100 * evaluate_test(net, mnist, d, 0.0):6.2f}%")
	print("Noisy test digits (Gaussian noise, standard deviation sigma):")
	for sigma in SIGMAS:
		print(f"  sigma = {sigma:.1f}: {100 * evaluate_test(net, mnist, 0, sigma):6.2f}%")

	if has_display():
		forward(net, mnist.get_train_image(0))
		viz = NetworkVisualizer(net)
		viz.update_visualization()
		viz.show()

	return net


def evaluate_validation(net: LeNet5, mnist: MNISTLoader) -> float:
	correct = 0
	for i in range(VAL_START, VAL_END):
		if get_prediction(forward(net, mnist.get_train_image(i))) == mnist.get_train_label(i):
			correct += 1
	return correct / (VAL_END - VAL_START)


def evaluate_test(net: LeNet5, mnist: MNISTLoader, d: int, sigma: float) -> float:
	"""Accuracy on all 10,000 test images, each shifted by d and with noise sigma."""
	count = mnist.get_num_test_images()
	indices = np.arange(count)

	# the whole test set is perturbed at once; each image still gets its own Java seed
	images = np.stack([mnist.get_test_image(i) for i in indices])
	if d > 0:
		images = shifted(images, indices, d)
	if sigma > 0:
		images = noisy(images, indices, sigma)

	correct = 0
	for i in indices:
		if get_prediction(forward(net, images[i])) == mnist.get_test_label(i):
			correct += 1
	return correct / count


def dead_unit_report(net: LeNet5, mnist: MNISTLoader) -> None:
	"""Feature maps / units that are never positive on 1,000 validation images."""
	c1 = np.zeros(net.c1_layer.out_channels, dtype=bool)
	c3 = np.zeros(net.c3_layer.out_channels, dtype=bool)
	c5 = np.zeros(net.c5_layer.out_channels, dtype=bool)
	f6 = np.zeros(len(net.f6_layer.bias), dtype=bool)

	for i in range(VAL_START, VAL_START + DEAD_UNIT_IMAGES):
		forward(net, mnist.get_train_image(i))
		# conv outputs (1, maps, rows, cols): a map is alive if any of its pixels is positive
		c1 |= np.any(net.c1_layer.last_outputs[0] > 0, axis=(1, 2))
		c3 |= np.any(net.c3_layer.last_outputs[0] > 0, axis=(1, 2))
		c5 |= np.any(net.c5_layer.last_outputs[0] > 0, axis=(1, 2))
		f6 |= net.f6_layer.last_output[0] > 0

	print(f"Dead units (never positive on 1,000 validation images): C1 {count_dead(c1)}/{len(c1)}, "
		f"C3 {count_dead(c3)}/{len(c3)}, C5 {count_dead(c5)}/{len(c5)}, F6 {count_dead(f6)}/{len(f6)}")


def count_dead(alive: np.ndarray) -> int:
	return int(np.sum(~alive))


if __name__ == "__main__":
	main(
		epochs=int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EPOCHS,
		seed=int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEED
	)
