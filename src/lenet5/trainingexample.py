"""
    Python port of TrainingExample.java (Lab 3 sanity test): can the network memorise three tiny synthetic digits?
	CONVERTED FROM JAVA TO PYTHON W/ AI ASSISTANCE
"""
import sys
from pathlib import Path

import numpy as np

# run directly (e.g. VS Code's Run button): put src/ on the path so the packages import
if not __package__:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lenet5.model_config import INPUT_CHANNELS, INPUT_SIZE, OUTPUT_UNITS, PADDING, WEIGHT_HIGH, WEIGHT_LOW
from lenet5.training_config import LEARNING_RATE
from lenet5.lenet5 import LeNet5
from numpy_utils import COSTS

# fixed seeds: identical runs (same numbers as the Java file)
NETWORK_SEED = 1
PATTERN_SEED = 313

EPOCHS = 100
PRINT_INTERVAL = 10

# random patterns (digits 3-9) switch a pixel on with this probability
PATTERN_DENSITY = 0.3

# like Java's static Random(313): one generator shared by every random pattern
pattern_random = np.random.default_rng(PATTERN_SEED)


def create_synthetic_digit(digit: int) -> np.ndarray:
	# drawn on the padded INPUT_SIZE x INPUT_SIZE grid, with the same coordinates as the Java file
	image = np.zeros((INPUT_SIZE, INPUT_SIZE))

	# simple patterns for each digit (very simplified); in reality you'd use actual MNIST data
	if digit == 0:
		# a ring for 0
		for i in range(8, 24):
			for j in range(8, 24):
				di = i - 16
				dj = j - 16
				if 36 < di * di + dj * dj < 64:
					image[i, j] = 1.0
	elif digit == 1:
		# a vertical line for 1
		image[8:24, 15:17] = 1.0
	elif digit == 2:
		# a simplified 2
		image[10, 10:22] = 1.0  # top
		image[16, 10:22] = 1.0  # middle
		image[22, 10:22] = 1.0  # bottom
		image[10:16, 21] = 1.0  # right side
		image[16:22, 10] = 1.0  # left side
	else:
		# other digits: random patterns
		image[8:24, 8:24] = pattern_random.random((16, 16)) < PATTERN_DENSITY

	# LeNet5 pads its input itself, so drop the padding border (all zeros) and give a batch of one:
	# (INPUT_SIZE, INPUT_SIZE) -> (1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
	unpadded = image[PADDING:-PADDING, PADDING:-PADDING]
	return unpadded.reshape(1, INPUT_CHANNELS, *unpadded.shape)


def one_hot(label: int) -> np.ndarray:
	return np.eye(OUTPUT_UNITS)[[label]]


def compute_loss(output: np.ndarray, label: int) -> float:
	return COSTS.CROSS_ENTROPY.function(output, one_hot(label))


def train(net: LeNet5, image: np.ndarray, label: int) -> None:
	# forward, backward with dC/dp, then one gradient-descent step
	output = net.forward(image)
	net.backward(COSTS.CROSS_ENTROPY.derivative(output, one_hot(label)))
	net.step(LEARNING_RATE)


def main(weight_low: float = WEIGHT_LOW, weight_high: float = WEIGHT_HIGH) -> None:
	print("=== LeNet-5 Training Example ===\n")
	print(f"Initial weights in [{weight_low}, {weight_high})\n")

	net = LeNet5(seed=NETWORK_SEED, weight_low=weight_low, weight_high=weight_high)

	# small training dataset: synthetic digits 0, 1, 2
	labels = [0, 1, 2]
	training_data = [create_synthetic_digit(label) for label in labels]

	print("Before training:")
	for i, (image, label) in enumerate(zip(training_data, labels)):
		output = net.forward(image)
		prediction = int(np.argmax(output))
		print(f"Sample {i} (label={label}): predicted={prediction}, "
			f"loss={compute_loss(output, label):.4f}, confidence={output[0, prediction]:.4f}")

	print("\n=== Training ===")
	for epoch in range(EPOCHS):
		total_loss = 0.0
		correct = 0

		for image, label in zip(training_data, labels):
			train(net, image, label)

			# metrics after training on this sample
			output = net.forward(image)
			total_loss += compute_loss(output, label)
			if np.argmax(output) == label:
				correct += 1

		if (epoch + 1) % PRINT_INTERVAL == 0:
			average_loss = total_loss / len(labels)
			accuracy = correct / len(labels) * 100
			print(f"Epoch {epoch + 1:3d}: avg_loss={average_loss:.4f}, accuracy={accuracy:.1f}%")

	print("\n=== After Training ===")
	for i, (image, label) in enumerate(zip(training_data, labels)):
		output = net.forward(image)
		prediction = int(np.argmax(output))

		print(f"\nSample {i} (label={label}):")
		print(f"  Predicted: {prediction}")
		print(f"  Loss: {compute_loss(output, label):.4f}")
		print(f"  Confidence: {output[0, prediction]:.4f}")
		print("  All probabilities:")
		for digit, probability in enumerate(output[0]):
			print(f"    Digit {digit}: {probability:.4f}{' (correct)' if digit == label else ''}")

	print("\n=== Training Complete ===")


if __name__ == "__main__":
	main()
