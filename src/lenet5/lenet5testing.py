"""
    Python port of LeNet5Testing.java: four sanity checks of the LeNet-5 forward pass.
	CONVERTED FROM JAVA TO PYTHON W/ AI ASSISTANCE
"""
import sys
from pathlib import Path

import numpy as np

# run directly (e.g. VS Code's Run button): put src/ on the path so the packages import
if not __package__:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lenet5.config import IMAGE_SIZE, INPUT_CHANNELS, OUTPUT_UNITS
from lenet5.lenet5 import LeNet5

# seeds of the random test images (same numbers as the Java file)
DETERMINISM_INPUT_SEED = 67890
FIRST_INPUT_SEED = 111
SECOND_INPUT_SEED = 222

# two outputs closer than this count as identical (determinism test)
IDENTICAL_TOLERANCE = 1e-10
# two outputs further apart than this count as different (different-inputs test)
DIFFERENT_TOLERANCE = 1e-6
# probabilities must sum to 1 within this (zero / ones tests)
SUM_TOLERANCE = 1e-6

INPUT_SHAPE = (INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


def random_input(seed: int) -> np.ndarray:
	# INPUT_SHAPE values in [0, 1[, like Java's Random(seed).nextDouble()
	return np.random.default_rng(seed).random(INPUT_SHAPE)


def print_output(output: np.ndarray) -> None:
	for digit, probability in enumerate(output):
		print(f"  Digit {digit}: {probability:.6f}")


def print_result(passed: bool, pass_message: str, fail_message: str) -> None:
	print(f"✓ PASS: {pass_message}" if passed else f"✗ FAIL: {fail_message}")


def is_valid_distribution(output: np.ndarray) -> bool:
	# softmax output: OUTPUT_UNITS values, all non-negative, summing to 1
	total = np.sum(output)
	all_non_negative = bool(np.all(output >= 0))

	print(f"\nSum: {total:.6f}")
	print(f"All non-negative: {all_non_negative}")

	return output.shape == (OUTPUT_UNITS,) and abs(total - 1.0) < SUM_TOLERANCE and all_non_negative


def test_determinism() -> bool:
	print("=== Determinism Test ===")
	print("Testing that same input produces same output...\n")

	net = LeNet5()
	image = random_input(DETERMINISM_INPUT_SEED)

	output1 = net.forward(image)
	output2 = net.forward(image)

	print("First run output:")
	print_output(output1)
	print("\nSecond run output:")
	print_output(output2)

	identical = bool(np.all(np.abs(output1 - output2) <= IDENTICAL_TOLERANCE))

	print(f"\nOutputs identical: {identical}")
	print_result(identical, "Network is deterministic", "Network is not deterministic")
	return identical


def test_different_inputs() -> bool:
	print("\n=== Different Inputs Test ===")
	print("Testing that different inputs produce different outputs...\n")

	net = LeNet5()
	output1 = net.forward(random_input(FIRST_INPUT_SEED))
	output2 = net.forward(random_input(SECOND_INPUT_SEED))

	print("Input 1 output:")
	print_output(output1)
	print("\nInput 2 output:")
	print_output(output2)

	different = bool(np.any(np.abs(output1 - output2) > DIFFERENT_TOLERANCE))

	print(f"\nOutputs different: {different}")
	print_result(different, "Network discriminates between inputs", "Network produces same output for different inputs")
	return different


def test_zero_input() -> bool:
	print("\n=== Zero Input Test ===")
	print("Testing network with all-zero input...\n")

	net = LeNet5()
	output = net.forward(np.zeros(INPUT_SHAPE))

	print("Zero input output:")
	print_output(output)

	valid = is_valid_distribution(output)
	print_result(valid, "Zero input produces valid output", "Zero input produces invalid output")
	return valid


def test_ones_input() -> bool:
	print("\n=== Ones Input Test ===")
	print("Testing network with all-ones input...\n")

	net = LeNet5()
	output = net.forward(np.ones(INPUT_SHAPE))

	print("Ones input output:")
	print_output(output)

	valid = is_valid_distribution(output)
	print_result(valid, "Ones input produces valid output", "Ones input produces invalid output")
	return valid


def main() -> bool:
	print("LeNet-5 Testing Suite\n")

	results = [
		test_determinism(),
		test_different_inputs(),
		test_zero_input(),
		test_ones_input(),
	]

	print("\n=== Testing Complete ===")
	print(f"{sum(results)} of {len(results)} tests passed")
	return all(results)


if __name__ == "__main__":
	main()