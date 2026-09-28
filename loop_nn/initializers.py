import random

from .config import Matrix, Vector


# Initialize Weights Functions
def random_weights(input_size: int, output_size: int, low: float = -1.0, high: float = 1.0) -> Matrix:
    weights = []
    for _ in range(output_size):
        row = []
        for _ in range(input_size):
            row.append(random.uniform(low, high))
        weights.append(row)
    return weights

# Initialize Biases Functions
def zero_bias(output_size: int) -> Vector:
    return [0.0] * output_size
