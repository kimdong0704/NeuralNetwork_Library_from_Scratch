import numpy as np


# Initialize Weights Functions
def zero_weights(input_size: int, output_size: int) -> np.ndarray:
    return np.zeros((output_size, input_size))

def random_weights(input_size: int, output_size: int, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    return np.random.uniform(low, high, size=(output_size, input_size))

# Initialize Biases Functions
def zero_bias(output_size: int) -> np.ndarray:
    return np.zeros(output_size)

def random_bias(output_size: int, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    return np.random.uniform(low, high, size=output_size)
