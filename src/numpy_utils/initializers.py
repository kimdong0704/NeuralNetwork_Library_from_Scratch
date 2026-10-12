import numpy as np


# Initialize Weights Functions
def random_weights(input_size: int, output_size: int, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    return np.random.uniform(low, high, size=(output_size, input_size))

def random_weights_conv(
    in_channels: int,
    out_channels: int,
    kernel_rows: int,
    kernel_cols: int,
    low: float = -1.0,
    high: float = 1.0
) -> np.ndarray:
    return np.random.uniform(low, high, size=(out_channels, in_channels, kernel_rows, kernel_cols))

# Initialize Biases Functions
def zero_bias(output_size: int) -> np.ndarray:
    return np.zeros(output_size)

def zero_bias_conv(out_channels: int) -> np.ndarray:
    # (out_chanels, out_h, out_w)
    return np.zeros((out_channels, 1, 1))
