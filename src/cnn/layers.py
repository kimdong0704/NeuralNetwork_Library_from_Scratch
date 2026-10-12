import numpy as np
from numpy.lib.stride_tricks import as_strided

from numpy_utils.config import DTYPE
from numpy_utils.activations import Activation

from .config import CHANNEL_AXIS, ROW_AXIS, COL_AXIS


class Pad:
    def __init__(self, amount: int):
        if amount < 0:
            raise ValueError("Padding amount must be non-negative.")
        
        self.amount = amount

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input = np.asarray(inputs, dtype=DTYPE)
        channels, rows, cols = self.last_input.shape

        padded = np.zeros((channels, rows + 2 * (self.amount), cols + 2 * self.amount), dtype=DTYPE)
        padded[:, self.amount:self.amount + rows, self.amount:self.amount + cols] = self.last_input

        self.last_output = padded

        return self.last_output

class Conv2D:
    def __init__(self, weights: np.ndarray, bias: np.ndarray, activation: Activation, stride: int = 1):
        weights = np.asarray(weights, dtype=DTYPE)
        bias = np.asarray(bias, dtype=DTYPE)

        if weights.ndim != 4:
            raise ValueError(f"weights must be (out, in, kernel_rows, kernel_cols), got shape {weights.shape}")
        if bias.shape != (weights.shape[0], 1, 1):
            raise ValueError(f"bias has shape {bias.shape}, expected {(weights.shape[0], 1, 1)}")
        if stride < 1:
            raise ValueError(f"stride must be 1 or more, got {stride}")

        self.weights = weights
        self.out_channels, self.in_channels, self.kernel_rows, self.kernel_cols = weights.shape
        self.bias = bias
        self.activation = activation
        self.stride = stride

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input = np.asarray(inputs, dtype=DTYPE)

        channels, rows, cols = self.last_input.shape
        if channels != self.in_channels:
            raise ValueError(f"input has {channels} channels, kernels expect {self.in_channels}")

        out_rows = ((rows - self.kernel_rows) // self.stride) + 1
        out_cols = ((cols - self.kernel_cols) // self.stride) + 1

        self.last_windows = self._windows_fast(self.last_input, out_rows, out_cols)
        self.last_z = self._convolve(self.last_windows)
        self.last_output = self.activation.function(self.last_z)

        return self.last_output

    def _windows(self, inputs: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
        # [:, :, :, row, col]
        # (in, kr, kc) --> weight matmul
        windows = np.empty((self.in_channels, self.kernel_rows, self.kernel_cols, out_rows, out_cols), dtype=DTYPE)

        for r in range(self.kernel_rows):
            for c in range(self.kernel_cols):
                windows[:, r, c] = inputs[:, r:r + self.stride * out_rows:self.stride, c:c + self.stride * out_cols:self.stride]

        return windows

    def _windows_fast(self, inputs: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
        channel_step, r_step, c_step = inputs.strides

        return as_strided(
            inputs,
            shape=(self.in_channels, self.kernel_rows, self.kernel_cols, out_rows, out_cols),
            strides=(channel_step, r_step, c_step, self.stride * r_step, self.stride * c_step),
            writeable=False,
        )

    def _convolve(self, windows: np.ndarray) -> np.ndarray:
        # matmul weights (out, in, kr, kc)  by windows (in, kr, kc, out_r, out_c)
        # output = (out, out_r, out_c)
        z = np.tensordot(self.weights, windows, axes=3) + self.bias
        return z 
