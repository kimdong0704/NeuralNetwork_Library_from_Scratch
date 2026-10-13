import numpy as np

from numpy_utils.config import DTYPE
from numpy_utils.activations import Activation

from .config import CHANNEL_AXIS
from .utils import windows_fast, windows_loop

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

		channels = self.last_input.shape[CHANNEL_AXIS]
		if channels != self.in_channels:
			raise ValueError(f"input has {channels} channels, kernels expect {self.in_channels}")

		# self.last_windows = windows_loop(
		#     inputs=self.last_input,
		#     window_rows=self.kernel_rows,
		#     window_cols=self.kernel_cols,
		#     stride=self.stride
		# )
		self.last_windows = windows_fast(
			inputs=self.last_input,
			window_rows=self.kernel_rows,
			window_cols=self.kernel_cols,
			stride=self.stride
		)

		self.last_z = self._convolve(self.last_windows)
		self.last_output = self.activation.function(self.last_z)

		return self.last_output

	def _convolve(self, windows: np.ndarray) -> np.ndarray:
		# matmul weights (out_ch, in_ch, kr, kc)  by windows (in_ch, kr, kc, out_r, out_c)
		# output = (out_ch, out_r, out_c)
		z = np.tensordot(self.weights, windows, axes=3) + self.bias
		return z 