import numpy as np

from numpy_utils.config import DTYPE
from numpy_utils.activations import Activation

from .config import CHANNEL_AXIS, WINDOW_CHANNEL_AXIS, WINDOW_ROW_AXIS, WINDOW_COL_AXIS
from .utils import check_image_batch, fold_windows, windows_fast, windows_loop

# weights (out_ch, in_ch, kr, kc): the axes matched against the windows
WEIGHT_SUM_AXES = (1, 2, 3)
WINDOW_SUM_AXES = (WINDOW_CHANNEL_AXIS, WINDOW_ROW_AXIS, WINDOW_COL_AXIS)

# dz (batch, out_ch, out_r, out_c) and windows (batch, in_ch, kr, kc, out_r, out_c): batch and output pixel axes
DZ_SUM_AXES = (0, 2, 3)
WINDOW_POSITION_AXES = (0, 4, 5)

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

	@property
	def differentiable(self) -> bool:
		return self.activation.derivative is not None

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		self.last_inputs = np.asarray(inputs, dtype=DTYPE)
		check_image_batch(self.last_inputs)

		channels = self.last_inputs.shape[CHANNEL_AXIS]
		if channels != self.in_channels:
			raise ValueError(f"input has {channels} channels, kernels expect {self.in_channels}")

		# self.last_windows = windows_loop(
		#     inputs=self.last_inputs,
		#     window_rows=self.kernel_rows,
		#     window_cols=self.kernel_cols,
		#     stride=self.stride
		# )
		self.last_windows = windows_fast(
			inputs=self.last_inputs,
			window_rows=self.kernel_rows,
			window_cols=self.kernel_cols,
			stride=self.stride
		)

		self.last_z = self._convolve(self.last_windows)
		self.last_outputs = self.activation.function(self.last_z)

		return self.last_outputs

	def _convolve(self, windows: np.ndarray) -> np.ndarray:
		# matmul windows (batch, in_ch, kr, kc, out_r, out_c) by weights (out_ch, in_ch, kr, kc)
		# tensordot gives (batch, out_r, out_c, out_ch) -> move out_ch to (batch, out_ch, out_r, out_c)
		z = np.tensordot(windows, self.weights, axes=(WINDOW_SUM_AXES, WEIGHT_SUM_AXES))
		z = np.moveaxis(z, -1, CHANNEL_AXIS) + self.bias
		return z

	def backward(self, deltas: np.ndarray) -> np.ndarray:
		dz = self._dz(deltas)

		# each weight touched one pixel of every window, so sum dz * that pixel over batch and output pixels
		# dz (batch, out_ch, out_r, out_c) by windows (batch, in_ch, kr, kc, out_r, out_c) -> (out_ch, in_ch, kr, kc)
		self.weight_gradient = np.tensordot(dz, self.last_windows, axes=(DZ_SUM_AXES, WINDOW_POSITION_AXES))
		self.bias_gradient = dz.sum(axis=DZ_SUM_AXES).reshape(self.bias.shape)

		# each window pixel fed every out channel through its weight
		# dz (batch, out_ch, out_r, out_c) by weights (out_ch, in_ch, kr, kc) -> (batch, out_r, out_c, in_ch, kr, kc)
		window_grads = np.tensordot(dz, self.weights, axes=([CHANNEL_AXIS], [0]))
		# -> (batch, in_ch, kr, kc, out_r, out_c), the layout of the windows
		window_grads = np.moveaxis(window_grads, (3, 4, 5), WINDOW_SUM_AXES)

		propagated = fold_windows(window_grads, self.last_inputs.shape, self.stride)

		return propagated

	def step(self, learning_rate: float) -> None:
		self.weights -= learning_rate * self.weight_gradient
		self.bias -= learning_rate * self.bias_gradient

	def _dz(self, deltas: np.ndarray) -> np.ndarray:
		# without a derivative (STEP) the deltas pass straight through, same as numpy_nn.Layer
		if self.activation.derivative is None:
			return deltas
		return self.activation.derivative(self.last_outputs, deltas)
