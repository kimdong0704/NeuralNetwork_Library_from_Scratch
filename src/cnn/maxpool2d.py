import numpy as np

from numpy_utils.config import DTYPE

from .config import WINDOW_ROW_AXIS, WINDOW_COL_AXIS
from .utils import check_image_batch, fold_windows, windows_fast, windows_loop

class MaxPool2D:
	def __init__(self, pool_rows: int, pool_cols: int, stride: int):
		if pool_rows < 1 or pool_cols < 1:
			raise ValueError(f"pool size must be 1 or more, got ({pool_rows}, {pool_cols})")
		if stride < 1:
			raise ValueError(f"stride must be 1 or more, got {stride}")

		self.pool_rows = pool_rows
		self.pool_cols = pool_cols
		self.stride = stride

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		self.last_inputs = np.asarray(inputs, dtype=DTYPE)
		check_image_batch(self.last_inputs)

		# self.last_windows = windows_loop(
		#     inputs=self.last_inputs,
		#     window_rows=self.pool_rows,
		#     window_cols=self.pool_cols,
		#     stride=self.stride
		# )
		self.last_windows = windows_fast(
			inputs=self.last_inputs,
			window_rows=self.pool_rows,
			window_cols=self.pool_cols,
			stride=self.stride
		)

		# max over pool rows and cols
		self.last_outputs = np.max(self.last_windows, axis=(WINDOW_ROW_AXIS, WINDOW_COL_AXIS))

		return self.last_outputs

	def backward(self, deltas: np.ndarray) -> np.ndarray:
		batch, channels, pool_rows, pool_cols, out_rows, out_cols = self.last_windows.shape

		# lay each window out as a list of its pixels: (batch, ch, pool_r * pool_c, out_r, out_c)
		flat_windows = self.last_windows.reshape(batch, channels, pool_rows * pool_cols, out_rows, out_cols)

		# which pixel of each window was the max (ties go to the first one): (batch, ch, 1, out_r, out_c)
		max_positions = np.argmax(flat_windows, axis=2, keepdims=True)

		# every window pixel starts at 0, then each window's delta goes to its max pixel
		flat_grads = np.zeros(flat_windows.shape, dtype=DTYPE)
		# holds delta for that window
		window_deltas = np.expand_dims(deltas, axis=2)  # (batch, ch, out_r, out_c) -> (batch, ch, 1, out_r, out_c)
		# writes the delta into the flat_grads at given position, rest is 0
		np.put_along_axis(flat_grads, max_positions, window_deltas, axis=2)

		# back to the window layout (batch, ch, pool_r, pool_c, out_r, out_c)
		window_grads = flat_grads.reshape(self.last_windows.shape)

		propagated = fold_windows(window_grads, self.last_inputs.shape, self.stride)

		return propagated

	def step(self, learning_rate: float) -> None:
		# no params to update
		pass
