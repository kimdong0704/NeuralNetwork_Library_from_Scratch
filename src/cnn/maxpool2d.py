import numpy as np

from numpy_utils.config import DTYPE

from .config import WINDOW_ROW_AXIS, WINDOW_COL_AXIS
from .utils import windows_fast, windows_loop

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