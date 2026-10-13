import numpy as np

from numpy_utils.config import DTYPE

from .config import CHANNEL_AXIS, ROW_AXIS, COL_AXIS

class Pad:
	def __init__(self, amount: int):
		if amount < 0:
			raise ValueError("Padding amount must be non-negative.")
		
		self.amount = amount

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		self.last_inputs = np.asarray(inputs, dtype=DTYPE)
		channels = self.last_inputs.shape[CHANNEL_AXIS]
		rows = self.last_inputs.shape[ROW_AXIS]
		cols = self.last_inputs.shape[COL_AXIS]

		padded = np.zeros((channels, rows + 2 * (self.amount), cols + 2 * self.amount), dtype=DTYPE)
		padded[:, self.amount:self.amount + rows, self.amount:self.amount + cols] = self.last_inputs

		self.last_outputs = padded

		return self.last_outputs