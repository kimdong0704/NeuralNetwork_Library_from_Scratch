import numpy as np

from numpy_utils.config import DTYPE

from .config import BATCH_AXIS, CHANNEL_AXIS, ROW_AXIS, COL_AXIS
from .utils import check_image_batch

class Pad:
	def __init__(self, amount: int):
		if amount < 0:
			raise ValueError("Padding amount must be non-negative.")
		
		self.amount = amount

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		self.last_inputs = np.asarray(inputs, dtype=DTYPE)
		check_image_batch(self.last_inputs)

		batch = self.last_inputs.shape[BATCH_AXIS]
		channels = self.last_inputs.shape[CHANNEL_AXIS]
		rows = self.last_inputs.shape[ROW_AXIS]
		cols = self.last_inputs.shape[COL_AXIS]

		padded = np.zeros((batch, channels, rows + 2 * self.amount, cols + 2 * self.amount), dtype=DTYPE)
		padded[:, :, self.amount:self.amount + rows, self.amount:self.amount + cols] = self.last_inputs

		self.last_outputs = padded

		return self.last_outputs

	def backward(self, deltas: np.ndarray) -> np.ndarray:
		rows = self.last_inputs.shape[ROW_AXIS]
		cols = self.last_inputs.shape[COL_AXIS]

		propagated = deltas[:, :, self.amount:self.amount + rows, self.amount:self.amount + cols]

		return propagated

	def step(self, learning_rate: float) -> None:
		# no weights to update
		pass
