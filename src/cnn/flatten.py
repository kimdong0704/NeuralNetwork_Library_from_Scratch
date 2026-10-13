import numpy as np

from numpy_utils.config import DTYPE

class Flatten:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_inputs = np.asarray(inputs, dtype=DTYPE)

        # (channels, rows, cols) -> (channels * rows * cols)
        self.last_outputs = self.last_inputs.reshape(-1)

        return self.last_outputs