import numpy as np

from numpy_utils.config import DTYPE

from .config import BATCH_AXIS

class Flatten:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_inputs = np.asarray(inputs, dtype=DTYPE)

        # (batch, channels, rows, cols) -> (batch, channels * rows * cols), the rows numpy_nn.Layer expects
        batch = self.last_inputs.shape[BATCH_AXIS]
        self.last_outputs = self.last_inputs.reshape(batch, -1)

        return self.last_outputs

    def backward(self, deltas: np.ndarray) -> np.ndarray:
        propagated = deltas.reshape(self.last_inputs.shape)

        return propagated

    def step(self, learning_rate: float) -> None:
        # no params to update
        pass
