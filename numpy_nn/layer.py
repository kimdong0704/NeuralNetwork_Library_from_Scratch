import numpy as np

from .activations import Activation
from .config import DTYPE


class Layer:
    """A fully connected layer: outputs = activation(inputs @ weights.T + bias), one row per sample."""

    def __init__(self, weights: np.ndarray, bias: np.ndarray, activation: Activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    @classmethod
    def build(
        cls,
        input_size: int,
        output_size: int,
        weight: np.ndarray,
        bias: np.ndarray,
        activation: Activation
    ) -> "Layer":
        weights = np.array(weight, dtype=DTYPE)
        bias_values = np.array(bias, dtype=DTYPE)

        if weights.shape != (output_size, input_size):
            raise ValueError(f"weight has shape {weights.shape}, expected {(output_size, input_size)}")

        if bias_values.shape != (output_size,):
            raise ValueError(f"bias has shape {bias_values.shape}, expected {(output_size,)}")

        return cls(weights, bias_values, activation)

    @property
    def differentiable(self) -> bool:
        return self.activation.derivative is not None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input = inputs

        z = inputs @ self.weights.T
        z += self.bias

        self.last_output = self.activation.function(z)

        return self.last_output

    def backward(self, deltas: np.ndarray) -> np.ndarray:
        """Stores the cost gradients of the weights and bias, and returns the deltas for the previous layer."""
        dz = self._dz(deltas)

        self.weight_gradient = dz.T @ self.last_input
        self.bias_gradient = dz.sum(axis=0)

        propagated = dz @ self.weights

        return propagated

    def step(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.weight_gradient
        self.bias -= learning_rate * self.bias_gradient

    def _dz(self, deltas: np.ndarray) -> np.ndarray:
        # without a derivative (STEP) the deltas pass straight through, as in the perceptron learning rule
        if self.activation.derivative is None:
            return deltas
        return self.activation.derivative(self.last_output, deltas)
