from typing import Callable

import numpy as np

from .config import DTYPE


class Layer:
    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        activation_function: Callable[[np.ndarray], np.ndarray],
        activation_derivative: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    ):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.gradient_descent = activation_derivative is not None

    @classmethod
    def build(
        cls,
        input_size: int,
        output_size: int,
        weight: np.ndarray,
        bias: np.ndarray,
        activation_function: Callable[[np.ndarray], np.ndarray],
        activation_derivative: Callable[[np.ndarray, np.ndarray], np.ndarray] | None
    ) -> "Layer":
        weights = np.array(weight, dtype=DTYPE)
        bias_values = np.array(bias, dtype=DTYPE)

        if weights.shape != (output_size, input_size):
            raise ValueError(f"weight has shape {weights.shape}, expected {(output_size, input_size)}")

        if bias_values.shape != (output_size,):
            raise ValueError(f"bias has shape {bias_values.shape}, expected {(output_size,)}")

        return cls(weights, bias_values, activation_function, activation_derivative)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input = inputs

        z = inputs @ self.weights.T
        z += self.bias
        self.last_output = self.activation_function(z)
        return self.last_output

    def backward(self, deltas: np.ndarray, learning_rate: float) -> np.ndarray:
        """Updates the weights and returns the deltas for the previous layer."""
        dz = self._dz(deltas)
        # computed before the update so the previous layer sees the weights that produced the output
        propagated = dz @ self.weights
        self._apply(dz, learning_rate)
        return propagated

    def update(self, deltas: np.ndarray, learning_rate: float) -> None:
        """Updates the weights like backward, without computing deltas for a previous layer."""
        self._apply(self._dz(deltas), learning_rate)

    def _dz(self, deltas: np.ndarray) -> np.ndarray:
        if self.activation_derivative is None:
            return deltas
        return self.activation_derivative(self.last_output, deltas)

    def _apply(self, dz: np.ndarray, learning_rate: float) -> None:
        step = learning_rate * dz
        self.weights += step.T @ self.last_input
        self.bias += step.sum(axis=0)
