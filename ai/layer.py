from typing import Callable

import numpy as np


class Layer:
    def __init__(
        self,
        weights: np.ndarray,
        activation_function: Callable[[np.ndarray], np.ndarray],
        activation_derivative: Callable[[np.ndarray], np.ndarray] | None,
    ):
        self.weights = weights
        self.bias = np.zeros(weights.shape[0], dtype=float)
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.gradient_descent = activation_derivative is not None

    @classmethod
    def build(
        cls,
        input_size: int,
        output_size: int,
        weight: np.ndarray,
        activation_function: Callable[[np.ndarray], np.ndarray],
        activation_derivative: Callable[[np.ndarray], np.ndarray] | None,
    ) -> "Layer":
        weights = np.asarray(weight, dtype=float)
        if weights.shape != (output_size, input_size):
            raise ValueError(
                f"weight has shape {weights.shape}, expected {(output_size, input_size)}"
            )

        return cls(
            weights=weights,
            activation_function=activation_function,
            activation_derivative=activation_derivative,
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input = inputs
        totals = self.weights @ inputs + self.bias
        self.last_output = self.activation_function(totals)
        return self.last_output

    def backward(self, delta: np.ndarray, learning_rate: float) -> np.ndarray:
        """Computes this layer's weight/bias update and returns the error term
        to propagate to the previous layer.

        `delta` is dLoss/dOutput for this layer: the raw
        (target - prediction) error for an output layer, or the propagated
        gradient handed down from the next layer for a hidden layer.
        Requires `forward` to have been called first (uses the cached
        `last_input`/`last_output` from that pass).
        """
        if self.gradient_descent:
            assert self.activation_derivative is not None
            amount_change = delta * self.activation_derivative(self.last_output)
        else:
            amount_change = delta

        propagated = self.weights.T @ amount_change

        self.weights += learning_rate * np.outer(amount_change, self.last_input)
        self.bias += learning_rate * amount_change

        return propagated
