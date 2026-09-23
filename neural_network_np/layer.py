from typing import Callable

import numpy as np


class Layer:
    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        activation_function: Callable[[np.ndarray], np.ndarray],
        activation_derivative: Callable[[np.ndarray], np.ndarray] | None,
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
        activation_derivative: Callable[[np.ndarray], np.ndarray] | None
    ) -> "Layer":
        weights = np.array(weight, dtype=float)
        bias_values = np.array(bias, dtype=float)

        if weights.shape != (output_size, input_size):
            raise ValueError(f"weight has shape {weights.shape}, expected {(output_size, input_size)}")

        if bias_values.shape != (output_size,):
            raise ValueError(f"bias has shape {bias_values.shape}, expected {(output_size,)}")

        return cls(weights, bias_values, activation_function, activation_derivative)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input = np.asarray(inputs, dtype=float)

        z = self.last_input @ self.weights.T + self.bias
        self.last_output = self.activation_function(z)
        return self.last_output

    def backward(self, deltas: np.ndarray, learning_rate: float) -> np.ndarray:
        deltas = np.asarray(deltas, dtype=float)

        if self.gradient_descent:
            assert self.activation_derivative is not None
            dz = deltas * self.activation_derivative(self.last_output)
        else:
            dz = deltas

        step = learning_rate * dz

        propagated = dz @ self.weights

        self.weights += step.T @ self.last_input
        self.bias += step.sum(axis=0)

        return propagated
