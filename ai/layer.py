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
        totals = self.weights @ inputs + self.bias
        return self.activation_function(totals)

    def update(
        self,
        index: int,
        inputs: np.ndarray,
        prediction: float,
        error: float,
        learning_rate: float,
    ) -> None:
        if self.gradient_descent:
            assert self.activation_derivative is not None
            derivative = float(self.activation_derivative(np.array([prediction]))[0])
            delta = learning_rate * error * derivative
        else:
            delta = learning_rate * error

        self.weights[index] += delta * inputs
        self.bias[index] += delta
