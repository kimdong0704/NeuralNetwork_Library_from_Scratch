from collections.abc import Sequence
from typing import Callable

import numpy as np


class Node:
    def __init__(
        self,
        input_size: int,
        weights: Sequence[float] | np.ndarray,
        bias: float,
        activation_function: Callable[[float], float],
        activation_derivative: Callable[[float], float] | None,
        gradient_descent: bool
    ):
        self.input_size = input_size
        self.weights = np.array(weights, dtype=float)
        self.bias = bias
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.gradient_descent = gradient_descent

    def sum(self, inputs: Sequence[float]) -> float:
        values = np.asarray(inputs, dtype=float)

        if values.shape[0] != self.input_size:
            raise ValueError(
                f"inputs has length {values.shape[0]}, expected {self.input_size}"
            )

        # dot product of weights and inputs + bias
        return float(np.dot(self.weights, values) + self.bias)

    # this is y_hat
    def predict(self, inputs: Sequence[float]) -> float:
        total = self.sum(inputs)
        return self.activation_function(total)

    def update(
        self,
        inputs: Sequence[float],
        prediction: float,
        error: float,
        learning_rate: float
    ) -> None:
        values = np.asarray(inputs, dtype=float)

        if self.gradient_descent:
            if self.activation_derivative is None:
                raise ValueError(
                    "gradient_descent=True requires an activation_derivative, "
                    "got None."
                )

            delta = learning_rate * error * self.activation_derivative(prediction)
        else:
            delta = learning_rate * error

        self.weights += delta * values
        self.bias += delta * 1
