import functools
from collections.abc import Sequence

import numpy as np

from .activations import ACTIVATIONS


class Node:
    """A single neuron: owns its weights, bias, and activation.

    This is the old Perceptron, renamed and backed by numpy, so a lone Node
    can still do everything a standalone perceptron did before (including
    the step-function / single-node training path).
    """

    def __init__(
        self,
        input_size: int,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
        learning_rate: float = 0.01,
        activation_function: str = "step",
        step_threshold: float = 0.0,
        gradient_descent: bool = False
    ):
        self.input_size = input_size
        self.bias = bias
        self.learning_rate = learning_rate
        self.step_threshold = step_threshold
        self.gradient_descent = gradient_descent
        self.activation_name = activation_function

        if weights is None:
            self.weights = np.zeros(input_size, dtype=float)
        else:
            if len(weights) != input_size:
                raise ValueError(
                    f"weights has length {len(weights)}, expected {input_size}"
                )

            self.weights = np.array(weights, dtype=float)

        if activation_function not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation_function: {activation_function!r}. "
                f"Expected one of {list(ACTIVATIONS)}."
            )

        entry = ACTIVATIONS[activation_function]
        self.activation_derivative = entry["derivative"]

        if activation_function == "step":
            self.activation_function = functools.partial(
                entry["function"], threshold=self.step_threshold
            )
        else:
            self.activation_function = entry["function"]

        if self.gradient_descent and self.activation_derivative is None:
            raise ValueError(
                "gradient_descent=True requires an activation function with a "
                f"usable derivative ({activation_function!r} has none)."
            )

    def sum(self, inputs: Sequence[float]) -> float:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.shape[0] != self.input_size:
            raise ValueError(
                f"inputs has length {inputs.shape[0]}, expected {self.input_size}"
            )

        return float(np.dot(self.weights, inputs) + self.bias)

    def predict(self, inputs: Sequence[float]) -> float:
        total = self.sum(inputs)
        return self.activation_function(total)

    def update(self, inputs: Sequence[float], prediction: float, error: float) -> None:
        inputs = np.asarray(inputs, dtype=float)

        if self.gradient_descent:
            delta = self.learning_rate * error * self.activation_derivative(prediction)
        else:
            delta = self.learning_rate * error

        self.weights += delta * inputs
        self.bias += delta
