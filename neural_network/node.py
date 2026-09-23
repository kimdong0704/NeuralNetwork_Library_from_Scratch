from typing import Callable


class Node:
    def __init__(
        self,
        weights: list[float],
        activation_function: Callable[[float], float],
        activation_derivative: Callable[[float], float] | None,
        bias: float,
    ):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.gradient_descent = activation_derivative is not None

    def forward(self, inputs: list[float]) -> float:
        self.last_input = inputs

        total = self.bias
        for weight, input in zip(self.weights, inputs):
            total += weight * input

        self.last_output = self.activation_function(total)
        return self.last_output

    def backward(self, delta: float, learning_rate: float) -> list[float]:
        if self.gradient_descent:
            assert self.activation_derivative is not None
            dz = delta * self.activation_derivative(self.last_output)
        else:
            dz = delta

        step = learning_rate * dz

        propagated = []
        for index, (weight, input_value) in enumerate(zip(self.weights, self.last_input)):
            propagated.append(weight * dz)
            self.weights[index] += step * input_value

        self.bias += step

        return propagated
