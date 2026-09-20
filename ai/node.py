from typing import Callable


class Node:
    """A single neuron: its weights, bias, and the calculations done for it."""

    def __init__(
        self,
        weights: list[float],
        activation_function: Callable[[float], float],
        activation_derivative: Callable[[float], float] | None,
    ):
        self.weights = weights
        self.bias = 0.0
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.gradient_descent = activation_derivative is not None

    def forward(self, inputs: list[float]) -> float:
        self.last_input = inputs

        total = self.bias
        for index in range(len(inputs)):
            total += self.weights[index] * inputs[index]

        self.last_output = self.activation_function(total)
        return self.last_output

    def backward(self, delta: float, learning_rate: float) -> list[float]:
        if self.gradient_descent:
            assert self.activation_derivative is not None
            amount_change = delta * self.activation_derivative(self.last_output)
        else:
            amount_change = delta

        step = learning_rate * amount_change
        
        propagated = []
        for index in range(len(self.weights)):
            propagated.append(self.weights[index] * amount_change)
            self.weights[index] += step * self.last_input[index]

        self.bias += step

        return propagated
