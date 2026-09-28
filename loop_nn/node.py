from .config import Matrix, Vector


class Node:
    """A single neuron: one weight per input plus a bias, combined into a weighted sum."""

    def __init__(self, weights: Vector, bias: float):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs: Vector) -> float:
        total = self.bias
        for weight, value in zip(self.weights, inputs):
            total += weight * value
        return total

    def backward(self, inputs: Matrix, dz: Vector) -> None:
        """Stores the cost gradients of the weights and bias, summed over a batch; dz holds this node's dz per sample."""
        self.weight_gradient = [0.0] * len(self.weights)
        self.bias_gradient = 0.0

        for row, d in zip(inputs, dz):
            for index, value in enumerate(row):
                self.weight_gradient[index] += d * value
            self.bias_gradient += d

    def step(self, learning_rate: float) -> None:
        for index, gradient in enumerate(self.weight_gradient):
            self.weights[index] -= learning_rate * gradient
        self.bias -= learning_rate * self.bias_gradient
