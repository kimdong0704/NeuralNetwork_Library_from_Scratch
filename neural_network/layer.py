from typing import Callable

from .activations import identity_derivative, identity_function, softmax_derivative, softmax_function
from .node import Node


class Layer:
    def __init__(self, nodes: list[Node], softmax: bool = False):
        self.nodes = nodes
        self.softmax = softmax

        self.gradient_descent = True
        for node in nodes:
            if not node.gradient_descent:
                self.gradient_descent = False
                break

    @classmethod
    def build(
        cls,
        input_size: int,
        output_size: int,
        weight: list[list[float]],
        bias: list[float],
        activation_function: Callable[..., float | list[float]],
        activation_derivative: Callable[..., float | list[float]] | None
    ) -> "Layer":
        if len(weight) != output_size:
            raise ValueError(f"weight has {len(weight)} rows, expected {output_size}")

        if len(bias) != output_size:
            raise ValueError(f"bias has {len(bias)} values, expected {output_size}")

        # softmax needs every node's output at once, so the nodes stay linear and the layer applies it
        softmax = activation_function is softmax_function
        if softmax:
            activation_function = identity_function
            activation_derivative = identity_derivative

        nodes = []
        for row, node_bias in zip(weight, bias):
            if len(row) != input_size:
                raise ValueError(f"weight row has {len(row)} values, expected {input_size}")

            nodes.append(
                Node(
                    weights=list(map(float, row)),
                    activation_function=activation_function,
                    activation_derivative=activation_derivative,
                    bias=float(node_bias),
                )
            )

        return cls(nodes, softmax=softmax)

    @property
    def weights(self) -> list[list[float]]:
        weights = []
        for node in self.nodes:
            weights.append(node.weights)
        return weights

    @property
    def bias(self) -> list[float]:
        bias = []
        for node in self.nodes:
            bias.append(node.bias)
        return bias

    def forward(self, inputs: list[float]) -> list[float]:
        outputs = []
        for node in self.nodes:
            outputs.append(node.forward(inputs))

        if self.softmax:
            outputs = softmax_function(outputs)
            self.last_output = outputs

        return outputs

    def backward(self, deltas: list[float], learning_rate: float) -> list[float]:
        if self.softmax:
            deltas = softmax_derivative(self.last_output, deltas)

        propagated = [0.0] * len(self.nodes[0].weights)

        for node, delta in zip(self.nodes, deltas):
            contributions = node.backward(delta, learning_rate)
            
            for index, contribution in enumerate(contributions):
                propagated[index] += contribution

        return propagated
