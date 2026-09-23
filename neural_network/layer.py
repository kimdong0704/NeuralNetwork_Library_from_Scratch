from typing import Callable

from .node import Node


class Layer:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

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
        activation_function: Callable[[float], float],
        activation_derivative: Callable[[float], float] | None
    ) -> "Layer":
        if len(weight) != output_size:
            raise ValueError(f"weight has {len(weight)} rows, expected {output_size}")

        if len(bias) != output_size:
            raise ValueError(f"bias has {len(bias)} values, expected {output_size}")

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

        return cls(nodes)

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
        return outputs

    def backward(self, deltas: list[float], learning_rate: float) -> list[float]:
        propagated = [0.0] * len(self.nodes[0].weights)

        for node, delta in zip(self.nodes, deltas):
            contributions = node.backward(delta, learning_rate)
            
            for index, contribution in enumerate(contributions):
                propagated[index] += contribution

        return propagated
