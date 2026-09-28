from .activations import Activation
from .config import Matrix, Vector
from .node import Node


class Layer:
    """A fully connected layer of Nodes; the activation runs over the whole layer so softmax can couple its outputs."""

    def __init__(self, nodes: list[Node], activation: Activation):
        self.nodes = nodes
        self.activation = activation

    @classmethod
    def build(
        cls,
        input_size: int,
        output_size: int,
        weight: Matrix,
        bias: Vector,
        activation: Activation
    ) -> "Layer":
        if len(weight) != output_size:
            raise ValueError(f"weight has {len(weight)} rows, expected {output_size}")

        if len(bias) != output_size:
            raise ValueError(f"bias has {len(bias)} values, expected {output_size}")

        nodes = []
        for row, node_bias in zip(weight, bias):
            if len(row) != input_size:
                raise ValueError(f"weight row has {len(row)} values, expected {input_size}")

            nodes.append(Node(weights=[float(value) for value in row], bias=float(node_bias)))

        return cls(nodes, activation)

    @property
    def differentiable(self) -> bool:
        return self.activation.derivative is not None

    @property
    def weights(self) -> Matrix:
        weights = []
        for node in self.nodes:
            weights.append(node.weights)
        return weights

    @property
    def bias(self) -> Vector:
        bias = []
        for node in self.nodes:
            bias.append(node.bias)
        return bias

    def forward(self, inputs: Matrix) -> Matrix:
        self.last_input = inputs

        outputs = []
        for row in inputs:
            z = []
            for node in self.nodes:
                z.append(node.forward(row))
            outputs.append(self.activation.function(z))

        self.last_output = outputs

        return self.last_output

    def backward(self, deltas: Matrix) -> Matrix:
        """Stores the cost gradients of every node, and returns the deltas for the previous layer."""
        dz = []
        for output_row, delta_row in zip(self.last_output, deltas):
            dz.append(self._dz(output_row, delta_row))

        for index, node in enumerate(self.nodes):
            node_dz = []
            for dz_row in dz:
                node_dz.append(dz_row[index])
            node.backward(self.last_input, node_dz)

        propagated = []
        for dz_row in dz:
            row = [0.0] * len(self.nodes[0].weights)
            for node, d in zip(self.nodes, dz_row):
                for index, weight in enumerate(node.weights):
                    row[index] += weight * d
            propagated.append(row)

        return propagated

    def step(self, learning_rate: float) -> None:
        for node in self.nodes:
            node.step(learning_rate)

    def _dz(self, output_row: Vector, delta_row: Vector) -> Vector:
        # without a derivative (STEP) the deltas pass straight through, as in the perceptron learning rule
        if self.activation.derivative is None:
            return list(delta_row)
        return self.activation.derivative(output_row, delta_row)
