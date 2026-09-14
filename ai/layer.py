from collections.abc import Sequence

import numpy as np

from .node import Node


class Layer:
    """An ordered list of Nodes that share a forward pass.

    Each Node still owns its own weights/bias as the source of truth;
    forward() just stacks them into a matrix on the fly so a layer with
    many nodes runs as one numpy matmul instead of a Python loop.
    """

    def __init__(self, nodes: Sequence[Node]):
        self.nodes = list(nodes)

    def forward(self, inputs: Sequence[float]) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)

        weights_matrix = np.stack([node.weights for node in self.nodes])
        bias_vector = np.array([node.bias for node in self.nodes])
        totals = weights_matrix @ inputs + bias_vector

        return np.array(
            [node.activation_function(total) for node, total in zip(self.nodes, totals)]
        )
