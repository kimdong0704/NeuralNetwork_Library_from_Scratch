from collections.abc import Sequence

import numpy as np

from .layer import Layer


class NeuralNetwork:
    """An ordered list of Layers, chaining forward() across them.

    Training currently only works when the network is a single layer with
    a single node (mirrors a lone Perceptron exactly). A multi-layer/
    multi-node network can still forward()/predict(); trainer.py raises a
    clear error if you try to train one, instead of quietly computing
    garbage until backprop is implemented.
    """

    def __init__(self, layers: Sequence[Layer]):
        self.layers = list(layers)

    def forward(self, inputs: Sequence[float]) -> np.ndarray:
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, inputs: Sequence[float]) -> np.ndarray:
        return self.forward(inputs)

    @property
    def is_single_node(self) -> bool:
        return len(self.layers) == 1 and len(self.layers[0].nodes) == 1
