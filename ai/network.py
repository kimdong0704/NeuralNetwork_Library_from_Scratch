import numpy as np

from .layer import Layer


class Network:
    def __init__(self, *layers: Layer):
        self.layers: list[Layer] = list(layers)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        values = inputs

        for layer in self.layers:
            values = layer.forward(values)

        return values
