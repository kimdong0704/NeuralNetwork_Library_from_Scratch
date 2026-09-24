from pathlib import Path

import numpy as np

from .activations import ACTIVATIONS
from .config import DTYPE
from .layer import Layer


class Network:
    def __init__(self, *layers: Layer):
        self.layers: list[Layer] = list(layers)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=DTYPE)

        for layer in self.layers:
            values = layer.forward(values)

        return values

    def backward(self, gradients: np.ndarray, learning_rate: float) -> None:
        deltas = gradients

        for layer in reversed(self.layers[1:]):
            deltas = layer.backward(deltas, learning_rate)

        # the first layer's deltas would only flow into the inputs, so they are never computed
        self.layers[0].update(deltas, learning_rate)

    def save(self, path: str | Path) -> None:
        """Saves every layer's weights, bias and activation name to one .npz file."""
        arrays = {}
        for index, layer in enumerate(self.layers):
            arrays[f"weights_{index}"] = layer.weights
            arrays[f"bias_{index}"] = layer.bias
            arrays[f"activation_{index}"] = np.array(ACTIVATIONS.by_function(layer.activation_function).name)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        """Rebuilds a network saved with save()."""
        layers = []
        with np.load(path) as arrays:
            layer_count = sum(1 for key in arrays.files if key.startswith("weights_"))
            for index in range(layer_count):
                weights = arrays[f"weights_{index}"]
                activation = ACTIVATIONS.by_name(str(arrays[f"activation_{index}"]))
                output_size, input_size = weights.shape
                layers.append(Layer.build(
                    input_size=input_size,
                    output_size=output_size,
                    weight=weights,
                    bias=arrays[f"bias_{index}"],
                    activation_function=activation.function,
                    activation_derivative=activation.derivative
                ))

        return cls(*layers)
