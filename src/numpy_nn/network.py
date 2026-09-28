from pathlib import Path

import numpy as np

from .activations import ACTIVATIONS
from .config import DTYPE
from .layer import Layer


class Network:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = np.asarray(inputs, dtype=DTYPE)
        for layer in self.layers:
            outputs = layer.forward(outputs)

        return outputs

    def backward(self, deltas: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            deltas = layer.backward(deltas)

        return deltas

    def step(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.step(learning_rate)

    def save(self, path: str | Path) -> None:
        """Saves every layer's weights, bias and activation name to one .npz file."""
        arrays = {}
        for index, layer in enumerate(self.layers):
            arrays[f"weights_{index}"] = layer.weights
            arrays[f"bias_{index}"] = layer.bias
            arrays[f"activation_{index}"] = np.array(layer.activation.name)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        """Rebuilds a network written by save()."""
        layers = []
        with np.load(path) as arrays:
            layer_count = sum(1 for key in arrays.files if key.startswith("weights_"))
            for index in range(layer_count):
                weights = arrays[f"weights_{index}"]
                output_size, input_size = weights.shape
                layers.append(Layer.build(
                    input_size=input_size,
                    output_size=output_size,
                    weight=weights,
                    bias=arrays[f"bias_{index}"],
                    activation=ACTIVATIONS.by_name(str(arrays[f"activation_{index}"]))
                ))

        return cls(*layers)
