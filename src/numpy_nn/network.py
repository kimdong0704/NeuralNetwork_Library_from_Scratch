from pathlib import Path
from typing import Generic, Protocol, TypeVar

import numpy as np

from numpy_utils.activations import ACTIVATIONS
from numpy_utils.config import DTYPE
from .layer import Layer


class Module(Protocol):
    """Anything Network can chain: dense Layer, Conv2D, MaxPool2D, Pad, Flatten."""

    def forward(self, inputs: np.ndarray) -> np.ndarray: ...

    def backward(self, deltas: np.ndarray) -> np.ndarray: ...

    def step(self, learning_rate: float) -> None: ...


L = TypeVar("L", bound=Module)


class Network(Generic[L]):
    """Network(*dense_layers) is a Network[Layer]; mixing in cnn layers gives a Network of their union."""

    def __init__(self, *layers: L):
        self.layers: list[L] = list(layers)

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
            if not isinstance(layer, Layer):
                raise TypeError(f"save() only supports dense Layers, got {type(layer).__name__} at index {index}")
            arrays[f"weights_{index}"] = layer.weights
            arrays[f"bias_{index}"] = layer.bias
            arrays[f"activation_{index}"] = np.array(layer.activation.name)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    @staticmethod
    def load(path: str | Path) -> "Network[Layer]":
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

        return Network(*layers)
