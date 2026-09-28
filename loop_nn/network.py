import json
from pathlib import Path

from .activations import ACTIVATIONS
from .config import Matrix
from .layer import Layer


class Network:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, inputs: Matrix) -> Matrix:
        outputs = []
        for row in inputs:
            outputs.append([float(value) for value in row])

        for layer in self.layers:
            outputs = layer.forward(outputs)

        return outputs

    def backward(self, deltas: Matrix) -> Matrix:
        for layer in reversed(self.layers):
            deltas = layer.backward(deltas)

        return deltas

    def step(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.step(learning_rate)

    def save(self, path: str | Path) -> None:
        """Saves every layer's weights, bias and activation name to one .json file."""
        layers = []
        for layer in self.layers:
            layers.append({"weights": layer.weights, "bias": layer.bias, "activation": layer.activation.name})

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"layers": layers}))

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        """Rebuilds a network written by save()."""
        layers = []
        for saved in json.loads(Path(path).read_text())["layers"]:
            layers.append(Layer.build(
                input_size=len(saved["weights"][0]),
                output_size=len(saved["weights"]),
                weight=saved["weights"],
                bias=saved["bias"],
                activation=ACTIVATIONS.by_name(saved["activation"])
            ))

        return cls(*layers)
