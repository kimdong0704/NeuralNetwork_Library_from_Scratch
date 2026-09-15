from collections.abc import Sequence

import numpy as np

from .config import GRADIENT_DESCENT, STEP_THRESHOLD
from .layer import Layer


class NeuralNetwork:
    def __init__(self):
        self.layers: list[Layer] = []

    def add_layer(
        self,
        input_size: int,
        output_size: int,
        activation_function: str,
        weights: Sequence[Sequence[float]] | None = None,
        bias: Sequence[float] | None = None,
        step_threshold: float = STEP_THRESHOLD,
        gradient_descent: bool = GRADIENT_DESCENT,
    ) -> "NeuralNetwork":
        self.layers.append(
            Layer.build(
                input_size=input_size,
                output_size=output_size,
                activation_function=activation_function,
                weights=weights,
                bias=bias,
                step_threshold=step_threshold,
                gradient_descent=gradient_descent,
            )
        )
        return self

    def forward(self, inputs: Sequence[float]) -> np.ndarray:
        values = np.asarray(inputs, dtype=float)

        for layer in self.layers:
            values = layer.forward(values)

        return values

    def predict(self, inputs: Sequence[float]) -> np.ndarray:
        return self.forward(inputs)

    # decorator for current state of network
    # for and test
    @property
    def is_single_node(self) -> bool:
        return len(self.layers) == 1 and len(self.layers[0].nodes) == 1
