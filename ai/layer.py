from collections.abc import Sequence
from typing import Callable

import functools
import numpy as np

from .activations import ACTIVATIONS
from .config import BIAS, GRADIENT_DESCENT, STEP_THRESHOLD, WEIGHTS
from .node import Node


class Layer:
    def __init__(self, nodes: Sequence[Node]):
        self.nodes = list(nodes)

    @staticmethod
    def zero_weights(input_size: int) -> np.ndarray:
        return np.zeros(input_size, dtype=float)

    @staticmethod
    def random_weights(input_size: int, low: float = -1.0, high: float = 1.0) -> np.ndarray:
        return np.random.uniform(low, high, size=input_size)

    @staticmethod
    def _resolve_activation(
        activation_function: str,
        step_threshold: float = STEP_THRESHOLD,
    ) -> tuple[Callable[[float], float], Callable[[float], float] | None]:
        if activation_function not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation_function: {activation_function!r}. "
                f"Expected one of {list(ACTIVATIONS)}."
            )

        entry = ACTIVATIONS[activation_function]
        derivative = entry["derivative"]

        if activation_function == "step":
            function = functools.partial(entry["function"], threshold=step_threshold)
        else:
            function = entry["function"]

        return function, derivative

    @classmethod
    def _build_node(
        cls,
        input_size: int,
        activation_function: str,
        weights: Sequence[float] | None = WEIGHTS,
        bias: float = BIAS,
        step_threshold: float = STEP_THRESHOLD,
        gradient_descent: bool = GRADIENT_DESCENT,
    ) -> Node:
        resolved_weights: np.ndarray
        if weights is None:
            resolved_weights = cls.zero_weights(input_size)
        elif weights == "random":
            resolved_weights = cls.random_weights(input_size)
        else:
            resolved_weights = np.array(weights, dtype=float)

        # shape check
        if resolved_weights.shape[0] != input_size:
            raise ValueError(
                f"weights has length {resolved_weights.shape[0]}, expected {input_size}"
            )

        activation, derivative = cls._resolve_activation(activation_function, step_threshold)

        # derivative exists check
        if gradient_descent and derivative is None:
            raise ValueError(
                "gradient_descent=True requires an activation function with a "
                f"usable derivative ({activation_function!r} has none)."
            )

        return Node(
            input_size=input_size,
            weights=resolved_weights,
            bias=bias,
            activation_function=activation,
            activation_derivative=derivative,
            gradient_descent=gradient_descent,
        )

    # call build first before any instance exists
    @classmethod
    def build(
        cls,
        input_size: int,
        output_size: int,
        activation_function: str,
        weights: Sequence[Sequence[float]] | None = None,
        bias: Sequence[float] | None = None,
        step_threshold: float = STEP_THRESHOLD,
        gradient_descent: bool = GRADIENT_DESCENT,
    ) -> "Layer":
        nodes = [
            cls._build_node(
                input_size=input_size,
                activation_function=activation_function,
                weights=weights[i] if weights is not None else None,
                bias=bias[i] if bias is not None else BIAS,
                step_threshold=step_threshold,
                gradient_descent=gradient_descent,
            )
            for i in range(output_size)
        ]
        return cls(nodes)

    def forward(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=float)

        # .array --> scalar to 1-D array
        # .stack --> 1-D array to High-D array
        node_weights = []
        for node in self.nodes:
            node_weights.append(node.weights)
        weights_matrix = np.stack(node_weights)

        node_biases = []
        for node in self.nodes:
            node_biases.append(node.bias)
        bias_vector = np.array(node_biases)

        # Performance/vectorization tradeoff
        # can use node.sum and .predict, but runs on python loop
        # Numpy --> runs on C; node per loop --> runs on Python
        totals = weights_matrix @ values + bias_vector

        outputs = []
        for node, total in zip(self.nodes, totals):
            outputs.append(node.activation_function(total))

        return np.array(outputs)
