from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import STEP_THRESHOLD

# Every function maps a batch of weighted sums z (one row per sample) to the layer's outputs y.
# Every derivative takes the outputs y and the incoming deltas dC/dy and returns dC/dz, so
# element-wise activations and softmax (whose outputs are coupled) share one signature.


# STEP Functions
def step_function(x: np.ndarray, threshold: float = STEP_THRESHOLD) -> np.ndarray:
    return np.where(x >= threshold, 1.0, 0.0)

# SIGMOID Functions
def sigmoid_function(x: np.ndarray) -> np.ndarray:
    # equal to 1 / (1 + e^-x), but the tanh form cannot overflow for large negative x
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def sigmoid_derivative(y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return delta * y * (1 - y)

# RELU Functions
def relu_function(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def relu_derivative(y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    # the delta only passes through units that were active
    return delta * (y > 0)

# IDENTITY Functions
def identity_function(x: np.ndarray) -> np.ndarray:
    return x

def identity_derivative(y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return delta

# SOFTMAX Functions
def softmax_function(x: np.ndarray) -> np.ndarray:
    # subtracting the row max guards overflow; softmax is unchanged by shifting a whole row
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def softmax_derivative(y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    # full jacobian applied to delta: dz_i = y_i * (delta_i - sum_j delta_j * y_j)
    return y * (delta - np.sum(delta * y, axis=-1, keepdims=True))


@dataclass(frozen=True)
class Activation:
    name: str
    function: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray, np.ndarray], np.ndarray] | None

class ACTIVATIONS:
    STEP = Activation(
        name="step",
        function=step_function,
        derivative=None
    )

    SIGMOID = Activation(
        name="sigmoid",
        function=sigmoid_function,
        derivative=sigmoid_derivative
    )

    RELU = Activation(
        name="relu",
        function=relu_function,
        derivative=relu_derivative
    )

    IDENTITY = Activation(
        name="identity",
        function=identity_function,
        derivative=identity_derivative
    )

    SOFTMAX = Activation(
        name="softmax",
        function=softmax_function,
        derivative=softmax_derivative
    )

    @classmethod
    def by_name(cls, name: str) -> Activation:
        for value in vars(cls).values():
            if isinstance(value, Activation) and value.name == name:
                return value
        raise ValueError(f"unknown activation {name!r}")
