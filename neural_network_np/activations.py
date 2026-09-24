from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import STEP_THRESHOLD

# exp() overflowing on very negative inputs is expected (sigmoid saturates to 0.0);
# silence the warning process-wide instead of paying for a clip() on every call
np.seterr(over="ignore")

# every derivative takes the layer's output y and the incoming delta and returns delta * dy/dz,
# so element-wise activations and softmax (whose outputs are coupled) share one signature

# STEP Functions
def step_function(x: np.ndarray, threshold: float = STEP_THRESHOLD) -> np.ndarray:
    return np.where(x >= threshold, 1.0, 0.0)

# SIGMOID Functions
def sigmoid_function(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return delta * y * (1 - y)

# RELU Functions
def relu_function(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def relu_derivative(y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    # relu derivative is a step function: the delta passes where the unit was active
    return delta * (y > 0)

# SOFTMAX Functions
# softmax couples every output of a layer, so it is applied across the last axis (one row per sample)
def softmax_function(x: np.ndarray) -> np.ndarray:
    # subtracting the row max guards overflow; softmax is unchanged by shifting all inputs
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

    SOFTMAX = Activation(
        name="softmax",
        function=softmax_function,
        derivative=softmax_derivative
    )

    @classmethod
    def all(cls) -> list[Activation]:
        return [value for value in vars(cls).values() if isinstance(value, Activation)]

    @classmethod
    def by_name(cls, name: str) -> Activation:
        for activation in cls.all():
            if activation.name == name:
                return activation
        raise ValueError(f"unknown activation {name!r}")

    @classmethod
    def by_function(cls, function: Callable[[np.ndarray], np.ndarray]) -> Activation:
        for activation in cls.all():
            if activation.function is function:
                return activation
        raise ValueError(f"{function!r} is not one of the ACTIVATIONS")
