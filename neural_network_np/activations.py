from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import STEP_THRESHOLD

# exp() overflowing on very negative inputs is expected (sigmoid saturates to 0.0);
# silence the warning process-wide instead of paying for a clip() on every call
np.seterr(over="ignore")

# STEP Functions
def step_function(x: np.ndarray, threshold: float = STEP_THRESHOLD) -> np.ndarray:
    return np.where(x >= threshold, 1.0, 0.0)

# SIGMOID Functions
def sigmoid_function(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y: np.ndarray) -> np.ndarray:
    return y * (1 - y)

# RELU Functions
def relu_function(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def relu_derivative(y: np.ndarray) -> np.ndarray:
    # relu derivative is step function
    return np.where(y > 0, 1.0, 0.0)

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
    function: Callable[..., np.ndarray]
    derivative: Callable[..., np.ndarray] | None

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
