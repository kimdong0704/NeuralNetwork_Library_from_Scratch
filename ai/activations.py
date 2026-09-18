from dataclasses import dataclass
from typing import Callable

import numpy as np


def step_function(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.where(x >= threshold, 1.0, 0.0)


def sigmoid_function(x: np.ndarray) -> np.ndarray:
    # guarding overflow error
    guard_overflow= np.clip(x, -60, 60)
    sigmoid_values = 1 / (1 + np.exp(-guard_overflow))

    result = np.where(x > 60, 1.0, sigmoid_values)
    result = np.where(x < -60, 0.0, result)

    return result


def sigmoid_derivative(y: np.ndarray) -> np.ndarray:
    return y * (1 - y)


@dataclass(frozen=True)
class Activation:
    name: str
    function: Callable[..., np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray] | None

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
