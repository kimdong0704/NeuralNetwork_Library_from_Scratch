from dataclasses import dataclass
from typing import Callable

import numpy as np


def mse_function(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((predicted - target) ** 2))

def mse_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 2 * (target - predicted) / predicted.shape[0]

# CROSS ENTROPY Functions
# clipping keeps log(0) and 0/0 out of the math when a probability reaches exactly 0
CROSS_ENTROPY_MIN_CUT = 1e-12

def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.maximum(probabilities, CROSS_ENTROPY_MIN_CUT)

def cross_entropy_function(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(-np.sum(target * np.log(_clip(predicted))) / predicted.shape[0])

def cross_entropy_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    # through the softmax jacobian this becomes (target - predicted) / n
    return target / _clip(predicted) / predicted.shape[0]

# BINARY CROSS ENTROPY Functions
# each output is its own yes/no probability, so both p and 1 - p are clipped
# (clipping 1 - p directly also works in float32, where 1 - 1e-12 rounds to 1.0)
def binary_cross_entropy_function(predicted: np.ndarray, target: np.ndarray) -> float:
    losses = target * np.log(_clip(predicted)) + (1 - target) * np.log(_clip(1 - predicted))
    return float(-np.sum(losses) / predicted.shape[0])

def binary_cross_entropy_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    # through the sigmoid derivative p * (1 - p) this becomes (target - predicted) / n
    return (target / _clip(predicted) - (1 - target) / _clip(1 - predicted)) / predicted.shape[0]


@dataclass(frozen=True)
class Cost:
    name: str
    function: Callable[[np.ndarray, np.ndarray], float]
    derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]

class COSTS:
    MSE = Cost(
        name="mse",
        function=mse_function,
        derivative=mse_derivative
    )

    CROSS_ENTROPY = Cost(
        name="cross_entropy",
        function=cross_entropy_function,
        derivative=cross_entropy_derivative
    )

    BINARY_CROSS_ENTROPY = Cost(
        name="binary_cross_entropy",
        function=binary_cross_entropy_function,
        derivative=binary_cross_entropy_derivative
    )
