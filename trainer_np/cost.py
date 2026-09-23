from dataclasses import dataclass
from typing import Callable

import numpy as np


def mse_function(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((predicted - target) ** 2))

def mse_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 2 * (target - predicted) / predicted.shape[0]

# CROSS ENTROPY Functions
# clipping keeps log(0) and 0/0 out of the math when a prediction reaches exactly 0
CROSS_ENTROPY_MIN_CUT = 1e-12

def cross_entropy_function(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.clip(predicted, CROSS_ENTROPY_MIN_CUT, 1.0)
    return float(-np.sum(target * np.log(predicted)) / predicted.shape[0])

def cross_entropy_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    # through the softmax jacobian this becomes (target - predicted) / n
    predicted = np.clip(predicted, CROSS_ENTROPY_MIN_CUT, 1.0)
    return (target / predicted) / predicted.shape[0]


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
