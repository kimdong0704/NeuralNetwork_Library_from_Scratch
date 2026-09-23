from dataclasses import dataclass
from typing import Callable

import numpy as np


def mse_function(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((predicted - target) ** 2))

def mse_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 2 * (target - predicted) / predicted.shape[0]


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
