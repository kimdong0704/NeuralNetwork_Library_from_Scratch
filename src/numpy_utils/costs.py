from dataclasses import dataclass
from typing import Callable

import numpy as np

# Every function returns the cost per sample, averaged over the batch (one row per sample).
# Every derivative returns dC/dpredicted for each output of each sample, already divided by the batch size.


# MSE Functions
def mse_function(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.sum((predicted - target) ** 2) / predicted.shape[0])

def mse_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 2 * (predicted - target) / predicted.shape[0]

# CROSS ENTROPY Functions
# the floor keeps log(0) and division by 0 out of the math when a probability reaches exactly 0
PROBABILITY_FLOOR = 1e-12

def cross_entropy_function(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(-np.sum(target * np.log(np.maximum(predicted, PROBABILITY_FLOOR))) / predicted.shape[0])

def cross_entropy_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    # through the softmax jacobian this becomes (predicted - target) / batch size
    return -target / np.maximum(predicted, PROBABILITY_FLOOR) / predicted.shape[0]


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
