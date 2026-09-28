import math
from dataclasses import dataclass
from typing import Callable

from .config import Matrix

# Every function returns the cost per sample, averaged over the batch (one row per sample).
# Every derivative returns dC/dpredicted for each output of each sample, already divided by the batch size.


# MSE Functions
def mse_function(predicted: Matrix, target: Matrix) -> float:
    total = 0.0
    for predicted_row, target_row in zip(predicted, target):
        for p, t in zip(predicted_row, target_row):
            total += (p - t) ** 2
    return total / len(predicted)

def mse_derivative(predicted: Matrix, target: Matrix) -> Matrix:
    gradients = []
    for predicted_row, target_row in zip(predicted, target):
        row = []
        for p, t in zip(predicted_row, target_row):
            row.append(2 * (p - t) / len(predicted))
        gradients.append(row)
    return gradients

# CROSS ENTROPY Functions
# the floor keeps log(0) and division by 0 out of the math when a probability reaches exactly 0
PROBABILITY_FLOOR = 1e-12

def cross_entropy_function(predicted: Matrix, target: Matrix) -> float:
    total = 0.0
    for predicted_row, target_row in zip(predicted, target):
        for p, t in zip(predicted_row, target_row):
            total -= t * math.log(max(p, PROBABILITY_FLOOR))
    return total / len(predicted)

def cross_entropy_derivative(predicted: Matrix, target: Matrix) -> Matrix:
    # through the softmax jacobian this becomes (predicted - target) / batch size
    gradients = []
    for predicted_row, target_row in zip(predicted, target):
        row = []
        for p, t in zip(predicted_row, target_row):
            row.append(-t / max(p, PROBABILITY_FLOOR) / len(predicted))
        gradients.append(row)
    return gradients


@dataclass(frozen=True)
class Cost:
    name: str
    function: Callable[[Matrix, Matrix], float]
    derivative: Callable[[Matrix, Matrix], Matrix]

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
