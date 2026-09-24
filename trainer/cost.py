import math
from dataclasses import dataclass
from typing import Callable

# Mean Squared Error Function
# For actual models: Coefficient 0.5 is multiplied to make the calculations simpler
def mse_function(target: float, predicted: float) -> float:
    return (target - predicted) ** 2

def mse_derivative(target: float, predicted: float) -> float:
    return -2 * (target - predicted)

# Cross Entropy Function
# summed over the outputs of a sample by the trainer, giving -sum(t * log(p))
# clipping keeps log(0) and 0/0 out of the math when a prediction reaches exactly 0
CROSS_ENTROPY_MIN_CUT = 1e-12

def cross_entropy_function(predicted: float, target: float) -> float:
    return -target * math.log(max(predicted, CROSS_ENTROPY_MIN_CUT))

def cross_entropy_derivative(predicted: float, target: float) -> float:
    # through the softmax jacobian this becomes (target - predicted)
    return target / max(predicted, CROSS_ENTROPY_MIN_CUT)

# Binary Cross Entropy Function
# each output is its own yes/no probability, so both log(p) and log(1 - p) are clipped
def binary_cross_entropy_function(predicted: float, target: float) -> float:
    predicted = min(max(predicted, CROSS_ENTROPY_MIN_CUT), 1.0 - CROSS_ENTROPY_MIN_CUT)
    return -(target * math.log(predicted) + (1 - target) * math.log(1 - predicted))

def binary_cross_entropy_derivative(predicted: float, target: float) -> float:
    # through the sigmoid derivative p * (1 - p) this becomes (target - predicted)
    predicted = min(max(predicted, CROSS_ENTROPY_MIN_CUT), 1.0 - CROSS_ENTROPY_MIN_CUT)
    return target / predicted - (1 - target) / (1 - predicted)


@dataclass(frozen=True)
class Cost:
    name: str
    function: Callable[[float, float], float]
    derivative: Callable[[float, float], float]

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
