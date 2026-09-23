from dataclasses import dataclass
from typing import Callable

# Mean Squared Error Function
# For actual models: Coefficient 0.5 is multiplied to make the calculations simpler
def mse_function(target: float, predicted: float) -> float:
    return (target - predicted) ** 2

def mse_derivative(target: float, predicted: float) -> float:
    return -2 * (target - predicted)


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
