import math
from dataclasses import dataclass
from typing import Callable


def step_function(x: float, threshold: float = 0.0) -> float:
    return 1.0 if x >= threshold else 0.0


def sigmoid_function(x: float) -> float:
    # guarding overflow error
    if x > 60:
        return 1.0
    if x < -60:
        return 0.0

    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(y: float) -> float:
    return y * (1 - y)


def relu_function(x: float) -> float:
    return x if x > 0 else 0.0


def relu_derivative(y: float) -> float:
    # relu derivative is step function
    return 1.0 if y > 0 else 0.0


@dataclass(frozen=True)
class Activation:
    name: str
    function: Callable[..., float]
    derivative: Callable[[float], float] | None

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
