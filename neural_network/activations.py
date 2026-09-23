import math
from dataclasses import dataclass
from typing import Callable

from .config import STEP_THRESHOLD

# STEP Functions
def step_function(x: float, threshold: float = STEP_THRESHOLD) -> float:
    return 1.0 if x >= threshold else 0.0

# SIGMOID Functions
def sigmoid_function(x: float) -> float:
    # guarding overflow error
    if x > 60:
        return 1.0
    if x < -60:
        return 0.0

    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y: float) -> float:
    return y * (1 - y)

# RELU Functions
def relu_function(x: float) -> float:
    return x if x > 0 else 0.0

def relu_derivative(y: float) -> float:
    # relu derivative is step function
    return 1.0 if y > 0 else 0.0

# IDENTITY Functions
def identity_function(x: float) -> float:
    return x

def identity_derivative(y: float) -> float:
    return 1.0

# SOFTMAX Functions
# softmax couples every output of a layer, so it works on the whole layer instead of one node
def softmax_function(x: list[float]) -> list[float]:
    # subtracting the max guards overflow error; softmax is unchanged by shifting all inputs
    shift = max(x)
    exps = []
    total = 0.0
    for value in x:
        exp_value = math.exp(value - shift)
        exps.append(exp_value)
        total += exp_value

    outputs = []
    for exp_value in exps:
        outputs.append(exp_value / total)
    return outputs

def softmax_derivative(y: list[float], delta: list[float]) -> list[float]:
    # full jacobian applied to delta: dz_i = y_i * (delta_i - sum_j delta_j * y_j)
    dot = 0.0
    for output, d in zip(y, delta):
        dot += d * output

    dz = []
    for output, d in zip(y, delta):
        dz.append(output * (d - dot))
    return dz

@dataclass(frozen=True)
class Activation:
    name: str
    function: Callable[..., float | list[float]]
    derivative: Callable[..., float | list[float]] | None

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
