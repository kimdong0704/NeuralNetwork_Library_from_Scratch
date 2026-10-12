import math
from dataclasses import dataclass
from typing import Callable

from .config import STEP_THRESHOLD, Vector

# Every function maps the weighted sums z of one sample to the layer's outputs y.
# Every derivative takes the outputs y and the incoming deltas dC/dy and returns dC/dz, so
# element-wise activations and softmax (whose outputs are coupled) share one signature.


# STEP Functions
def step_function(x: Vector, threshold: float = STEP_THRESHOLD) -> Vector:
    outputs = []
    for value in x:
        outputs.append(1.0 if value >= threshold else 0.0)
    return outputs

# SIGMOID Functions
def sigmoid_function(x: Vector) -> Vector:
    # equal to 1 / (1 + e^-x), but the tanh form cannot overflow for large negative x
    outputs = []
    for value in x:
        outputs.append(0.5 * (1.0 + math.tanh(0.5 * value)))
    return outputs

def sigmoid_derivative(y: Vector, delta: Vector) -> Vector:
    dz = []
    for output, d in zip(y, delta):
        dz.append(d * output * (1 - output))
    return dz

# RELU Functions
def relu_function(x: Vector) -> Vector:
    outputs = []
    for value in x:
        outputs.append(value if value > 0 else 0.0)
    return outputs

def relu_derivative(y: Vector, delta: Vector) -> Vector:
    # the delta only passes through units that were active
    dz = []
    for output, d in zip(y, delta):
        dz.append(d if output > 0 else 0.0)
    return dz

# IDENTITY Functions
def identity_function(x: Vector) -> Vector:
    return list(x)

def identity_derivative(y: Vector, delta: Vector) -> Vector:
    return list(delta)

# SOFTMAX Functions
def softmax_function(x: Vector) -> Vector:
    # subtracting the max guards overflow; softmax is unchanged by shifting every input
    shift = max(x)
    exps = []
    for value in x:
        exps.append(math.exp(value - shift))

    total = sum(exps)
    outputs = []
    for exp_value in exps:
        outputs.append(exp_value / total)
    return outputs

def softmax_derivative(y: Vector, delta: Vector) -> Vector:
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
    function: Callable[[Vector], Vector]
    derivative: Callable[[Vector, Vector], Vector] | None

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

    IDENTITY = Activation(
        name="identity",
        function=identity_function,
        derivative=identity_derivative
    )

    SOFTMAX = Activation(
        name="softmax",
        function=softmax_function,
        derivative=softmax_derivative
    )

    @classmethod
    def by_name(cls, name: str) -> Activation:
        for value in vars(cls).values():
            if isinstance(value, Activation) and value.name == name:
                return value
        raise ValueError(f"unknown activation {name!r}")
