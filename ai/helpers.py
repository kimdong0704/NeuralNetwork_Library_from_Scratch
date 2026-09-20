import functools
import random
from typing import Callable

from .activations import ACTIVATIONS, Activation
from .config import STEP_THRESHOLD


def zero_weights(input_size: int, output_size: int) -> list[list[float]]:
    weights = []
    for _ in range(output_size):
        weights.append([0.0] * input_size)
    return weights


def random_weights(input_size: int, output_size: int, low: float = -1.0, high: float = 1.0) -> list[list[float]]:
    weights = []
    for _ in range(output_size):
        row = []
        for _ in range(input_size):
            row.append(random.uniform(low, high))
        weights.append(row)
    return weights


def resolve_activation(
    activation: Activation,
    step_threshold: float = STEP_THRESHOLD,
) -> tuple[Callable[[float], float], Callable[[float], float] | None]:
    if activation is ACTIVATIONS.STEP:
        function = functools.partial(activation.function, threshold=step_threshold)
    else:
        function = activation.function

    return function, activation.derivative
