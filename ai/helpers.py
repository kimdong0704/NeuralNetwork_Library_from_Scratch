import functools
from typing import Callable

import numpy as np

from .activations import ACTIVATIONS, Activation
from .config import STEP_THRESHOLD


def zero_weights(input_size: int, output_size: int) -> np.ndarray:
    return np.zeros((output_size, input_size), dtype=float)


def random_weights(input_size: int, output_size: int, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    return np.random.uniform(low, high, size=(output_size, input_size))


def resolve_activation(
    activation: Activation,
    step_threshold: float = STEP_THRESHOLD,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray] | None]:
    if activation is ACTIVATIONS.STEP:
        function = functools.partial(activation.function, threshold=step_threshold)
    else:
        function = activation.function

    return function, activation.derivative
