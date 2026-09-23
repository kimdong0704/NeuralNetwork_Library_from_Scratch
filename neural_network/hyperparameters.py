import random
from typing import Callable, TypeVar

T = TypeVar("T")


def _repeat(count: int, value_fn: Callable[[], T]) -> list[T]:
    values = []
    for _ in range(count):
        values.append(value_fn())
    return values


# Initialize Weights Functions
def zero_weights(input_size: int, output_size: int) -> list[list[float]]:
    return _repeat(output_size, lambda: [0.0] * input_size)

def random_weights(input_size: int, output_size: int, low: float = -1.0, high: float = 1.0) -> list[list[float]]:
    return _repeat(output_size, lambda: _repeat(input_size, lambda: random.uniform(low, high)))

# Initialize Biases Functions
def zero_bias(output_size: int) -> list[float]:
    return _repeat(output_size, lambda: 0.0)

def random_bias(output_size: int, low: float = -1.0, high: float = 1.0) -> list[float]:
    return _repeat(output_size, lambda: random.uniform(low, high))
