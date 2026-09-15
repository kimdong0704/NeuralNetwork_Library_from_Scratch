import math


def step_function(x: float, threshold: float = 0.0) -> int:
    if x >= threshold:
        return 1
    else:
        return 0


def sigmoid_function(x: float) -> float:
    # guarding overflow error
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0

    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(y: float) -> float:
    return y * (1 - y)


ACTIVATIONS = {
    "step": {"function": step_function, "derivative": None},
    "sigmoid": {"function": sigmoid_function, "derivative": sigmoid_derivative},
}
