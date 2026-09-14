import math

# Registry of activation functions, each paired with its derivative.
# A derivative of None means the activation has no usable derivative for
# gradient descent (e.g. the step function).
#
# "function" is called as function(x) for every activation except "step",
# which additionally takes a threshold (bound per-Node via functools.partial,
# since the threshold is a per-node setting, not a global one).
#
# "derivative" (when present) is called as derivative(y), where y is the
# activation's *output* (matches how sigmoid_derivative is normally expressed).


def step_function(x: float, threshold: float = 0.0) -> int:
    if x >= threshold:
        return 1
    else:
        return 0


def sigmoid_function(x: float) -> float:
    # Guard against OverflowError from math.exp on very large |x|.
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0

    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(y: float) -> float:
    # y here is expected to be a sigmoid *output* (i.e. already in (0, 1)),
    # so this is sigmoid'(net) expressed via the output, per prediction.
    return y * (1 - y)


ACTIVATIONS = {
    "step": {"function": step_function, "derivative": None},
    "sigmoid": {"function": sigmoid_function, "derivative": sigmoid_derivative},
}
