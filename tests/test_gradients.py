"""Backpropagation is checked against finite differences of the cost."""
import numpy as np
import pytest

from numpy_nn import ACTIVATIONS, COSTS, Layer, Network

EPSILON = 1e-6


def numerical_gradient(network, cost, inputs, targets, parameter):
    gradient = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        original = parameter[index]
        parameter[index] = original + EPSILON
        higher = cost.function(network.forward(inputs), targets)
        parameter[index] = original - EPSILON
        lower = cost.function(network.forward(inputs), targets)
        parameter[index] = original
        gradient[index] = (higher - lower) / (2 * EPSILON)
    return gradient


@pytest.mark.parametrize(
    "hidden, output, cost",
    [
        (ACTIVATIONS.SIGMOID, ACTIVATIONS.SIGMOID, COSTS.MSE),
        (ACTIVATIONS.SIGMOID, ACTIVATIONS.IDENTITY, COSTS.MSE),
        (ACTIVATIONS.RELU, ACTIVATIONS.SOFTMAX, COSTS.CROSS_ENTROPY),
    ],
    ids=["sigmoid-mse", "identity-mse", "softmax-cross-entropy"],
)
def test_backpropagation_matches_finite_differences(hidden, output, cost):
    rng = np.random.default_rng(0)
    network = Network(
        Layer.build(3, 5, rng.uniform(-1, 1, (5, 3)), rng.uniform(-1, 1, 5), hidden),
        Layer.build(5, 4, rng.uniform(-1, 1, (4, 5)), rng.uniform(-1, 1, 4), output),
    )
    inputs = rng.normal(size=(6, 3))
    targets = np.eye(4)[rng.integers(4, size=6)]

    network.backward(cost.derivative(network.forward(inputs), targets))

    for layer in network.layers:
        np.testing.assert_allclose(
            layer.weight_gradient, numerical_gradient(network, cost, inputs, targets, layer.weights), atol=1e-6
        )
        np.testing.assert_allclose(
            layer.bias_gradient, numerical_gradient(network, cost, inputs, targets, layer.bias), atol=1e-6
        )


def test_softmax_rows_are_probabilities():
    outputs = ACTIVATIONS.SOFTMAX.function(np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]]))
    assert np.all(np.isfinite(outputs))
    np.testing.assert_allclose(outputs.sum(axis=1), 1.0)


def test_sigmoid_does_not_overflow():
    outputs = ACTIVATIONS.SIGMOID.function(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(outputs, [0.0, 0.5, 1.0])
