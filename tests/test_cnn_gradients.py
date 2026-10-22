"""The cnn layers' and LeNet-5's backpropagation is checked against finite differences of the cost."""
import numpy as np
import pytest

from cnn import Conv2D, Flatten, MaxPool2D, Pad
from lenet5 import LeNet5
from lenet5.model_config import IMAGE_SIZE, INPUT_CHANNELS, OUTPUT_UNITS
from numpy_nn import Layer, Module, Network
from numpy_utils import ACTIVATIONS, COSTS

EPSILON = 1e-6
# entries of each LeNet-5 parameter checked numerically (checking all ~60k would be too slow)
SAMPLED_ENTRIES = 5
LENET5_WEIGHT_SCALE = 3


def numerical_derivative(network, cost, inputs, targets, array, index):
    original = array[index]
    array[index] = original + EPSILON
    higher = cost.function(network.forward(inputs), targets)
    array[index] = original - EPSILON
    lower = cost.function(network.forward(inputs), targets)
    array[index] = original
    return (higher - lower) / (2 * EPSILON)


def numerical_gradient(network, cost, inputs, targets, array):
    gradient = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        gradient[index] = numerical_derivative(network, cost, inputs, targets, array, index)
    return gradient


def small_cnn(rng, pool_stride) -> Network[Module]:
    # pad -> conv -> pool -> conv (stride 2) -> flatten -> dense softmax, on 2-channel 6x6 images
    return Network(
        Pad(amount=1),
        Conv2D(rng.uniform(-1, 1, (3, 2, 3, 3)), rng.uniform(-1, 1, (3, 1, 1)), ACTIVATIONS.RELU),
        MaxPool2D(pool_rows=2, pool_cols=2, stride=pool_stride),
        Conv2D(rng.uniform(-1, 1, (4, 3, 2, 2)), rng.uniform(-1, 1, (4, 1, 1)), ACTIVATIONS.SIGMOID, stride=2),
        Flatten(),
    )


# stride 1 makes the pooling windows overlap, so one pixel can be the max of several windows
@pytest.mark.parametrize("pool_stride", [1, 2], ids=["overlapping-pool", "pool"])
def test_cnn_backpropagation_matches_finite_differences(pool_stride):
    rng = np.random.default_rng(0)
    network = small_cnn(rng, pool_stride)
    inputs = rng.normal(size=(3, 2, 6, 6))
    flat_size = network.forward(inputs).shape[1]
    network.layers.append(Layer.build(flat_size, 5, rng.uniform(-1, 1, (5, flat_size)), rng.uniform(-1, 1, 5), ACTIVATIONS.SOFTMAX))
    targets = np.eye(5)[rng.integers(5, size=3)]
    cost = COSTS.CROSS_ENTROPY

    input_gradient = network.backward(cost.derivative(network.forward(inputs), targets))

    for layer in network.layers:
        if isinstance(layer, (Layer, Conv2D)):
            np.testing.assert_allclose(
                layer.weight_gradient, numerical_gradient(network, cost, inputs, targets, layer.weights), atol=1e-6
            )
            np.testing.assert_allclose(
                layer.bias_gradient, numerical_gradient(network, cost, inputs, targets, layer.bias), atol=1e-6
            )
    np.testing.assert_allclose(input_gradient, numerical_gradient(network, cost, inputs, targets, inputs), atol=1e-6)


def test_lenet5_backpropagation_matches_finite_differences():
    rng = np.random.default_rng(0)
    # the default init gives gradients near 1e-3 and x10 saturates the softmax (gradients ~1e-32);
    # x3 keeps them around 1e-2 to 1e-1 so the comparison means something
    net = LeNet5(seed=0)
    for layer in net.layers:
        if hasattr(layer, "weights"):
            layer.weights *= LENET5_WEIGHT_SCALE
            layer.bias += rng.uniform(-0.1, 0.1, layer.bias.shape)
    inputs = rng.random((2, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    targets = np.eye(OUTPUT_UNITS)[[3, 7]]
    cost = COSTS.CROSS_ENTROPY

    net.backward(cost.derivative(net.forward(inputs), targets))

    for layer in net.layers:
        if not hasattr(layer, "weights"):
            continue
        for array, gradient in [(layer.weights, layer.weight_gradient), (layer.bias, layer.bias_gradient)]:
            for flat in rng.choice(array.size, size=min(SAMPLED_ENTRIES, array.size), replace=False):
                index = np.unravel_index(flat, array.shape)
                expected = numerical_derivative(net, cost, inputs, targets, array, index)
                assert gradient[index] == pytest.approx(expected, rel=1e-4, abs=1e-9)


def test_lenet5_step_lowers_the_cost():
    rng = np.random.default_rng(0)
    net = LeNet5(seed=0)
    inputs = rng.random((8, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    targets = np.eye(OUTPUT_UNITS)[rng.integers(OUTPUT_UNITS, size=8)]
    cost = COSTS.CROSS_ENTROPY

    before = cost.function(net.forward(inputs), targets)
    for _ in range(5):
        net.backward(cost.derivative(net.forward(inputs), targets))
        net.step(learning_rate=0.1)
    after = cost.function(net.forward(inputs), targets)

    assert after < before


def test_lenet5_step_updates_the_named_parameters():
    net = LeNet5(seed=0)
    inputs = np.random.default_rng(0).random((2, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    targets = np.eye(OUTPUT_UNITS)[[0, 1]]
    c1_before = net.c1_weights.copy()

    net.backward(COSTS.CROSS_ENTROPY.derivative(net.forward(inputs), targets))
    net.step(learning_rate=0.1)

    assert net.c1_weights is net.c1_layer.weights
    assert not np.array_equal(net.c1_weights, c1_before)
