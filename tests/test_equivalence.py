"""loop_nn/loop_trainer and numpy_nn/numpy_trainer implement the same math, so from the same weights they must agree."""
import numpy as np
import pytest

import loop_nn
import loop_trainer
import numpy_nn
import numpy_trainer

TOLERANCE = 1e-9

# (layer sizes, hidden activation, output activation, cost)
ARCHITECTURES = {
    "sigmoid-mse": ([2, 4, 1], "sigmoid", "sigmoid", "MSE"),
    "relu-softmax-cross-entropy": ([3, 5, 4], "relu", "softmax", "CROSS_ENTROPY"),
    "deep-identity-mse": ([2, 3, 3, 2], "sigmoid", "identity", "MSE"),
    "step-perceptron": ([2, 1], None, "step", "MSE"),
}


def build_pair(sizes, hidden, output, seed=0):
    """The same randomly initialized network, once per implementation."""
    rng = np.random.default_rng(seed)
    numpy_layers, loop_layers = [], []

    for index, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
        name = output if index == len(sizes) - 2 else hidden
        weights = rng.uniform(-1, 1, size=(outputs, inputs))
        bias = rng.uniform(-1, 1, size=outputs)

        numpy_layers.append(numpy_nn.Layer.build(inputs, outputs, weights, bias, numpy_nn.ACTIVATIONS.by_name(name)))
        loop_layers.append(loop_nn.Layer.build(inputs, outputs, weights.tolist(), bias.tolist(), loop_nn.ACTIVATIONS.by_name(name)))

    return numpy_nn.Network(*numpy_layers), loop_nn.Network(*loop_layers)


def make_data(sizes, output, samples=12, seed=1):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(samples, sizes[0]))
    if output == "softmax":
        targets = np.eye(sizes[-1])[rng.integers(sizes[-1], size=samples)]
    else:
        targets = rng.integers(0, 2, size=(samples, sizes[-1])).astype(float)
    return inputs, targets


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_forward_and_backward_match(name):
    sizes, hidden, output, cost_name = ARCHITECTURES[name]
    numpy_network, loop_network = build_pair(sizes, hidden, output)
    inputs, targets = make_data(sizes, output)
    numpy_cost, loop_cost = getattr(numpy_nn.COSTS, cost_name), getattr(loop_nn.COSTS, cost_name)

    numpy_predicted = numpy_network.forward(inputs)
    loop_predicted = loop_network.forward(inputs.tolist())
    np.testing.assert_allclose(loop_predicted, numpy_predicted, atol=TOLERANCE)
    assert loop_cost.function(loop_predicted, targets.tolist()) == pytest.approx(
        numpy_cost.function(numpy_predicted, targets), abs=TOLERANCE
    )

    numpy_propagated = numpy_network.backward(numpy_cost.derivative(numpy_predicted, targets))
    loop_propagated = loop_network.backward(loop_cost.derivative(loop_predicted, targets.tolist()))
    np.testing.assert_allclose(loop_propagated, numpy_propagated, atol=TOLERANCE)

    for numpy_layer, loop_layer in zip(numpy_network.layers, loop_network.layers):
        loop_weight_gradient = [node.weight_gradient for node in loop_layer.nodes]
        loop_bias_gradient = [node.bias_gradient for node in loop_layer.nodes]
        np.testing.assert_allclose(loop_weight_gradient, numpy_layer.weight_gradient, atol=TOLERANCE)
        np.testing.assert_allclose(loop_bias_gradient, numpy_layer.bias_gradient, atol=TOLERANCE)


@pytest.mark.parametrize("name", ARCHITECTURES)
@pytest.mark.parametrize("batch_size", [1, 5])
def test_training_matches(name, batch_size):
    sizes, hidden, output, cost_name = ARCHITECTURES[name]
    numpy_network, loop_network = build_pair(sizes, hidden, output)
    inputs, targets = make_data(sizes, output)

    numpy_run = numpy_trainer.Trainer(numpy_network, getattr(numpy_nn.COSTS, cost_name), learning_rate=0.1)
    loop_run = loop_trainer.Trainer(loop_network, getattr(loop_nn.COSTS, cost_name), learning_rate=0.1)
    validation = (inputs[:4], targets[:4])

    numpy_history = numpy_run.train(
        numpy_trainer.DataLoader(inputs, targets, batch_size, shuffle=False), epochs=25, validation=validation
    )
    loop_history = loop_run.train(
        loop_trainer.DataLoader(inputs.tolist(), targets.tolist(), batch_size, shuffle=False), epochs=25,
        validation=(validation[0].tolist(), validation[1].tolist())
    )

    for numpy_result, loop_result in zip(numpy_history, loop_history, strict=True):
        assert loop_result.loss == pytest.approx(numpy_result.loss, abs=TOLERANCE)
        assert loop_result.accuracy == numpy_result.accuracy
        assert loop_result.validation_loss == pytest.approx(numpy_result.validation_loss, abs=TOLERANCE)
        assert loop_result.validation_accuracy == numpy_result.validation_accuracy

    for numpy_layer, loop_layer in zip(numpy_network.layers, loop_network.layers):
        np.testing.assert_allclose(loop_layer.weights, numpy_layer.weights, atol=TOLERANCE)
        np.testing.assert_allclose(loop_layer.bias, numpy_layer.bias, atol=TOLERANCE)


def test_stop_when_ends_training_at_the_same_epoch():
    sizes, hidden, output, cost_name = ARCHITECTURES["sigmoid-mse"]
    numpy_network, loop_network = build_pair(sizes, hidden, output)
    inputs, targets = make_data(sizes, output)
    stop = lambda result: result.epoch == 7

    numpy_history = numpy_trainer.Trainer(numpy_network, numpy_nn.COSTS.MSE, 0.5).train(
        numpy_trainer.DataLoader(inputs, targets, 4, shuffle=False), epochs=50, stop_when=stop
    )
    loop_history = loop_trainer.Trainer(loop_network, loop_nn.COSTS.MSE, 0.5).train(
        loop_trainer.DataLoader(inputs.tolist(), targets.tolist(), 4, shuffle=False), epochs=50, stop_when=stop
    )

    assert len(numpy_history) == len(loop_history) == 7
