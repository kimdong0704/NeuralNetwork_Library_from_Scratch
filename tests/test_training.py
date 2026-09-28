import numpy as np

import loop_nn
import loop_trainer
import numpy_nn
from numpy_trainer import DataLoader, Trainer

XOR_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_TARGETS = np.array([[0], [1], [1], [0]], dtype=float)


def test_hidden_layer_learns_xor():
    np.random.seed(1)
    network = numpy_nn.Network(
        numpy_nn.Layer.build(2, 4, numpy_nn.random_weights(2, 4), numpy_nn.zero_bias(4), numpy_nn.ACTIVATIONS.SIGMOID),
        numpy_nn.Layer.build(4, 1, numpy_nn.random_weights(4, 1), numpy_nn.zero_bias(1), numpy_nn.ACTIVATIONS.SIGMOID),
    )
    trainer = Trainer(network, numpy_nn.COSTS.MSE, learning_rate=0.7)

    trainer.train(DataLoader(XOR_INPUTS, XOR_TARGETS, batch_size=1, shuffle=False), epochs=5000)

    assert trainer.evaluate(XOR_INPUTS, XOR_TARGETS)[1] == 1.0


def test_regressors_skip_accuracy():
    inputs = np.linspace(-1, 1, 20).reshape(-1, 1)
    targets = 3 * inputs + 1

    numpy_trainer_ = Trainer(
        numpy_nn.Network(numpy_nn.Layer.build(1, 1, [[0.0]], [0.0], numpy_nn.ACTIVATIONS.IDENTITY)),
        numpy_nn.COSTS.MSE, learning_rate=0.1, classifier=False
    )
    loop_run = loop_trainer.Trainer(
        loop_nn.Network(loop_nn.Layer.build(1, 1, [[0.0]], [0.0], loop_nn.ACTIVATIONS.IDENTITY)),
        loop_nn.COSTS.MSE, learning_rate=0.1, classifier=False
    )
    numpy_history = numpy_trainer_.train(DataLoader(inputs, targets, batch_size=4, shuffle=False), epochs=50)
    loop_history = loop_run.train(
        loop_trainer.DataLoader(inputs.tolist(), targets.tolist(), batch_size=4, shuffle=False), epochs=50
    )

    for history, trainer in ((numpy_history, numpy_trainer_), (loop_history, loop_run)):
        assert all(result.accuracy is None for result in history)
        assert "accuracy" not in str(history[-1])
        assert history[-1].loss < 1e-3
    assert numpy_trainer_.evaluate(inputs, targets)[1] is None
    assert loop_run.evaluate(inputs.tolist(), targets.tolist())[1] is None


def test_numpy_network_save_and_load(tmp_path):
    network = numpy_nn.Network(
        numpy_nn.Layer.build(3, 4, np.ones((4, 3)), np.arange(4.0), numpy_nn.ACTIVATIONS.RELU),
        numpy_nn.Layer.build(4, 2, np.eye(2, 4), np.zeros(2), numpy_nn.ACTIVATIONS.SOFTMAX),
    )
    inputs = np.random.default_rng(0).normal(size=(5, 3))

    network.save(tmp_path / "network.npz")
    loaded = numpy_nn.Network.load(tmp_path / "network.npz")

    np.testing.assert_array_equal(loaded.forward(inputs), network.forward(inputs))
    assert [layer.activation for layer in loaded.layers] == [layer.activation for layer in network.layers]


def test_loop_network_save_and_load(tmp_path):
    network = loop_nn.Network(
        loop_nn.Layer.build(2, 3, [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]], [0.0, 0.1, 0.2], loop_nn.ACTIVATIONS.SIGMOID),
        loop_nn.Layer.build(3, 1, [[0.7, -0.8, 0.9]], [0.5], loop_nn.ACTIVATIONS.SIGMOID),
    )
    inputs = [[0.0, 1.0], [1.0, 0.5]]

    network.save(tmp_path / "network.json")
    loaded = loop_nn.Network.load(tmp_path / "network.json")

    assert loaded.forward(inputs) == network.forward(inputs)
