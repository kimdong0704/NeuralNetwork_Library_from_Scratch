"""Trains the same network with loop_nn and numpy_nn and compares their results and speed.

Run from the repository root:  python benchmarks/compare_implementations.py
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import loop_nn
import loop_trainer
import numpy_nn
import numpy_trainer


def build_pair(sizes, hidden, output, seed=0):
    """The same randomly initialized network, once per implementation."""
    rng = np.random.default_rng(seed)
    numpy_layers, loop_layers = [], []

    for index, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
        name = output if index == len(sizes) - 2 else hidden
        weights = rng.uniform(-0.5, 0.5, size=(outputs, inputs))
        bias = np.zeros(outputs)

        numpy_layers.append(numpy_nn.Layer.build(inputs, outputs, weights, bias, numpy_nn.ACTIVATIONS.by_name(name)))
        loop_layers.append(loop_nn.Layer.build(inputs, outputs, weights.tolist(), bias.tolist(), loop_nn.ACTIVATIONS.by_name(name)))

    return numpy_nn.Network(*numpy_layers), loop_nn.Network(*loop_layers)


def timed(train):
    start = time.perf_counter()
    history = train()
    return history[-1], time.perf_counter() - start


def compare(title, sizes, hidden, output, cost, inputs, targets, batch_size, epochs, learning_rate):
    numpy_network, loop_network = build_pair(sizes, hidden, output)

    numpy_run = numpy_trainer.Trainer(numpy_network, getattr(numpy_nn.COSTS, cost), learning_rate)
    loop_run = loop_trainer.Trainer(loop_network, getattr(loop_nn.COSTS, cost), learning_rate)

    numpy_result, numpy_seconds = timed(lambda: numpy_run.train(
        numpy_trainer.DataLoader(inputs, targets, batch_size, shuffle=False), epochs
    ))
    loop_result, loop_seconds = timed(lambda: loop_run.train(
        loop_trainer.DataLoader(inputs.tolist(), targets.tolist(), batch_size, shuffle=False), epochs
    ))

    largest_difference = max(
        float(np.max(np.abs(np.array(loop_layer.weights) - numpy_layer.weights)))
        for numpy_layer, loop_layer in zip(numpy_network.layers, loop_network.layers)
    )

    print(f"\n{title}  ({' -> '.join(map(str, sizes))} | batch size: {batch_size} | epochs: {epochs})")
    print(f"  loop_nn : {loop_result}  | {loop_seconds:7.2f} s")
    print(f"  numpy_nn: {numpy_result}  | {numpy_seconds:7.2f} s")
    print(f"  largest weight difference: {largest_difference:.1e} | numpy_nn speedup: {loop_seconds / numpy_seconds:.1f}x")


def main():
    moons = np.loadtxt(REPO_ROOT / "data" / "two_moons" / "train.csv", delimiter=",")
    compare(
        "Two moons", [2, 8, 1], "sigmoid", "sigmoid", "MSE",
        moons[:, :2], moons[:, 2:], batch_size=1, epochs=20, learning_rate=0.7
    )

    # MNIST-shaped data: a slice of 2,000 images keeps the pure-Python run to a few seconds
    images = np.frombuffer((REPO_ROOT / "data" / "mnist" / "train-images-idx3-ubyte").read_bytes(), dtype=np.uint8, offset=16)
    labels = np.frombuffer((REPO_ROOT / "data" / "mnist" / "train-labels-idx1-ubyte").read_bytes(), dtype=np.uint8, offset=8)
    inputs = images.reshape(-1, 784)[:2000] / 255.0
    targets = np.eye(10)[labels[:2000]]
    compare(
        "MNIST slice", [784, 32, 10], "relu", "softmax", "CROSS_ENTROPY",
        inputs, targets, batch_size=32, epochs=1, learning_rate=0.3
    )


if __name__ == "__main__":
    main()
