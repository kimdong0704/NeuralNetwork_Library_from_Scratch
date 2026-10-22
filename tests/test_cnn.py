"""The cnn layers match explicit loops, and LeNet-5's forward pass has the right shapes and outputs."""
import numpy as np
import pytest

from cnn import Conv2D, Flatten, MaxPool2D, Pad
from cnn.utils import windows_fast, windows_loop
from lenet5 import LeNet5
from lenet5.model_config import IMAGE_SIZE, INPUT_CHANNELS, OUTPUT_UNITS
from numpy_utils import ACTIVATIONS

BATCH_SIZE = 4
INPUT_SHAPE = (BATCH_SIZE, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


@pytest.mark.parametrize("window, stride", [(2, 1), (2, 2), (3, 2), (5, 1)])
def test_strided_windows_match_loop_windows(window, stride):
    inputs = np.random.default_rng(0).normal(size=(2, 3, 9, 11))
    fast = windows_fast(inputs, window, window, stride)
    loop = windows_loop(inputs, window, window, stride)
    assert np.array_equal(fast, loop)


def test_pad_surrounds_every_channel_with_zeros():
    inputs = np.random.default_rng(0).normal(size=(2, 2, 3, 4))
    padded = Pad(amount=2).forward(inputs)
    assert padded.shape == (2, 2, 7, 8)
    assert np.array_equal(padded[:, :, 2:5, 2:6], inputs)
    assert padded.sum() == pytest.approx(inputs.sum())


@pytest.mark.parametrize("stride", [1, 2])
def test_conv2d_matches_explicit_loop(stride):
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(2, 3, 8, 8))
    weights = rng.normal(size=(4, 3, 3, 3))
    bias = rng.normal(size=(4, 1, 1))

    outputs = Conv2D(weights, bias, ACTIVATIONS.IDENTITY, stride=stride).forward(inputs)

    out_size = (8 - 3) // stride + 1
    expected = np.zeros((2, 4, out_size, out_size))
    for b in range(2):
        for k in range(4):
            for r in range(out_size):
                for c in range(out_size):
                    patch = inputs[b, :, r * stride:r * stride + 3, c * stride:c * stride + 3]
                    expected[b, k, r, c] = np.sum(weights[k] * patch) + bias[k, 0, 0]
    assert np.allclose(outputs, expected)


def test_maxpool_and_flatten():
    inputs = np.arange(64, dtype=float).reshape(2, 2, 4, 4)
    pooled = MaxPool2D(pool_rows=2, pool_cols=2, stride=2).forward(inputs)
    assert np.array_equal(pooled[0, 0], [[5, 7], [13, 15]])
    assert np.array_equal(pooled[0, 1], [[21, 23], [29, 31]])
    assert np.array_equal(pooled[1, 0], [[37, 39], [45, 47]])
    assert np.array_equal(Flatten().forward(pooled), [
        [5, 7, 13, 15, 21, 23, 29, 31],
        [37, 39, 45, 47, 53, 55, 61, 63],
    ])


@pytest.mark.parametrize("layer", [Pad(amount=1), MaxPool2D(pool_rows=2, pool_cols=2, stride=2)])
def test_image_layers_reject_unbatched_inputs(layer):
    with pytest.raises(ValueError, match="batch"):
        layer.forward(np.zeros((1, 4, 4)))


@pytest.mark.parametrize("image", ["random", "zeros", "ones"])
def test_lenet5_outputs_a_probability_distribution(image):
    inputs = {
        "random": np.random.default_rng(0).random(INPUT_SHAPE),
        "zeros": np.zeros(INPUT_SHAPE),
        "ones": np.ones(INPUT_SHAPE),
    }[image]
    outputs = LeNet5(seed=0).forward(inputs)
    assert outputs.shape == (BATCH_SIZE, OUTPUT_UNITS)
    assert np.all(outputs >= 0)
    assert np.allclose(outputs.sum(axis=1), 1.0)


def test_lenet5_batch_matches_one_image_at_a_time():
    inputs = np.random.default_rng(0).random(INPUT_SHAPE)
    net = LeNet5(seed=0)
    batched = net.forward(inputs)
    one_by_one = np.concatenate([net.forward(inputs[i:i + 1]) for i in range(BATCH_SIZE)])
    assert np.allclose(batched, one_by_one)


def test_lenet5_is_deterministic_and_input_sensitive():
    rng = np.random.default_rng(0)
    first, second = rng.random(INPUT_SHAPE), rng.random(INPUT_SHAPE)

    net = LeNet5(seed=0)
    assert np.array_equal(net.forward(first), net.forward(first))
    assert np.array_equal(net.forward(first), LeNet5(seed=0).forward(first))
    assert not np.allclose(net.forward(first), net.forward(second), rtol=0, atol=1e-9)
