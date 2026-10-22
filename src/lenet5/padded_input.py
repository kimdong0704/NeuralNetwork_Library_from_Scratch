"""
    The Java API's forward(image32): images that are already padded to 32x32 (as MNISTLoader returns them)
	go through every LeNet5 layer after its own Pad layer.
"""
import numpy as np

from .lenet5 import LeNet5


def layers_after_pad(net: LeNet5) -> list:
	return net.layers[net.layers.index(net.pad_layer) + 1:]


def forward(net: LeNet5, image32: np.ndarray) -> np.ndarray:
	# one 32x32 image as a batch of one: (1, 1, 32, 32) -> probabilities (1, OUTPUT_UNITS)
	outputs = image32[np.newaxis, np.newaxis, :, :]
	for layer in layers_after_pad(net):
		outputs = layer.forward(outputs)
	return outputs
