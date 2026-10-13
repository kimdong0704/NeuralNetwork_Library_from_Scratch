import numpy as np

from cnn import Conv2D, Flatten, MaxPool2D, Pad
from numpy_nn import Layer
from numpy_utils import ACTIVATIONS, random_weights, random_weights_conv, zero_bias, zero_bias_conv

from .config import (
	C1_MAPS,
    C3_MAPS,
    C5_MAPS,
    F6_UNITS,
    FLAT_SIZE,
    INPUT_CHANNELS,
	KERNEL_SIZE,
    CONV_STRIDE,
    OUTPUT_UNITS,
    PADDING,
    POOL_SIZE,
    POOL_STRIDE,
    WEIGHT_HIGH,
    WEIGHT_LOW,
)


class LeNet5:
	def __init__(self, seed: int | None = None):
		# same seed -> same starting weights
        # None -> different every time
		if seed is not None:
			np.random.seed(seed)
	
		self.init_weights()
		self.build_layers()

	def init_weights(self) -> None:
		# convolutions: (out maps, in maps, kernel rows, kernel cols), one bias per out map
		self.c1_weights = random_weights_conv(
			in_channels=INPUT_CHANNELS,
			out_channels=C1_MAPS,
			kernel_rows=KERNEL_SIZE,
			kernel_cols=KERNEL_SIZE,
			low=WEIGHT_LOW,
			high=WEIGHT_HIGH
        )
		self.c1_bias = zero_bias_conv(out_channels=C1_MAPS)

		self.c3_weights = random_weights_conv(
			in_channels=C1_MAPS,
            out_channels=C3_MAPS,
            kernel_rows=KERNEL_SIZE,
            kernel_cols=KERNEL_SIZE,
            low=WEIGHT_LOW,
            high=WEIGHT_HIGH
        )
		self.c3_bias = zero_bias_conv(out_channels=C3_MAPS)

		self.c5_weights = random_weights_conv(
			in_channels=C3_MAPS,
            out_channels=C5_MAPS,
            kernel_rows=KERNEL_SIZE,
            kernel_cols=KERNEL_SIZE,
            low=WEIGHT_LOW,
            high=WEIGHT_HIGH
        )
		self.c5_bias = zero_bias_conv(out_channels=C5_MAPS)

		# dense: (outputs, inputs), same as numpy_nn.Layer
		self.f6_weights = random_weights(
			input_size=FLAT_SIZE,
            output_size=F6_UNITS,
            low=WEIGHT_LOW,
            high=WEIGHT_HIGH
        )
		self.f6_bias = zero_bias(output_size=F6_UNITS)

		self.output_weights = random_weights(
			input_size=F6_UNITS,
            output_size=OUTPUT_UNITS,
            low=WEIGHT_LOW,
            high=WEIGHT_HIGH
        )
		self.output_bias = zero_bias(output_size=OUTPUT_UNITS)

	def build_layers(self) -> None:
		self.pad_layer = Pad(amount=PADDING)

		self.c1_layer = Conv2D(weights=self.c1_weights, bias=self.c1_bias, activation=ACTIVATIONS.RELU, stride=CONV_STRIDE)
		self.s2_layer = MaxPool2D(pool_rows=POOL_SIZE, pool_cols=POOL_SIZE, stride=POOL_STRIDE)

		self.c3_layer = Conv2D(weights=self.c3_weights, bias=self.c3_bias, activation=ACTIVATIONS.RELU, stride=CONV_STRIDE)
		self.s4_layer = MaxPool2D(pool_rows=POOL_SIZE, pool_cols=POOL_SIZE, stride=POOL_STRIDE)

		self.c5_layer = Conv2D(weights=self.c5_weights, bias=self.c5_bias, activation=ACTIVATIONS.RELU, stride=CONV_STRIDE)
		self.flatten_layer = Flatten()

		self.f6_layer = Layer.build(input_size=FLAT_SIZE, output_size=F6_UNITS, weight=self.f6_weights, bias=self.f6_bias, activation=ACTIVATIONS.RELU)
		self.output_layer = Layer.build(input_size=F6_UNITS, output_size=OUTPUT_UNITS, weight=self.output_weights, bias=self.output_bias, activation=ACTIVATIONS.SOFTMAX)

		# Layer.build copies its arrays, so point our names at the copies the layers actually use
		self.f6_weights, self.f6_bias = self.f6_layer.weights, self.f6_layer.bias
		self.output_weights, self.output_bias = self.output_layer.weights, self.output_layer.bias

		# in forward order
		self.layers = [
			self.pad_layer,
			self.c1_layer,
			self.s2_layer,
			self.c3_layer,
			self.s4_layer,
			self.c5_layer,
			self.flatten_layer,
			self.f6_layer,
			self.output_layer,
		]

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		outputs = inputs
		for layer in self.layers:
			outputs = layer.forward(outputs)
		return outputs
