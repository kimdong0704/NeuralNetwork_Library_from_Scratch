from .activations import ACTIVATIONS
from .hyperparameters import zero_bias, random_bias, zero_weights, random_weights
from .layer import Layer
from .network import Network

INPUT_SIZE = 2
OUTPUT_SIZE = 1

def create_model() -> Network:
    weights = random_weights(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)

    biases = random_bias(output_size=OUTPUT_SIZE)

    return Network(
        Layer.build(
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            weight=weights,
            bias=biases,
            activation_function=ACTIVATIONS.RELU.function,
            activation_derivative=ACTIVATIONS.RELU.derivative
        )
    )
