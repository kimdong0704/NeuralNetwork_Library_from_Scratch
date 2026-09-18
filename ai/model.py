import numpy as np

from .activations import ACTIVATIONS
from .helpers import resolve_activation, zero_weights
from .layer import Layer
from .network import Network

INPUT_SIZE = 2
OUTPUT_SIZE = 1

def create_model() -> Network:
    weight = zero_weights(input_size=INPUT_SIZE, output_size=1)
    activation_function, activation_derivative = resolve_activation(ACTIVATIONS.SIGMOID)

    return Network(
        Layer.build(
            input_size=2,
            output_size=4,
            weight=weight,
            activation_function=activation_function,
            activation_derivative=activation_derivative
        )
    )
