from .activations import ACTIVATIONS
from .helpers import resolve_activation, zero_weights, random_weights
from .layer import Layer
from .network import Network

INPUT_SIZE = 2
OUTPUT_SIZE = 1

def create_model() -> Network:
    weight1 = random_weights(input_size=INPUT_SIZE, output_size=2)
    weight2 = random_weights(input_size=2, output_size=OUTPUT_SIZE)
    activation_function, activation_derivative = resolve_activation(ACTIVATIONS.SIGMOID)

    return Network(
        Layer.build(
            input_size=INPUT_SIZE,
            output_size=2,
            weight=weight1,
            activation_function=activation_function,
            activation_derivative=activation_derivative
        ),
        Layer.build(
            input_size=2,
            output_size=OUTPUT_SIZE,
            weight=weight2,
            activation_function=activation_function,
            activation_derivative=activation_derivative
        )
    )
