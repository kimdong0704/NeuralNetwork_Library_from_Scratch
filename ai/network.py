from .layer import Layer


class Network:
    def __init__(self, *layers: Layer):
        self.layers: list[Layer] = list(layers)

    def forward(self, inputs: list[float]) -> list[float]:
        values = inputs

        for layer in self.layers:
            values = layer.forward(values)

        return values

    def backward(self, error: list[float], learning_rate: float) -> list[float]:
        """Backpropagates `error` (target - prediction, at the output layer)
        through the layers in reverse, updating each one's weights/bias.
        Must be called after `forward` so each layer has its cached
        input/output from that pass. Returns the gradient propagated past
        the first layer (dLoss/dInput of the whole network).
        """
        gradient = error

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

        return gradient
