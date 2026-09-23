from .layer import Layer


class Network:
    def __init__(self, *layers: Layer):
        self.layers: list[Layer] = list(layers)

    def forward(self, inputs: list[float]) -> list[float]:
        values = inputs

        for layer in self.layers:
            values = layer.forward(values)

        return values

    def backward(self, gradients: list[float], learning_rate: float) -> list[float]:
        deltas = gradients

        for layer in reversed(self.layers):
            deltas = layer.backward(deltas, learning_rate)

        return deltas
