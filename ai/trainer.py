import numpy as np

from .config import ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS
from .network import Network
from .reporter import Reporter


class Trainer:
    def __init__(self, network: Network):
        if len(network.layers) != 1 or network.layers[0].weights.shape[0] != 1:
            raise NotImplementedError(
                "Training only supports a single-layer, single-node network for "
                "now; backpropagation for hidden layers/multiple nodes is not "
                "implemented yet."
            )

        self.network = network
        self.layer = network.layers[0]
        self.errors_by_epoch: dict[int, float] = {}

    def train(
        self,
        training_data: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.errors_by_epoch = {}

        epoch = 0
        while epoch < MAX_EPOCHS:
            epoch += 1

            total_error = 0.0
            for inputs, target in zip(training_data, targets):
                prediction = self.network.forward(inputs)[0]
                error = target - prediction
                total_error += abs(error)

                # single node
                self.layer.update(0, inputs, prediction, error, LEARNING_RATE)

            average_error = total_error / len(training_data)
            self.errors_by_epoch[epoch] = average_error

            Reporter.epoch_summary(
                epoch=epoch,
                total_epochs=MAX_EPOCHS,
                average_error=average_error,
                weights=self.layer.weights[0],
            )

            if average_error < ERROR_THRESHOLD:
                Reporter.training_completed()
                return

        Reporter.max_epochs_reached()
