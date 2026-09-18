import numpy as np

from .config import ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS
from .network import Network
from .reporter import Reporter


class Trainer:
    def __init__(self, network: Network):
        for layer in network.layers[:-1]:
            if not layer.gradient_descent:
                raise ValueError(
                    "Hidden layers must use a differentiable activation (one "
                    "with a derivative) for backpropagation; only the output "
                    "layer may use a non-differentiable activation like STEP."
                )

        self.network = network
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
                prediction = self.network.forward(inputs)
                error = np.atleast_1d(target) - prediction
                total_error += np.sum(np.abs(error))

                self.network.backward(error, LEARNING_RATE)

            average_error = total_error / len(training_data)
            self.errors_by_epoch[epoch] = average_error

            Reporter.epoch_summary(
                epoch=epoch,
                total_epochs=MAX_EPOCHS,
                average_error=average_error
            )

            if average_error < ERROR_THRESHOLD:
                Reporter.training_completed()
                return

        Reporter.max_epochs_reached()
