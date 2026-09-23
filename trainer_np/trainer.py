import numpy as np

from neural_network_np.network import Network

from .config import BATCH_SIZE, COST, ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS, REPORT_EACH_EPOCH, REPORT_INTERVAL
from .reporter import Reporter


class Trainer:
    def __init__(self, network: Network):
        self._validate_hidden_layers(network)

        self.network = network
        self.errors_by_epoch: dict[int, float] = {}

    @staticmethod
    def _validate_hidden_layers(network: Network) -> None:
        for layer in network.layers[:-1]:
            if not layer.gradient_descent:
                raise ValueError(
                    "Hidden layers must use a differentiable activation (one "
                    "with a derivative) for backpropagation; only the output "
                    "layer may use a non-differentiable activation like STEP."
                )

    def train(
        self,
        training_data: np.ndarray,
        targets: np.ndarray
    ) -> None:
        training_data = np.asarray(training_data, dtype=float)
        targets = np.asarray(targets, dtype=float)

        self.errors_by_epoch = {}

        epoch = 0
        average_error = 0.0
        for epoch in range(1, MAX_EPOCHS + 1):
            average_error = self._train_epoch(training_data, targets)
            self.errors_by_epoch[epoch] = average_error

            if REPORT_EACH_EPOCH and epoch % REPORT_INTERVAL == 0:
                Reporter.epoch_summary(
                    epoch=epoch,
                    total_epochs=MAX_EPOCHS,
                    average_error=average_error
                )

            if average_error < ERROR_THRESHOLD:
                Reporter.training_completed(
                    epoch=epoch,
                    total_epochs=MAX_EPOCHS,
                    average_error=average_error
                )
                return

        Reporter.max_epochs_reached(
            epoch=epoch,
            total_epochs=MAX_EPOCHS,
            average_error=average_error
        )

    def _train_epoch(
        self,
        training_data: np.ndarray,
        targets: np.ndarray
    ) -> float:
        total_error = 0.0
        batch_count = 0
        n_samples = training_data.shape[0]
        
        for start in range(0, n_samples, BATCH_SIZE):
            end = start + BATCH_SIZE
            total_error += self._train_batch(training_data[start:end], targets[start:end])
            batch_count += 1

        return total_error / batch_count

    def _train_batch(
        self,
        batch_inputs: np.ndarray,
        batch_targets: np.ndarray
    ) -> float:
        predictions = self.network.forward(batch_inputs)

        batch_cost = COST.function(predictions, batch_targets)
        gradients = COST.derivative(predictions, batch_targets)

        self.network.backward(gradients, LEARNING_RATE)
        return batch_cost
