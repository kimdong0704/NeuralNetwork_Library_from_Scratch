import numpy as np

from neural_network_np.network import Network

from .config import BATCH_SIZE, COST, ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS, REPORT_EACH_EPOCH, REPORT_INTERVAL
from .metrics import count_correct
from .reporter import Reporter


class Trainer:
    def __init__(self, network: Network):
        self._validate_hidden_layers(network)

        self.network = network
        self.errors_by_epoch: dict[int, float] = {}
        self.accuracy_by_epoch: dict[int, float] = {}
        self.validation_errors_by_epoch: dict[int, float] = {}
        self.validation_accuracy_by_epoch: dict[int, float] = {}

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
        targets: np.ndarray,
        validation_data: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None
    ) -> None:
        training_data = np.asarray(training_data, dtype=float)
        targets = np.asarray(targets, dtype=float)

        self.errors_by_epoch = {}
        self.accuracy_by_epoch = {}
        self.validation_errors_by_epoch = {}
        self.validation_accuracy_by_epoch = {}

        epoch = 0
        average_error = 0.0
        for epoch in range(1, MAX_EPOCHS + 1):
            average_error, accuracy = self._train_epoch(training_data, targets)
            self.errors_by_epoch[epoch] = average_error
            self.accuracy_by_epoch[epoch] = accuracy

            if validation_data is not None and validation_targets is not None:
                validation_error, validation_accuracy = self.evaluate(validation_data, validation_targets)
                self.validation_errors_by_epoch[epoch] = validation_error
                self.validation_accuracy_by_epoch[epoch] = validation_accuracy

            if REPORT_EACH_EPOCH and epoch % REPORT_INTERVAL == 0:
                Reporter.epoch_summary(**self._epoch_metrics(epoch))

            if average_error < ERROR_THRESHOLD:
                Reporter.training_completed(**self._epoch_metrics(epoch))
                return

        Reporter.max_epochs_reached(**self._epoch_metrics(epoch))

    def _epoch_metrics(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "total_epochs": MAX_EPOCHS,
            "average_error": self.errors_by_epoch[epoch],
            "accuracy": self.accuracy_by_epoch[epoch],
            "validation_error": self.validation_errors_by_epoch.get(epoch),
            "validation_accuracy": self.validation_accuracy_by_epoch.get(epoch),
        }

    def evaluate(
        self,
        inputs: np.ndarray,
        targets: np.ndarray
    ) -> tuple[float, float]:
        """Returns (cost, accuracy) of the network on the data without training on it."""
        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets, dtype=float)

        predictions = self.network.forward(inputs)
        return COST.function(predictions, targets), count_correct(predictions, targets) / inputs.shape[0]

    def _train_epoch(
        self,
        training_data: np.ndarray,
        targets: np.ndarray
    ) -> tuple[float, float]:
        total_error = 0.0
        total_correct = 0
        batch_count = 0
        n_samples = training_data.shape[0]

        for start in range(0, n_samples, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_cost, batch_correct = self._train_batch(training_data[start:end], targets[start:end])
            total_error += batch_cost
            total_correct += batch_correct
            batch_count += 1

        return total_error / batch_count, total_correct / n_samples

    def _train_batch(
        self,
        batch_inputs: np.ndarray,
        batch_targets: np.ndarray
    ) -> tuple[float, int]:
        predictions = self.network.forward(batch_inputs)

        batch_cost = COST.function(predictions, batch_targets)
        gradients = COST.derivative(predictions, batch_targets)
        # measured before the update, so accuracy reflects what the network predicted during the epoch
        batch_correct = count_correct(predictions, batch_targets)

        self.network.backward(gradients, LEARNING_RATE)
        return batch_cost, batch_correct
