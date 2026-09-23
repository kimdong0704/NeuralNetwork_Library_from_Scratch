import numpy as np

from neural_network_np.network import Network

from .config import (
    BATCH_SIZE, COST, ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS, REPORT_EACH_EPOCH, REPORT_INTERVAL, SHUFFLE
)
from .cost import Cost
from .metrics import count_correct
from .reporter import Reporter


class Trainer:
    def __init__(
        self,
        network: Network,
        learning_rate: float = LEARNING_RATE,
        cost: Cost = COST,
        max_epochs: int = MAX_EPOCHS,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = SHUFFLE,
        report_interval: int = REPORT_INTERVAL,
    ):
        self._validate_hidden_layers(network)

        self.network = network
        self.learning_rate = learning_rate
        self.cost = cost
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.report_interval = report_interval

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
        validation_targets: np.ndarray | None = None,
        target_accuracy: float | None = None,
        target_validation_accuracy: float | None = None,
    ) -> None:
        """Trains until the cost drops below ERROR_THRESHOLD, the target accuracies are reached, or max_epochs runs out."""
        training_data = np.asarray(training_data, dtype=float)
        targets = np.asarray(targets, dtype=float)

        self.errors_by_epoch = {}
        self.accuracy_by_epoch = {}
        self.validation_errors_by_epoch = {}
        self.validation_accuracy_by_epoch = {}

        epoch = 0
        for epoch in range(1, self.max_epochs + 1):
            average_error, accuracy = self._train_epoch(training_data, targets)
            self.errors_by_epoch[epoch] = average_error
            self.accuracy_by_epoch[epoch] = accuracy

            if validation_data is not None and validation_targets is not None:
                validation_error, validation_accuracy = self.evaluate(validation_data, validation_targets)
                self.validation_errors_by_epoch[epoch] = validation_error
                self.validation_accuracy_by_epoch[epoch] = validation_accuracy

            if REPORT_EACH_EPOCH and epoch % self.report_interval == 0:
                Reporter.epoch_summary(**self._epoch_metrics(epoch))

            if average_error < ERROR_THRESHOLD or self._targets_reached(epoch, target_accuracy, target_validation_accuracy):
                Reporter.training_completed(**self._epoch_metrics(epoch))
                return

        Reporter.max_epochs_reached(**self._epoch_metrics(epoch))

    def _targets_reached(
        self,
        epoch: int,
        target_accuracy: float | None,
        target_validation_accuracy: float | None
    ) -> bool:
        # no targets given means training only stops on the error threshold or max_epochs
        if target_accuracy is None and target_validation_accuracy is None:
            return False

        if target_accuracy is not None and self.accuracy_by_epoch[epoch] < target_accuracy:
            return False

        if target_validation_accuracy is not None:
            validation_accuracy = self.validation_accuracy_by_epoch.get(epoch)
            if validation_accuracy is None or validation_accuracy < target_validation_accuracy:
                return False

        return True

    def _epoch_metrics(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "total_epochs": self.max_epochs,
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
        return self.cost.function(predictions, targets), count_correct(predictions, targets) / inputs.shape[0]

    def _train_epoch(
        self,
        training_data: np.ndarray,
        targets: np.ndarray
    ) -> tuple[float, float]:
        total_error = 0.0
        total_correct = 0
        batch_count = 0
        n_samples = training_data.shape[0]

        if self.shuffle:
            # a new order every epoch keeps batches from repeating the same mix of examples
            order = np.random.permutation(n_samples)
            training_data, targets = training_data[order], targets[order]

        for start in range(0, n_samples, self.batch_size):
            end = start + self.batch_size
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

        batch_cost = self.cost.function(predictions, batch_targets)
        gradients = self.cost.derivative(predictions, batch_targets)
        # measured before the update, so accuracy reflects what the network predicted during the epoch
        batch_correct = count_correct(predictions, batch_targets)

        self.network.backward(gradients, self.learning_rate)
        return batch_cost, batch_correct
