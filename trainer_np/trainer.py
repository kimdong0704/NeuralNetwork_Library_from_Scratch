import numpy as np
from threadpoolctl import threadpool_limits

from neural_network_np.config import DTYPE
from neural_network_np.network import Network

from .config import (
    BATCH_SIZE, BLAS_THREADS, COST, ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS, REPORT_EACH_EPOCH, REPORT_INTERVAL,
    SHUFFLE
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
        error_threshold: float | None = ERROR_THRESHOLD,
    ):
        self._validate_hidden_layers(network)

        self.network = network
        self.learning_rate = learning_rate
        self.cost = cost
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.report_interval = report_interval
        # None turns the error stop off, so training always runs to max_epochs unless targets are reached
        self.error_threshold = error_threshold

        self._reset_history()

    @staticmethod
    def _validate_hidden_layers(network: Network) -> None:
        for layer in network.layers[:-1]:
            if not layer.gradient_descent:
                raise ValueError(
                    "Hidden layers must use a differentiable activation (one "
                    "with a derivative) for backpropagation; only the output "
                    "layer may use a non-differentiable activation like STEP."
                )

    def _reset_history(self) -> None:
        self.errors_by_epoch: dict[int, float] = {}
        self.accuracy_by_epoch: dict[int, float] = {}
        self.validation_errors_by_epoch: dict[int, float] = {}
        self.validation_accuracy_by_epoch: dict[int, float] = {}

    def train(
        self,
        training_data: np.ndarray,
        targets: np.ndarray,
        validation_data: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
        target_accuracy: float | None = None,
        target_validation_accuracy: float | None = None,
    ) -> None:
        """Trains until the cost drops below error_threshold, the target accuracies are reached, or max_epochs runs out."""
        # converted once here so no batch or epoch has to convert again
        training_data = np.asarray(training_data, dtype=DTYPE)
        targets = np.asarray(targets, dtype=DTYPE)
        if validation_data is not None and validation_targets is not None:
            validation_data = np.asarray(validation_data, dtype=DTYPE)
            validation_targets = np.asarray(validation_targets, dtype=DTYPE)

        self._reset_history()

        epoch = 0
        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            for epoch in range(1, self.max_epochs + 1):
                self.errors_by_epoch[epoch], self.accuracy_by_epoch[epoch] = self._train_epoch(training_data, targets)

                if validation_data is not None and validation_targets is not None:
                    validation_error, validation_accuracy = self.evaluate(validation_data, validation_targets)
                    self.validation_errors_by_epoch[epoch] = validation_error
                    self.validation_accuracy_by_epoch[epoch] = validation_accuracy

                if REPORT_EACH_EPOCH and epoch % self.report_interval == 0:
                    Reporter.epoch_summary(**self._epoch_metrics(epoch))

                if self._should_stop(epoch, target_accuracy, target_validation_accuracy):
                    Reporter.training_completed(**self._epoch_metrics(epoch))
                    return

        Reporter.max_epochs_reached(**self._epoch_metrics(epoch))

    def _should_stop(
        self,
        epoch: int,
        target_accuracy: float | None,
        target_validation_accuracy: float | None
    ) -> bool:
        if self.error_threshold is not None and self.errors_by_epoch[epoch] < self.error_threshold:
            return True

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
        targets = np.asarray(targets, dtype=DTYPE)
        cost, correct = self._score(self.network.forward(inputs), targets)
        return cost, correct / targets.shape[0]

    def _score(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> tuple[float, int]:
        return self.cost.function(predictions, targets), count_correct(predictions, targets)

    def _train_epoch(
        self,
        training_data: np.ndarray,
        targets: np.ndarray
    ) -> tuple[float, float]:
        total_error = 0.0
        total_correct = 0
        n_samples = training_data.shape[0]

        if self.shuffle:
            # a new order every epoch keeps batches from repeating the same mix of examples
            order = np.random.permutation(n_samples)
            training_data, targets = training_data[order], targets[order]

        batch_starts = range(0, n_samples, self.batch_size)
        for start in batch_starts:
            end = start + self.batch_size
            batch_cost, batch_correct = self._train_batch(training_data[start:end], targets[start:end])
            total_error += batch_cost
            total_correct += batch_correct

        return total_error / len(batch_starts), total_correct / n_samples

    def _train_batch(
        self,
        batch_inputs: np.ndarray,
        batch_targets: np.ndarray
    ) -> tuple[float, int]:
        predictions = self.network.forward(batch_inputs)

        # measured before the update, so accuracy reflects what the network predicted during the epoch
        batch_cost, batch_correct = self._score(predictions, batch_targets)
        gradients = self.cost.derivative(predictions, batch_targets)

        self.network.backward(gradients, self.learning_rate)
        return batch_cost, batch_correct
