from neural_network.network import Network

from .config import COST, ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS, REPORT_EACH_EPOCH, REPORT_INTERVAL
from .metrics import is_correct
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
        training_data: list[list[float]],
        targets: list[list[float]],
        validation_data: list[list[float]] | None = None,
        validation_targets: list[list[float]] | None = None
    ) -> None:
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
        inputs: list[list[float]],
        targets: list[list[float]]
    ) -> tuple[float, float]:
        """Returns (average error, accuracy) of the network on the data without training on it."""
        total_error = 0.0
        correct = 0
        for input_row, target in zip(inputs, targets):
            prediction = self.network.forward(input_row)
            total_error += self._sample_error(prediction, target)
            if is_correct(prediction, target):
                correct += 1

        return total_error / len(inputs), correct / len(inputs)

    def _train_epoch(
        self,
        training_data: list[list[float]],
        targets: list[list[float]]
    ) -> tuple[float, float]:
        total_error = 0.0
        correct = 0
        for inputs, target in zip(training_data, targets):
            sample_error, sample_correct = self._train_sample(inputs, target)
            total_error += sample_error
            if sample_correct:
                correct += 1

        return total_error / len(training_data), correct / len(training_data)

    def _train_sample(
        self,
        inputs: list[float],
        target: list[float]
    ) -> tuple[float, bool]:
        prediction = self.network.forward(inputs)

        gradients = []
        for target_value, predicted_value in zip(target, prediction):
            gradients.append(COST.derivative(predicted_value, target_value))

        # measured before the update, so accuracy reflects what the network predicted during the epoch
        sample_error = self._sample_error(prediction, target)
        sample_correct = is_correct(prediction, target)

        self.network.backward(gradients, LEARNING_RATE)
        return sample_error, sample_correct

    @staticmethod
    def _sample_error(prediction: list[float], target: list[float]) -> float:
        sample_error = 0.0
        for target_value, predicted_value in zip(target, prediction):
            sample_error += COST.function(predicted_value, target_value)
        return sample_error
