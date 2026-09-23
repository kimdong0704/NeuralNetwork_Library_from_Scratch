from neural_network.network import Network

from .config import COST, ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS, REPORT_EACH_EPOCH, REPORT_INTERVAL
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
        training_data: list[list[float]],
        targets: list[list[float]]
    ) -> None:
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
        training_data: list[list[float]],
        targets: list[list[float]]
    ) -> float:
        total_error = 0.0
        for inputs, target in zip(training_data, targets):
            total_error += self._train_sample(inputs, target)

        return total_error / len(training_data)

    def _train_sample(
        self,
        inputs: list[float],
        target: list[float]
    ) -> float:
        prediction = self.network.forward(inputs)

        sample_error = 0.0
        gradients = []
        for target_value, predicted_value in zip(target, prediction):
            gradients.append(COST.derivative(predicted_value, target_value))
            sample_error += COST.function(predicted_value, target_value)

        self.network.backward(gradients, LEARNING_RATE)
        return sample_error
