from .config import ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS, REPORT_EACH_EPOCH, REPORT_INTERVAL
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
        training_data: list[list[float]],
        targets: list[float] | list[list[float]],
    ) -> None:
        self.errors_by_epoch = {}

        epoch = 0
        average_error = 0.0
        while epoch < MAX_EPOCHS:
            epoch += 1

            total_error = 0.0
            for inputs, target in zip(training_data, targets):
                prediction = self.network.forward(inputs)

                target_values = None

                if isinstance(target, (list, tuple)):
                    target_values = target
                else:
                    target_values = [target]

                error = []
                for target_value, predicted_value in zip(target_values, prediction):
                    difference = target_value - predicted_value
                    error.append(difference)
                    total_error += abs(difference)

                self.network.backward(error, LEARNING_RATE)

            average_error = total_error / len(training_data)
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
