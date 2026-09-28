from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from numpy_nn.costs import Cost
from numpy_nn.network import Network

from .dataloader import DataLoader
from .metrics import count_correct
from .reporter import EpochResult, Reporter

# Mini-batch matrices are small, so a few BLAS threads beat the default of one per core,
# which spends most of its time coordinating threads (about 10x slower on MNIST batches of 32)
BLAS_THREADS = 4


class Trainer:
    """Mini-batch gradient descent: forward, cost, backward and one weight update per batch."""

    def __init__(
        self,
        network: Network,
        cost: Cost,
        learning_rate: float,
        classifier: bool = True
    ):
        """classifier=False is for regression: accuracy is then not measured or reported."""
        self._validate_hidden_layers(network)

        self.network = network
        self.cost = cost
        self.learning_rate = learning_rate
        self.classifier = classifier
        self.reporter = Reporter()

    @staticmethod
    def _validate_hidden_layers(network: Network) -> None:
        for layer in network.layers[:-1]:
            if not layer.differentiable:
                raise ValueError(
                    "Hidden layers need an activation with a derivative for backpropagation; "
                    "only the output layer may use a non-differentiable activation like STEP."
                )

    @property
    def history(self) -> list[EpochResult]:
        return self.reporter.history

    def train(
        self,
        loader: DataLoader,
        epochs: int,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        print_interval: int = 0,
        stop_when: Callable[[EpochResult], bool] | None = None
    ) -> list[EpochResult]:
        """Trains for up to `epochs` more epochs; training ends early once stop_when returns True for an epoch."""
        first_epoch = len(self.history) + 1
        stopped_early = False

        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            for epoch in range(first_epoch, first_epoch + epochs):
                loss, accuracy = self._train_epoch(loader)

                validation_loss = validation_accuracy = None
                if validation is not None:
                    validation_loss, validation_accuracy = self.evaluate(*validation)

                result = EpochResult(epoch, loss, accuracy, validation_loss, validation_accuracy)
                self.reporter.record(result, print_interval)

                if stop_when is not None and stop_when(result):
                    stopped_early = True
                    break

        self.reporter.finish(stopped_early, print_interval)
        return self.history

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> tuple[float, float | None]:
        """Returns (cost, accuracy) of the network on the data without training on it."""
        targets = np.asarray(targets)
        predicted = self.network.forward(inputs)

        return self.cost.function(predicted, targets), self._accuracy(count_correct(predicted, targets), len(targets))

    def _accuracy(self, correct: int, total: int) -> float | None:
        return correct / total if self.classifier else None

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float | None]:
        total_cost = 0.0
        total_correct = 0

        for inputs, targets in loader:
            predicted = self.network.forward(inputs)

            # measured before the update, so they describe what the network predicted during the epoch
            total_cost += self.cost.function(predicted, targets) * len(inputs)
            if self.classifier:
                total_correct += count_correct(predicted, targets)

            self.network.backward(self.cost.derivative(predicted, targets))
            self.network.step(self.learning_rate)

        return total_cost / len(loader.inputs), self._accuracy(total_correct, len(loader.inputs))
