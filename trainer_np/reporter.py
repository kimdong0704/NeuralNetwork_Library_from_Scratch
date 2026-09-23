import numpy as np

from neural_network_np.network import Network

from .config import REPORT_DIGITS
from .metrics import count_correct


def _format(value: float | list | np.ndarray) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        return "[" + ", ".join(_format(item) for item in value) + "]"

    return f"{value:.{REPORT_DIGITS}f}"


def _format_accuracy(accuracy: float) -> str:
    return f"{accuracy:.2%}"


class Reporter:
    """Centralizes the console output produced during training and reporting."""

    @staticmethod
    def report(
        network: Network,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        Reporter.final_report(network)

        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets, dtype=float)
        predictions = network.forward(inputs)

        for input_row, target_row, prediction_row in zip(inputs, targets, predictions):
            Reporter.prediction(input_row, target_row, prediction_row)

        correct = count_correct(predictions, targets)
        print(f"\nAccuracy: {_format_accuracy(correct / len(inputs))} ({correct}/{len(inputs)} correct)")

    @staticmethod
    def epoch_summary(
        epoch: int,
        total_epochs: int,
        average_error: float,
        accuracy: float,
        validation_error: float | None = None,
        validation_accuracy: float | None = None,
    ) -> None:
        summary = (
            f"Epoch: {epoch}/{total_epochs} | "
            f"Error: {_format(average_error)} | "
            f"Accuracy: {_format_accuracy(accuracy)}"
        )

        if validation_error is not None and validation_accuracy is not None:
            summary += (
                f" | Val Error: {_format(validation_error)} | "
                f"Val Accuracy: {_format_accuracy(validation_accuracy)}"
            )

        print(summary)

    @staticmethod
    def _final_summary(message: str, **metrics) -> None:
        print(message)
        print("\nFinal Epoch:")
        Reporter.epoch_summary(**metrics)

    @staticmethod
    def training_completed(**metrics) -> None:
        Reporter._final_summary("Training completed.", **metrics)

    @staticmethod
    def max_epochs_reached(**metrics) -> None:
        Reporter._final_summary("Reached maximum epochs without meeting the error threshold.", **metrics)

    @staticmethod
    def evaluation(label: str, error: float, accuracy: float) -> None:
        print(f"{label} | Error: {_format(error)} | Accuracy: {_format_accuracy(accuracy)}")

    @staticmethod
    def final_report(network: Network) -> None:
        print()
        for index, layer in enumerate(network.layers):
            print(f"Layer {index} weights:\n{_format(layer.weights)}")
            print(f"Layer {index} bias:\n{_format(layer.bias)}")
            print()

    @staticmethod
    def prediction(
        input_row: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
    ) -> None:
        print(f"Inputs: {_format(input_row)} | Target: {_format(target)} | Prediction: {_format(prediction)}")

    @staticmethod
    def plot_saved(output_path: str) -> None:
        print(f"\nSaved error plot to {output_path}")
