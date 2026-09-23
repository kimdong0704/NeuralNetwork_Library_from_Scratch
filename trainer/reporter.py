from neural_network.network import Network

from .config import REPORT_DIGITS
from .metrics import is_correct


def _format(value: float | list) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.append(_format(item))
        return "[" + ", ".join(parts) + "]"

    return f"{value:.{REPORT_DIGITS}f}"


def _format_accuracy(accuracy: float) -> str:
    return f"{accuracy:.2%}"


class Reporter:
    """Centralizes the console output produced during training and reporting."""

    @staticmethod
    def report(
        network: Network,
        inputs: list[list[float]],
        targets: list[list[float]],
    ) -> None:
        Reporter.final_report(network)

        correct = 0
        for input_row, target in zip(inputs, targets):
            prediction = network.forward(input_row)
            Reporter.prediction(input_row, target, prediction)
            if is_correct(prediction, target):
                correct += 1

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
        input_row: list[float],
        target: list[float],
        prediction: list[float],
    ) -> None:
        print(f"Inputs: {_format(input_row)} | Target: {_format(target)} | Prediction: {_format(prediction)}")

    @staticmethod
    def plot_saved(output_path: str) -> None:
        print(f"\nSaved error plot to {output_path}")
