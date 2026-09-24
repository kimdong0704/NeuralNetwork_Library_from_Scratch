from .config import REPORT_DIGITS


def _format_error(error: float) -> str:
    return f"{error:.{REPORT_DIGITS}f}"


def _format_accuracy(accuracy: float) -> str:
    return f"{accuracy:.2%}"


class Reporter:
    """Centralizes the console output produced during training."""

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
            f"Error: {_format_error(average_error)} | "
            f"Accuracy: {_format_accuracy(accuracy)}"
        )

        if validation_error is not None and validation_accuracy is not None:
            summary += (
                f" | Val Error: {_format_error(validation_error)} | "
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
