import numpy as np

from neural_network_np.network import Network


def _to_list(value):
    return value.tolist() if isinstance(value, np.ndarray) else value


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

    @staticmethod
    def epoch_summary(epoch: int, total_epochs: int, average_error: float) -> None:
        print(
            f"Epoch: {epoch}/{total_epochs} | "
            f"Average Error: {average_error}"
        )

    @staticmethod
    def _final_error(message: str, epoch: int, total_epochs: int, average_error: float) -> None:
        print(message)
        print("\nFinal Error:")
        Reporter.epoch_summary(
            epoch=epoch,
            total_epochs=total_epochs,
            average_error=average_error
        )

    @staticmethod
    def training_completed(epoch: int, total_epochs: int, average_error: float) -> None:
        Reporter._final_error("Training completed.", epoch, total_epochs, average_error)

    @staticmethod
    def max_epochs_reached(epoch: int, total_epochs: int, average_error: float) -> None:
        Reporter._final_error(
            "Reached maximum epochs without meeting the error threshold.",
            epoch, total_epochs, average_error
        )

    @staticmethod
    def final_report(network: Network) -> None:
        print()
        for index, layer in enumerate(network.layers):
            print(f"Layer {index} weights:\n{_to_list(layer.weights)}")
            print(f"Layer {index} bias:\n{_to_list(layer.bias)}")
            print()

    @staticmethod
    def prediction(
        input_row: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
    ) -> None:
        print(f"Inputs: {_to_list(input_row)} | Target: {_to_list(target)} | Prediction: {_to_list(prediction)}")

    @staticmethod
    def plot_saved(output_path: str) -> None:
        print(f"\nSaved error plot to {output_path}")
