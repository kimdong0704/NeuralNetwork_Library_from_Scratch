import numpy as np

from .network import Network


class Reporter:
    """Centralizes the console output produced during training and reporting."""

    @staticmethod
    def report(network: Network, inputs: np.ndarray, targets: np.ndarray) -> None:
        layer = network.layers[0]
        Reporter.final_report(layer.weights[0], layer.bias[0])

        for input_row, target in zip(inputs, targets):
            prediction = network.forward(input_row)[0]
            Reporter.prediction(input_row, target, prediction)

    @staticmethod
    def epoch_summary(epoch: int, total_epochs: int, average_error: float, weights: np.ndarray) -> None:
        print(
            f"Epoch: {epoch}/{total_epochs} | "
            f"Average Error: {average_error} | "
            f"Weights: {weights}"
        )

    @staticmethod
    def training_completed() -> None:
        print("Training completed.")

    @staticmethod
    def max_epochs_reached() -> None:
        print("Reached maximum epochs without meeting the error threshold.")

    @staticmethod
    def final_report(weights: np.ndarray, bias: float) -> None:
        print("\nFinal weights:", weights)
        print("Final bias:", bias)

    @staticmethod
    def prediction(input_row: np.ndarray, target: float, prediction: float) -> None:
        print(f"Inputs: {input_row} | Target: {target} | Prediction: {prediction}")

    @staticmethod
    def plot_saved(output_path: str) -> None:
        print(f"\nSaved error plot to {output_path}")
