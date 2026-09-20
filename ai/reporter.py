from .network import Network


class Reporter:
    """Centralizes the console output produced during training and reporting."""

    @staticmethod
    def report(network: Network, inputs: list[list[float]], targets: list[float]) -> None:
        Reporter.final_report(network)

        for input_row, target in zip(inputs, targets):
            prediction = network.forward(input_row)[0]
            Reporter.prediction(input_row, target, prediction)

    @staticmethod
    def epoch_summary(epoch: int, total_epochs: int, average_error: float) -> None:
        print(
            f"Epoch: {epoch}/{total_epochs} | "
            f"Average Error: {average_error}"
        )

    @staticmethod
    def training_completed(epoch: int, total_epochs: int, average_error: float) -> None:
        print("Training completed.")
        print("\nFinal Error:")
        Reporter.epoch_summary(
            epoch=epoch,
            total_epochs=total_epochs,
            average_error=average_error
        )

    @staticmethod
    def max_epochs_reached(epoch: int, total_epochs: int, average_error: float) -> None:
        print("Reached maximum epochs without meeting the error threshold.")
        print("\nFinal Error:")
        Reporter.epoch_summary(
            epoch=epoch,
            total_epochs=total_epochs,
            average_error=average_error
        )

    @staticmethod
    def final_report(network: Network) -> None:
        print()
        for index, layer in enumerate(network.layers):
            print(f"Layer {index} weights:\n{layer.weights}")
            print(f"Layer {index} bias:\n{layer.bias}")
            print()

    @staticmethod
    def prediction(input_row: list[float], target: float, prediction: float) -> None:
        print(f"Inputs: {input_row} | Target: {target} | Prediction: {prediction}")

    @staticmethod
    def plot_saved(output_path: str) -> None:
        print(f"\nSaved error plot to {output_path}")
