import functools
import os

from ai.activations import ACTIVATIONS
from ai.config import ERROR_THRESHOLD, LEARNING_RATE
from ai.layer import Layer
from ai.network import Network
from experiment_lab3_visualizer import OUTPUT_DIR, find_misclassified


class ExperimentReporter:
    """Builds a text report of a trained model's configuration and validation results."""

    def __init__(
        self,
        network: Network,
        errors_by_epoch: dict[int, float],
        val_inputs: list[list[float]],
        val_targets: list[float],
    ):
        self.network = network
        self.errors_by_epoch = errors_by_epoch
        self.val_inputs = val_inputs
        self.val_targets = val_targets

    @staticmethod
    def activation_name(layer: Layer) -> str:
        function = layer.nodes[0].activation_function
        # STEP is wrapped in functools.partial to bind its threshold
        if isinstance(function, functools.partial):
            function = function.func

        for activation in (ACTIVATIONS.STEP, ACTIVATIONS.SIGMOID, ACTIVATIONS.RELU):
            if activation.function is function:
                return activation.name
        return "unknown"

    @staticmethod
    def node_count(count: int) -> str:
        return f"{count} node" if count == 1 else f"{count} nodes"

    def structure_section(self) -> str:
        layers = self.network.layers
        lines = [
            "Perceptron:",
            f"  Input: {self.node_count(len(layers[0].nodes[0].weights))}",
        ]
        for index, layer in enumerate(layers, start=1):
            role = "output" if index == len(layers) else "hidden"
            lines.append(
                f"  Layer {index} ({role}, {self.activation_name(layer)}): "
                f"{self.node_count(len(layer.nodes))}"
            )
        return "\n".join(lines)

    def parameters_section(self) -> str:
        epochs_trained = len(self.errors_by_epoch)
        final_loss = self.errors_by_epoch[epochs_trained] if epochs_trained else float("nan")
        return "\n".join([
            f"Learning rate: {LEARNING_RATE}",
            f"Error threshold: {ERROR_THRESHOLD}",
            f"Epochs trained: {epochs_trained}",
            f"Average loss at end of training: {final_loss}",
        ])

    def weights_section(self) -> str:
        lines = []
        for index, layer in enumerate(self.network.layers, start=1):
            lines.append(f"Layer {index} weights:")
            for node_number, row in enumerate(layer.weights, start=1):
                lines.append(f"  Node {node_number}: {row}")
            lines.append(f"Layer {index} bias: {layer.bias}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def validation_section(self) -> str:
        total = len(self.val_inputs)
        misclassified = len(find_misclassified(self.network, self.val_inputs, self.val_targets))
        accuracy = (total - misclassified) / total * 100 if total else float("nan")
        return "\n".join([
            f"Misclassified / total: {misclassified} / {total}",
            f"Accuracy: {accuracy:.2f}%",
        ])

    def build_report(self) -> str:
        return "\n\n".join([
            "=== Model Structure ===\n" + self.structure_section(),
            "=== Training Configuration ===\n" + self.parameters_section(),
            "=== Final Weights and Bias ===\n" + self.weights_section(),
            "=== Validation Results ===\n" + self.validation_section(),
        ]) + "\n"

    def save(self, timestamp: str, folder_name: str = OUTPUT_DIR) -> str:
        os.makedirs(folder_name, exist_ok=True)
        output_path = os.path.join(folder_name, f"{timestamp}_results.txt")
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(self.build_report())
        return output_path
