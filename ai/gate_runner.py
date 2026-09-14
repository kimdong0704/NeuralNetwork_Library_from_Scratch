from collections.abc import Sequence

from .layer import Layer
from .network import NeuralNetwork
from .node import Node
from .trainer import train_by_epoch, train_by_error
from .visualizer import plot_errors

# Shared by and_test.py / xor_test.py so each stays a small, gate-specific
# config instead of duplicating the network/train/report pipeline.


def run_gate(
    gate: str,
    inputs: Sequence[Sequence[float]],
    targets: Sequence[float],
    train_method: str = "error",
    activation_function: str = "step",
    learning_rate: float = 0.01,
    weights: Sequence[float] | None = None,
    bias: float = 0.0,
    gradient_descent: bool = False,
    step_threshold: float = 0.0,
    error_threshold: float = 0.1,
    max_epochs: int = 500,
    epochs: int = 20,
) -> None:
    train_method = train_method.lower()

    if train_method not in ("error", "epoch"):
        raise ValueError(
            f"Unknown train_method: {train_method!r}. Expected 'error' or 'epoch'."
        )

    network = NeuralNetwork([
        Layer([
            Node(
                input_size=2,
                weights=weights,
                bias=bias,
                learning_rate=learning_rate,
                activation_function=activation_function,
                step_threshold=step_threshold,
                gradient_descent=gradient_descent,
            )
        ])
    ])
    node = network.layers[0].nodes[0]

    if train_method == "error":
        errors_by_epoch = train_by_error(
            network,
            training_data=inputs,
            targets=targets,
            error_threshold=error_threshold,
            max_epochs=max_epochs,
        )
    else:
        errors_by_epoch = train_by_epoch(
            network,
            training_data=inputs,
            targets=targets,
            epochs=epochs,
        )

    print("\nFinal weights:", node.weights.tolist())
    print("Final bias:", node.bias)

    for input_row, target in zip(inputs, targets):
        prediction = network.predict(input_row)[0]
        print(f"Inputs: {input_row} | Target: {target} | Prediction: {prediction}")

    filename = f"{gate}_{activation_function}_by-{train_method}"
    output_path = plot_errors(
        errors_by_epoch=errors_by_epoch,
        filename=filename,
        title=(
            f"{gate.upper()} Gate | {activation_function} activation | "
            f"train by {train_method}"
        ),
    )
    print(f"\nSaved error plot to {output_path}")
