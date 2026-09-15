from collections.abc import Sequence

from .config import EPOCHS, ERROR_THRESHOLD, LEARNING_RATE, MAX_EPOCHS
from .network import NeuralNetwork
from .trainer import train_by_epoch, train_by_error
from visualizer import plot_errors

# activation_function is an architectural choice for the network being
# built here, not a training hyperparameter - so it lives here rather
# than as a config default, and add_layer requires it explicitly.
ACTIVATION_FUNCTION_FILENAME = "sigmoid"


def _print_report(
    network: NeuralNetwork,
    inputs: Sequence[Sequence[float]],
    targets: Sequence[float],
) -> None:
    node = network.layers[0].nodes[0]
    print("\nFinal weights:", node.weights.tolist())
    print("Final bias:", node.bias)

    for input_row, target in zip(inputs, targets):
        prediction = network.predict(input_row)[0]
        print(f"Inputs: {input_row} | Target: {target} | Prediction: {prediction}")


def _plot_graph(
    gate: str,
    train_method: str,
    errors_by_epoch: dict[int, float],
) -> None:
    filename = f"{gate}_{ACTIVATION_FUNCTION_FILENAME}_by-{train_method}"
    output_path = plot_errors(
        errors_by_epoch=errors_by_epoch,
        filename=filename,
        title=(
            f"{gate.upper()} Gate | {ACTIVATION_FUNCTION_FILENAME} activation | "
            f"train by {train_method}"
        ),
    )
    print(f"\nSaved error plot to {output_path}")


def run_model(
    gate: str,
    inputs: Sequence[Sequence[float]],
    targets: Sequence[float],
    train_method: str = "error",
) -> None:
    train_method = train_method.lower()

    if train_method not in ("error", "epoch"):
        raise ValueError(
            f"Unknown train_method: {train_method!r}. Expected 'error' or 'epoch'."
        )

    # building neural network
    network = NeuralNetwork()
    network.add_layer(input_size=2, output_size=1, activation_function="sigmoid")

    if train_method == "error":
        errors_by_epoch = train_by_error(
            network,
            training_data=inputs,
            targets=targets,
            learning_rate=LEARNING_RATE,
            error_threshold=ERROR_THRESHOLD,
            max_epochs=MAX_EPOCHS,
        )
    else:
        errors_by_epoch = train_by_epoch(
            network,
            training_data=inputs,
            targets=targets,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
        )

    _print_report(network, inputs, targets)
    _plot_graph(gate, train_method, errors_by_epoch)
