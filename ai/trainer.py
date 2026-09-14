from collections.abc import Callable, Sequence

from .network import NeuralNetwork


def _require_single_node(network: NeuralNetwork):
    if not network.is_single_node:
        raise NotImplementedError(
            "Training only supports a single-layer, single-node network for "
            "now; backpropagation for hidden layers/multiple nodes is not "
            "implemented yet."
        )
    return network.layers[0].nodes[0]


def _print_sample_status(
    current_epoch: int,
    total_epochs: int,
    inputs: Sequence[float],
    weights: Sequence[float],
    bias: float,
    target: float,
    learning_rate: float,
    error: float
) -> None:
    print(
        f"Epoch: {current_epoch}/{total_epochs} | "
        f"Inputs: {inputs} | "
        f"Weights: {weights} | "
        f"Bias: {bias} | "
        f"Target: {target} | "
        f"Learning Rate: {learning_rate} | "
        f"Error: {error}"
    )


def _train(
    network: NeuralNetwork,
    training_data: Sequence[Sequence[float]],
    targets: Sequence[float],
    max_epochs: int,
    stop_condition: Callable[[int, float], bool],
    verbose_per_sample: bool,
) -> dict[int, float]:
    """One training loop shared by train_by_epoch/train_by_error.

    Runs epochs until `stop_condition(epoch, average_error)` is True or
    `max_epochs` is reached, whichever comes first.
    """
    node = _require_single_node(network)
    errors_by_epoch: dict[int, float] = {}

    epoch = 0
    while epoch < max_epochs:
        epoch += 1

        if verbose_per_sample:
            print("---")

        total_error = 0.0
        for inputs, target in zip(training_data, targets):
            prediction = node.predict(inputs)
            error = target - prediction
            total_error += abs(error)

            node.update(inputs, prediction, error)

            if verbose_per_sample:
                _print_sample_status(
                    current_epoch=epoch,
                    total_epochs=max_epochs,
                    inputs=inputs,
                    weights=node.weights.tolist(),
                    bias=node.bias,
                    target=target,
                    learning_rate=node.learning_rate,
                    error=error
                )

        average_error = total_error / len(training_data)
        errors_by_epoch[epoch] = average_error

        if not verbose_per_sample:
            print(
                f"Epoch: {epoch}/{max_epochs} | "
                f"Average Error: {average_error} | "
                f"Weights: {node.weights.tolist()}"
            )

        if stop_condition(epoch, average_error):
            print("Training completed.")
            return errors_by_epoch

    if verbose_per_sample:
        print("Training completed.")
    else:
        print("Reached maximum epochs without meeting the error threshold.")

    return errors_by_epoch


def train_by_epoch(
    network: NeuralNetwork,
    training_data: Sequence[Sequence[float]],
    targets: Sequence[float],
    epochs: int = 20
) -> dict[int, float]:
    return _train(
        network,
        training_data,
        targets,
        max_epochs=epochs,
        stop_condition=lambda epoch, average_error: epoch >= epochs,
        verbose_per_sample=True,
    )


def train_by_error(
    network: NeuralNetwork,
    training_data: Sequence[Sequence[float]],
    targets: Sequence[float],
    error_threshold: float = 0.01,
    max_epochs: int = 999
) -> dict[int, float]:
    return _train(
        network,
        training_data,
        targets,
        max_epochs=max_epochs,
        stop_condition=lambda epoch, average_error: average_error < error_threshold,
        verbose_per_sample=False,
    )
