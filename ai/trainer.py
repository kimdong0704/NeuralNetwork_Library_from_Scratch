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
    learning_rate: float,
    stop_condition: Callable[[int, float], bool]
) -> dict[int, float]:
    node = _require_single_node(network)
    errors_by_epoch: dict[int, float] = {}

    epoch = 0
    while epoch < max_epochs:
        epoch += 1

        print("---")

        total_error = 0.0
        for inputs, target in zip(training_data, targets):
            prediction = node.predict(inputs)
            error = target - prediction
            total_error += abs(error)

            # single node
            node.update(inputs, prediction, error, learning_rate)

            _print_sample_status(
                current_epoch=epoch,
                total_epochs=max_epochs,
                inputs=inputs,
                weights=node.weights.tolist(),
                bias=node.bias,
                target=target,
                learning_rate=learning_rate,
                error=error
            )

        average_error = total_error / len(training_data)
        errors_by_epoch[epoch] = average_error

        print(
            f"Epoch: {epoch}/{max_epochs} | "
            f"Average Error: {average_error} | "
            f"Weights: {node.weights.tolist()}"
        )

        if stop_condition(epoch, average_error):
            print("Training completed.")
            return errors_by_epoch

    print("Reached maximum epochs without meeting the error threshold.")
    return errors_by_epoch


def train_by_epoch(
    network: NeuralNetwork,
    training_data: Sequence[Sequence[float]],
    targets: Sequence[float],
    learning_rate: float,
    epochs: int
) -> dict[int, float]:
    return _train(
        network,
        training_data,
        targets,
        max_epochs=epochs,
        learning_rate=learning_rate,
        stop_condition=lambda epoch, average_error: epoch >= epochs,
    )


def train_by_error(
    network: NeuralNetwork,
    training_data: Sequence[Sequence[float]],
    targets: Sequence[float],
    learning_rate: float,
    error_threshold: float,
    max_epochs: int,
) -> dict[int, float]:
    return _train(
        network,
        training_data,
        targets,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        stop_condition=lambda epoch, average_error: average_error < error_threshold
    )
