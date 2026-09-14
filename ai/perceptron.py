import math
from collections.abc import Sequence


class Perceptron:
    def __init__(
        self,
        input_size: int,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
        learning_rate: float = 0.01,
        activation_function: str = "step",
        step_threshold: float = 0.0,
        gradient_descent: bool = False
    ):
        self.input_size = input_size
        self.bias = bias
        self.learning_rate = learning_rate
        self.step_threshold = step_threshold
        self.gradient_descent = gradient_descent
        self.activation_name = activation_function

        if weights is None:
            self.weights = [0.0] * input_size
        else:
            if len(weights) != input_size:
                raise ValueError(
                    f"weights has length {len(weights)}, expected {input_size}"
                )
           
            self.weights = list(weights)

        if activation_function == "step":
            self.activation_function = self.step_function
        elif activation_function == "sigmoid":
            self.activation_function = self.sigmoid_function
        else:
            raise ValueError(
                f"Unknown activation_function: {activation_function!r}. "
                "Expected 'step' or 'sigmoid'."
            )

        if self.gradient_descent and activation_function != "sigmoid":
            raise ValueError(
                "gradient_descent=True requires activation_function='sigmoid' "
                "(the step function has no usable derivative)."
            )

    def step_function(self, x: float) -> int:
        if x >= self.step_threshold:
            return 1
        else:
            return 0

    def sigmoid_function(self, x: float) -> float:
        # Guard against OverflowError from math.exp on very large |x|.
        if x < -60:
            return 0.0
        if x > 60:
            return 1.0
        
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x: float) -> float:
        # x here is expected to be a sigmoid *output* (i.e. already in (0, 1)),
        # so this is sigmoid'(net) expressed via the output, per prediction.
        # brings x closer to 0 or 1
        return x * (1 - x)

    def sum(self, inputs: Sequence[float]) -> float:
        if len(inputs) != self.input_size:
            raise ValueError(
                f"inputs has length {len(inputs)}, expected {self.input_size}"
            )

        total = 0.0
        for i in range(self.input_size):
            total += self.weights[i] * inputs[i]
        total += self.bias

        return total

    def predict(self, inputs: Sequence[float]) -> float:
        total = self.sum(inputs)
        return self.activation_function(total)

    def print_training_status(
        self,
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

    def update(self, inputs: Sequence[float], prediction: float, error: float) -> None:
        if self.gradient_descent:
            delta = self.learning_rate * error * self.sigmoid_derivative(prediction)
        else:
            delta = self.learning_rate * error

        for j in range(len(self.weights)):
            self.weights[j] += delta * inputs[j]

        self.bias += delta

    def train_by_epoch(
        self,
        training_data: Sequence[Sequence[float]],
        targets: Sequence[float],
        epochs: int = 20
    ) -> dict[int, float]:
        errors_by_epoch: dict[int, float] = {}

        for epoch in range(epochs):
            print("---")
            total_error = 0.0
            for inputs, target in zip(training_data, targets):
                prediction = self.predict(inputs)
                error = target - prediction
                total_error += abs(error)

                self.update(inputs, prediction, error)

                self.print_training_status(
                    current_epoch=epoch + 1,
                    total_epochs=epochs,
                    inputs=inputs,
                    weights=self.weights,
                    bias=self.bias,
                    target=target,
                    learning_rate=self.learning_rate,
                    error=error
                )

            errors_by_epoch[epoch + 1] = total_error / len(training_data)

        print("Training completed.")

        return errors_by_epoch

    def train_by_error(
        self,
        training_data: Sequence[Sequence[float]],
        targets: Sequence[float],
        error_threshold: float = 0.01,
        max_epochs: int = 999
    ) -> dict[int, float]:
        errors_by_epoch: dict[int, float] = {}

        epoch = 0
        while epoch < max_epochs:
            total_error = 0.0
            for inputs, target in zip(training_data, targets):
                prediction = self.predict(inputs)
                error = target - prediction
                total_error += abs(error)

                self.update(inputs, prediction, error)

            average_error = total_error / len(training_data)
            errors_by_epoch[epoch + 1] = average_error
            print(
                f"Epoch: {epoch + 1}/{max_epochs} | "
                f"Average Error: {average_error} | "
                f"Weights: {self.weights}"
            )

            if average_error < error_threshold:
                print("Training completed.")
                return errors_by_epoch

            epoch += 1

        print("Reached maximum epochs without meeting the error threshold.")
        return errors_by_epoch