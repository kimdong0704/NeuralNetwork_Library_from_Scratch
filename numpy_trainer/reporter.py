from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    loss: float
    accuracy: float
    validation_loss: float | None = None
    validation_accuracy: float | None = None

    def __str__(self) -> str:
        summary = f"epoch {self.epoch:>4} | loss: {self.loss:.4f} | accuracy: {self.accuracy:.2%}"
        if self.validation_loss is not None:
            summary += f" | val loss: {self.validation_loss:.4f} | val accuracy: {self.validation_accuracy:.2%}"
        return summary


class Reporter:
    """Keeps the per-epoch history of training and prints its progress."""

    def __init__(self):
        self.history: list[EpochResult] = []

    def record(self, result: EpochResult, print_interval: int) -> None:
        self.history.append(result)

        if print_interval and result.epoch % print_interval == 0:
            print(result)

    def finish(self, stopped_early: bool, print_interval: int) -> None:
        if not print_interval:
            return

        last = self.history[-1]
        if last.epoch % print_interval:
            print(last)
        if stopped_early:
            print(f"stopping condition reached at epoch {last.epoch}")


def _format(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"

def print_predictions(inputs: Sequence, targets: Sequence, predicted: Sequence) -> None:
    for row, target, prediction in zip(inputs, targets, predicted):
        print(f"inputs: {_format(row)} | target: {_format(target)} | prediction: {_format(prediction)}")
