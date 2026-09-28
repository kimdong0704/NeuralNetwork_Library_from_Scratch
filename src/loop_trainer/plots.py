import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

from loop_nn.config import Matrix, Vector
from loop_nn.network import Network

from .metrics import DECISION_THRESHOLD
from .reporter import EpochResult

TRAIN_COLOR = "#2a78d6"
VALIDATION_COLOR = "#e8743b"
CLASS_COLORS = ["#2a78d6", "#e8743b"]
REGION_COLORS = ["#dbe9ff", "#ffe3cf"]
MISCLASSIFIED_COLOR = "#d62728"
GRID_COLOR = "#e5e4e0"


def plot_history(history: list[EpochResult], title: str | None = None) -> None:
    """Loss and (for classifiers) accuracy per epoch, for the training data and, when recorded, the validation data."""
    epochs = [result.epoch for result in history]
    first = history[0]
    metrics = [("Loss", "loss", "validation_loss")]
    if first.accuracy is not None:
        metrics.append(("Accuracy", "accuracy", "validation_accuracy"))

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, (name, train_field, validation_field) in zip(axes[0], metrics):
        ax.plot(epochs, [getattr(result, train_field) for result in history], color=TRAIN_COLOR, label="train")
        if getattr(first, validation_field) is not None:
            ax.plot(
                epochs, [getattr(result, validation_field) for result in history],
                color=VALIDATION_COLOR, label="validation"
            )
        if name == "Accuracy":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        ax.set_title(f"{name} per epoch", loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def _axis_values(values: Vector, steps: int, padding: float) -> Vector:
    low, high = min(values) - padding, max(values) + padding
    return [low + (high - low) * index / (steps - 1) for index in range(steps)]

def plot_decision_boundary(
    network: Network,
    inputs: Matrix,
    targets: Matrix,
    title: str | None = None,
    ax: Axes | None = None,
    steps: int = 200,
    padding: float = 0.5
) -> Axes:
    """Colors the 2D input plane by the predicted class of a single-output network and circles misclassified points in red."""
    classes = [target[0] >= DECISION_THRESHOLD for target in targets]

    created_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    x1 = _axis_values([row[0] for row in inputs], steps, padding)
    x2 = _axis_values([row[1] for row in inputs], steps, padding)
    grid = [[a, b] for b in x2 for a in x1]
    flat_outputs = [output[0] for output in network.forward(grid)]
    outputs = [flat_outputs[row * steps:(row + 1) * steps] for row in range(steps)]

    lowest, highest = min(flat_outputs), max(flat_outputs)
    levels = [min(lowest, 0.0), DECISION_THRESHOLD, max(highest, 1.0)]
    ax.contourf(x1, x2, outputs, levels=levels, colors=REGION_COLORS, alpha=0.6)
    if lowest < DECISION_THRESHOLD <= highest:
        ax.contour(x1, x2, outputs, levels=[DECISION_THRESHOLD], colors="black", linewidths=1.5)

    # the 4-point logic gates get big markers, larger datasets small ones
    size = 160 if len(inputs) <= 16 else 30
    ax.scatter(
        [row[0] for row in inputs], [row[1] for row in inputs], s=size,
        c=[CLASS_COLORS[int(label)] for label in classes], edgecolors="black", linewidths=0.5
    )

    wrong = []
    for row, prediction, label in zip(inputs, network.forward(inputs), classes):
        if (prediction[0] >= DECISION_THRESHOLD) != label:
            wrong.append(row)
    ax.scatter(
        [row[0] for row in wrong], [row[1] for row in wrong], s=size * 3, facecolors="none",
        edgecolors=MISCLASSIFIED_COLOR, linewidths=1.5
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if title:
        ax.set_title(title, loc="left")

    if created_figure:
        plt.show()

    return ax
