from matplotlib.axes import Axes

from neural_network.network import Network

from .config import DECISION_THRESHOLD

CLASS_COLORS = {0: "tab:blue", 1: "tab:orange"}


def class_of(target: list[float]) -> int:
    return int(round(target[0]))


def scatter_by_class(
    ax: Axes,
    inputs: list[list[float]],
    targets: list[list[float]],
    **scatter_kwargs,
) -> None:
    colors = [CLASS_COLORS[class_of(target)] for target in targets]
    ax.scatter(
        [point[0] for point in inputs],
        [point[1] for point in inputs],
        c=colors,
        edgecolors="black",
        linewidths=0.5,
        **scatter_kwargs,
    )


def find_misclassified(
    network: Network,
    inputs: list[list[float]],
    targets: list[list[float]],
    threshold: float = DECISION_THRESHOLD,
) -> list[list[float]]:
    misclassified = []
    for point, target in zip(inputs, targets):
        predicted_class = 1 if network.forward(point)[0] >= threshold else 0
        if predicted_class != class_of(target):
            misclassified.append(point)
    return misclassified


def circle_points(ax: Axes, points: list[list[float]], **scatter_kwargs) -> None:
    if not points:
        return

    ax.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        s=120,
        **scatter_kwargs,
    )
