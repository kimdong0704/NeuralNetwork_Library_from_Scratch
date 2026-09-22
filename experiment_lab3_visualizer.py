import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ai.config import STEP_THRESHOLD
from ai.network import Network

CLASS_COLORS = {1: "tab:orange", 0: "tab:blue"}
CLASS_LABELS = {1: "y = 1 (orange)", 0: "y = 0 (blue)"}

OUTPUT_DIR = "lab3_results"

GRID_STEPS = 100
GRID_PADDING = 0.5


def draw_points(ax: Axes, inputs: list[list[float]], targets: list[float]) -> None:
    colors = [CLASS_COLORS[int(target)] for target in targets]
    x1 = [point[0] for point in inputs]
    x2 = [point[1] for point in inputs]
    ax.scatter(x1, x2, c=colors, edgecolors="black", linewidths=0.5)


def style_axes(ax: Axes, title: str) -> None:
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(True)


def add_legend(ax: Axes, total: int | None = None, misclassified_count: int | None = None) -> None:
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=CLASS_COLORS[label], label=CLASS_LABELS[label])
        for label in (1, 0)
    ]
    if misclassified_count is not None:
        handles.append(Line2D([0], [0], color="black", label="decision boundary"))
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor="red", markersize=10,
                   label=f"misclassified: {misclassified_count}/{total}")
        )
    ax.legend(handles=handles, title="Class", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)


def find_misclassified(
    network: Network,
    inputs: list[list[float]],
    targets: list[float],
) -> list[list[float]]:
    misclassified = []
    for point, target in zip(inputs, targets):
        predicted_class = 1 if network.forward(point)[0] >= STEP_THRESHOLD else 0
        if predicted_class != int(target):
            misclassified.append(point)
    return misclassified


def draw_decision_boundary(ax: Axes, network: Network, inputs: list[list[float]]) -> None:
    x1_values = [point[0] for point in inputs]
    x2_values = [point[1] for point in inputs]
    x1_min, x1_max = min(x1_values) - GRID_PADDING, max(x1_values) + GRID_PADDING
    x2_min, x2_max = min(x2_values) - GRID_PADDING, max(x2_values) + GRID_PADDING

    x1_grid = [x1_min + (x1_max - x1_min) * i / (GRID_STEPS - 1) for i in range(GRID_STEPS)]
    x2_grid = [x2_min + (x2_max - x2_min) * i / (GRID_STEPS - 1) for i in range(GRID_STEPS)]
    outputs = [[network.forward([x1, x2])[0] for x1 in x1_grid] for x2 in x2_grid]

    ax.contour(x1_grid, x2_grid, outputs, levels=[STEP_THRESHOLD], colors="black")


def draw_misclassified(ax: Axes, points: list[list[float]]) -> None:
    if not points:
        return
    x1 = [point[0] for point in points]
    x2 = [point[1] for point in points]
    ax.scatter(x1, x2, s=120, facecolors="none", edgecolors="red", linewidths=1.5)


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_figure(fig: Figure, filename: str, folder_name: str, timestamp: str | None = None) -> str:
    os.makedirs(folder_name, exist_ok=True)
    timestamp = timestamp or make_timestamp()
    output_path = os.path.join(folder_name, f"{filename}_{timestamp}.png")
    fig.savefig(output_path, bbox_inches="tight")
    return output_path


def plot_points(
    inputs: list[list[float]],
    targets: list[float],
    title: str = "Two Moons",
    filename: str = "two_moons",
    folder_name: str = OUTPUT_DIR,
    show: bool = False,
) -> str:
    fig, ax = plt.subplots()
    draw_points(ax, inputs, targets)
    style_axes(ax, title)
    add_legend(ax)

    output_path = save_figure(fig, filename, folder_name)

    if show:
        plt.show()
    plt.close(fig)

    return output_path


def plot_predictions(
    network: Network,
    inputs: list[list[float]],
    targets: list[float],
    title: str = "Two Moons",
    filename: str = "two_moons_predictions",
    folder_name: str = OUTPUT_DIR,
    show: bool = False,
    timestamp: str | None = None,
) -> str:
    fig, ax = plt.subplots(layout="constrained")
    draw_points(ax, inputs, targets)
    draw_decision_boundary(ax, network, inputs)
    misclassified = find_misclassified(network, inputs, targets)
    draw_misclassified(ax, misclassified)
    style_axes(ax, title)
    add_legend(ax, total=len(inputs), misclassified_count=len(misclassified))

    output_path = save_figure(fig, filename, folder_name, timestamp)

    if show:
        plt.show()
    plt.close(fig)

    return output_path
