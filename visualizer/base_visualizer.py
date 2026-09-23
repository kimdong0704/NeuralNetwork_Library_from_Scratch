import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from neural_network.network import Network

from .config import DECISION_THRESHOLD, GRID_PADDING, GRID_STEPS
from .grid import Grid, build_grid, predict_grid


class DecisionBoundaryVisualizer:
    """Base class for plotting a 2D (x1, x2) input space alongside a model's decision boundary."""

    grid_steps: int = GRID_STEPS
    grid_padding: float = GRID_PADDING
    decision_threshold: float = DECISION_THRESHOLD

    def plot(
        self,
        network: Network,
        inputs: list[list[float]],
        targets: list[list[float]],
        title: str | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        created_fig = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        grid = build_grid(inputs, padding=self.grid_padding, steps=self.grid_steps)
        outputs = predict_grid(network, grid)

        self._draw_regions(ax, grid, outputs)
        self._draw_points(ax, network, inputs, targets)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        if title:
            ax.set_title(title)

        if created_fig:
            plt.show()

        return ax

    def _draw_regions(self, ax: Axes, grid: Grid, outputs: list[list[float]]) -> None:
        raise NotImplementedError

    def _draw_points(
        self,
        ax: Axes,
        network: Network,
        inputs: list[list[float]],
        targets: list[list[float]],
    ) -> None:
        raise NotImplementedError
