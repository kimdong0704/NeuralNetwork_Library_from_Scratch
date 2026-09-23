from matplotlib.axes import Axes

from neural_network.network import Network

from .base_visualizer import DecisionBoundaryVisualizer
from .grid import Grid
from .plotting import circle_points, find_misclassified, scatter_by_class

REGION_COLORS = ["#dbe9ff", "#ffe3cf"]


class MoonsVisualizer(DecisionBoundaryVisualizer):
    """Plots two-moons data with the model's decision boundary dividing the two crescents."""

    grid_padding = 0.5

    def _draw_regions(self, ax: Axes, grid: Grid, outputs: list[list[float]]) -> None:
        ax.contourf(
            grid.x1_values,
            grid.x2_values,
            outputs,
            levels=[0, self.decision_threshold, 1],
            colors=REGION_COLORS,
            alpha=0.4,
        )
        ax.contour(
            grid.x1_values,
            grid.x2_values,
            outputs,
            levels=[self.decision_threshold],
            colors="black",
            linewidths=1.5,
        )

    def _draw_points(
        self,
        ax: Axes,
        network: Network,
        inputs: list[list[float]],
        targets: list[list[float]],
    ) -> None:
        scatter_by_class(ax, inputs, targets, s=40)

        misclassified = find_misclassified(network, inputs, targets, self.decision_threshold)
        circle_points(ax, misclassified)
