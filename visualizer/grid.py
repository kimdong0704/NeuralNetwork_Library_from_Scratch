from dataclasses import dataclass

from neural_network.network import Network


@dataclass
class Grid:
    x1_values: list[float]
    x2_values: list[float]


def build_grid(inputs: list[list[float]], padding: float, steps: int) -> Grid:
    x1 = [point[0] for point in inputs]
    x2 = [point[1] for point in inputs]

    x1_min, x1_max = min(x1) - padding, max(x1) + padding
    x2_min, x2_max = min(x2) - padding, max(x2) + padding

    x1_values = [x1_min + (x1_max - x1_min) * i / (steps - 1) for i in range(steps)]
    x2_values = [x2_min + (x2_max - x2_min) * i / (steps - 1) for i in range(steps)]

    return Grid(x1_values=x1_values, x2_values=x2_values)


def predict_grid(network: Network, grid: Grid) -> list[list[float]]:
    return [
        [network.forward([x1, x2])[0] for x1 in grid.x1_values]
        for x2 in grid.x2_values
    ]
