import os
from datetime import datetime

import matplotlib.pyplot as plt


class Visualizer:
    """Plots an epoch -> error/loss mapping and saves it to the plot directory."""

    OUTPUT_DIR = "plot"

    def __init__(
        self,
        filename: str,
        errors_by_epoch: dict[int, float],
        title: str = "Training Error by Epoch",
    ):
        self.filename = filename
        self.errors_by_epoch = errors_by_epoch
        self.title = title

    def plot(self) -> str:
        epochs = list(self.errors_by_epoch.keys())
        errors = list(self.errors_by_epoch.values())

        fig, ax = plt.subplots()
        ax.plot(epochs, errors, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.set_title(self.title)
        ax.grid(True)

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.OUTPUT_DIR, f"{self.filename}_{timestamp}.png")

        fig.savefig(output_path)
        plt.close(fig)

        return output_path
