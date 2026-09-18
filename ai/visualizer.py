import os
import re
from datetime import datetime

import matplotlib.pyplot as plt


class Visualizer:
    OUTPUT_DIR = "plot"
    DATETIME_TAG_PATTERN = re.compile(r"\d{8}_\d{6}")

    def __init__(
        self,
        filename: str,
        errors_by_epoch: dict[int, float],
        title: str = "Training Error by Epoch",
        folder_name: str | None = None,
    ):
        self.filename = filename
        self.errors_by_epoch = errors_by_epoch
        self.title = title
        self.folder_name = folder_name or self.OUTPUT_DIR

    def plot(self) -> str:
        epochs = list(self.errors_by_epoch.keys())
        errors = list(self.errors_by_epoch.values())

        fig, ax = plt.subplots()
        ax.plot(epochs, errors, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.set_title(self.title)
        ax.grid(True)

        os.makedirs(self.folder_name, exist_ok=True)

        if self.DATETIME_TAG_PATTERN.search(self.folder_name):
            name = f"{self.filename}.png"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{self.filename}_{timestamp}.png"

        output_path = os.path.join(self.folder_name, name)

        fig.savefig(output_path)
        plt.close(fig)

        return output_path
