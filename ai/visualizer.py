import os
from datetime import datetime

import matplotlib.pyplot as plt


def plot_errors(
    errors_by_epoch: dict[int, float],
    filename: str,
    output_dir: str = "plot",
    title: str = "Training Error by Epoch",
) -> str:
    """Plot an epoch -> error/loss mapping and save it to disk.

    Takes just the dictionary (as returned by Perceptron.train_by_epoch /
    train_by_error) so this module has no dependency on the Perceptron class.

    Saves to "{output_dir}/{filename}_{timestamp}.png" and returns that path.
    """
    epochs = list(errors_by_epoch.keys())
    errors = list(errors_by_epoch.values())

    fig, ax = plt.subplots()
    ax.plot(epochs, errors, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.grid(True)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{filename}_{timestamp}.png")

    fig.savefig(output_path)
    plt.close(fig)

    return output_path
