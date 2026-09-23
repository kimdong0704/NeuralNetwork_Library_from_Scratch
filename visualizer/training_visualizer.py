import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

from trainer.trainer import Trainer
from trainer_np.trainer import Trainer as NpTrainer

TRAIN_COLOR = "tab:blue"
VALIDATION_COLOR = "tab:orange"


class TrainingVisualizer:
    """Plots a trainer's loss and accuracy history across epochs (train, plus validation when recorded)."""

    def plot(self, trainer: Trainer | NpTrainer, title: str | None = None) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        self.plot_loss(trainer, ax=axes[0])
        self.plot_accuracy(trainer, ax=axes[1])

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        plt.show()

    def plot_loss(
        self,
        trainer: Trainer | NpTrainer,
        title: str = "Loss vs Epoch",
        ax: Axes | None = None,
    ) -> Axes:
        created_fig = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        self._draw_history(ax, trainer.errors_by_epoch, "train", TRAIN_COLOR)
        self._draw_history(ax, trainer.validation_errors_by_epoch, "validation", VALIDATION_COLOR)

        ax.set_ylabel("Loss")
        self._finish(ax, title)

        if created_fig:
            plt.show()

        return ax

    def plot_accuracy(
        self,
        trainer: Trainer | NpTrainer,
        title: str = "Accuracy vs Epoch",
        ax: Axes | None = None,
    ) -> Axes:
        created_fig = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        self._draw_history(ax, trainer.accuracy_by_epoch, "train", TRAIN_COLOR)
        self._draw_history(ax, trainer.validation_accuracy_by_epoch, "validation", VALIDATION_COLOR)

        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        self._finish(ax, title)

        if created_fig:
            plt.show()

        return ax

    @staticmethod
    def _draw_history(ax: Axes, history: dict[int, float], label: str, color: str) -> None:
        if not history:
            return

        ax.plot(list(history.keys()), list(history.values()), label=label, color=color, linewidth=1.5)

    @staticmethod
    def _finish(ax: Axes, title: str) -> None:
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
