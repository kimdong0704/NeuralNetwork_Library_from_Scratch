import csv
import os
import random

from ai.model import create_model
from ai.network import Network
from ai.reporter import Reporter
from ai.trainer import Trainer
from experiment_lab3_reporter import ExperimentReporter
from experiment_lab3_visualizer import OUTPUT_DIR, make_timestamp, plot_predictions

TRAIN_PATH = "lab3_data/two_moons_train.csv"
VAL_PATH = "lab3_data/two_moons_val.csv"

SEED = 1


def load_two_moons(path: str) -> tuple[list[list[float]], list[float]]:
    inputs: list[list[float]] = []
    targets: list[float] = []
    with open(path, newline="") as file:
        for x1, x2, y in csv.reader(file):
            inputs.append([float(x1), float(x2)])
            targets.append(float(y))
    return inputs, targets


def load_datasets() -> tuple[tuple[list[list[float]], list[float]], tuple[list[list[float]], list[float]]]:
    train = load_two_moons(TRAIN_PATH)
    val = load_two_moons(VAL_PATH)
    return train, val


def train_model(inputs: list[list[float]], targets: list[float]) -> tuple[Network, dict[int, float]]:
    random.seed(SEED)

    model = create_model()
    trainer = Trainer(model)
    trainer.train(training_data=inputs, targets=targets)
    Reporter.final_report(model)

    return model, trainer.errors_by_epoch


def plot_split(
    model: Network,
    inputs: list[list[float]],
    targets: list[float],
    split: str,
    timestamp: str,
    folder_name: str,
) -> str:
    return plot_predictions(
        model,
        inputs,
        targets,
        title=f"Two Moons | {split}",
        filename=f"two_moons_{split}",
        folder_name=folder_name,
        show=True,
        timestamp=timestamp,
    )


def main() -> None:
    (train_inputs, train_targets), (val_inputs, val_targets) = load_datasets()
    print(f"Loaded {len(train_inputs)} train samples and {len(val_inputs)} validation samples")

    model, errors_by_epoch = train_model(train_inputs, train_targets)

    timestamp = make_timestamp()
    run_folder = os.path.join(OUTPUT_DIR, timestamp)
    report_path = ExperimentReporter(model, errors_by_epoch, val_inputs, val_targets).save(timestamp, run_folder)
    print(f"\nSaved results report to {report_path}")

    # plt.show() blocks, so the report is saved before the plot is displayed
    Reporter.plot_saved(plot_split(model, val_inputs, val_targets, "val", timestamp, run_folder))


if __name__ == "__main__":
    main()
