import numpy as np

from ai.activations import ACTIVATIONS
from ai.model import create_model
from ai.reporter import Reporter
from ai.trainer import Trainer
from ai.visualizer import Visualizer
from data import AND_INPUTS, AND_TARGETS, XOR_INPUTS, XOR_TARGETS


def run_experiment(
    gate: str,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> None:
    model = create_model()

    trainer = Trainer(model)

    trainer.train(
        training_data=inputs,
        targets=targets,
    )

    Reporter.report(model, inputs, targets)

    filename = f"{gate}_{ACTIVATIONS.SIGMOID.name}"
    title = f"{gate.upper()} Gate | {ACTIVATIONS.SIGMOID.name} activation"
    
    output_path = Visualizer(filename, trainer.errors_by_epoch, title).plot()
    Reporter.plot_saved(output_path)


if __name__ == "__main__":
    run_experiment(gate="xor", inputs=XOR_INPUTS, targets=XOR_TARGETS)
