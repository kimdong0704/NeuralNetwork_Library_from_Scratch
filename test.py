import os
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np

from ai.activations import ACTIVATIONS
from ai.config import ERROR_THRESHOLD
from ai.model import create_model
from ai.reporter import Reporter
from ai.trainer import Trainer
from ai.visualizer import Visualizer

from data import AND_INPUTS, AND_TARGETS, XOR_INPUTS, XOR_TARGETS

GATES = {
    "xor": (XOR_INPUTS, XOR_TARGETS),
    "and": (AND_INPUTS, AND_TARGETS),
}

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

OUTPUT_DIR = "tests"


def run_trial(
    gate: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    seed: int,
    trial_number: int,
    output_dir: str,
) -> bool:
    np.random.seed(seed)

    model = create_model()
    trainer = Trainer(model)
    trainer.train(training_data=inputs, targets=targets, report_each_epoch=False)

    Reporter.report(model, inputs, targets)

    filename = f"{trial_number}_{gate}_{ACTIVATIONS.SIGMOID.name}"
    title = f"{gate.upper()} Gate | {ACTIVATIONS.SIGMOID.name} activation | seed={seed}"

    output_path = Visualizer(
        filename,
        trainer.errors_by_epoch,
        title,
        folder_name=output_dir
    ).plot()

    Reporter.plot_saved(output_path)

    final_error = list(trainer.errors_by_epoch.values())[-1]
    return final_error < ERROR_THRESHOLD


def run_test(gate: str, seeds: list[int]) -> None:
    inputs, targets = GATES[gate]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    result_path = os.path.join(output_dir, f"{timestamp}_result.txt")

    with open(result_path, "w") as result_file, redirect_stdout(result_file):
        successful_trials = 0
        for trial_number, seed in enumerate(seeds, start=1):
            print(f"\n=== Trial {trial_number}/{len(seeds)} (seed={seed}) ===")
            if run_trial(gate, inputs, targets, seed, trial_number, output_dir):
                successful_trials += 1

        print(f"\nSuccessful trials: {successful_trials}/{len(seeds)}")
        print(f"\nSaved {len(seeds)} plots to {output_dir}")

    print(f"Results written to {result_path}")


if __name__ == "__main__":
    run_test(gate="xor", seeds=SEEDS)
