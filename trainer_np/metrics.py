import numpy as np

from .config import DECISION_THRESHOLD


def count_correct(predictions: np.ndarray, targets: np.ndarray) -> int:
    # single output: compare which side of the threshold each falls on; multiple outputs: compare the largest
    if predictions.shape[1] == 1:
        return int(np.sum((predictions >= DECISION_THRESHOLD) == (targets >= DECISION_THRESHOLD)))

    return int(np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)))
