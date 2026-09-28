import numpy as np


# A single output at/above this is class 1, otherwise class 0
DECISION_THRESHOLD = 0.5


def count_correct(predicted: np.ndarray, targets: np.ndarray) -> int:
    # single output: compare which side of the threshold each falls on; multiple outputs: compare the largest
    if predicted.shape[1] == 1:
        return int(np.sum((predicted >= DECISION_THRESHOLD) == (targets >= DECISION_THRESHOLD)))

    return int(np.sum(np.argmax(predicted, axis=1) == np.argmax(targets, axis=1)))

def accuracy(predicted: np.ndarray, targets: np.ndarray) -> float:
    return count_correct(predicted, targets) / len(predicted)
