from .config import DECISION_THRESHOLD


def is_correct(prediction: list[float], target: list[float]) -> bool:
    # single output: compare which side of the threshold each falls on; multiple outputs: compare the largest
    if len(prediction) == 1:
        return (prediction[0] >= DECISION_THRESHOLD) == (target[0] >= DECISION_THRESHOLD)

    return prediction.index(max(prediction)) == target.index(max(target))
