from loop_nn.config import Matrix, Vector


# A single output at/above this is class 1, otherwise class 0
DECISION_THRESHOLD = 0.5


def _largest(values: Vector) -> int:
    return values.index(max(values))

def count_correct(predicted: Matrix, targets: Matrix) -> int:
    # single output: compare which side of the threshold each falls on; multiple outputs: compare the largest
    correct = 0
    for prediction, target in zip(predicted, targets):
        if len(prediction) == 1:
            correct += (prediction[0] >= DECISION_THRESHOLD) == (target[0] >= DECISION_THRESHOLD)
        else:
            correct += _largest(prediction) == _largest(target)
    return int(correct)

def accuracy(predicted: Matrix, targets: Matrix) -> float:
    return count_correct(predicted, targets) / len(predicted)
