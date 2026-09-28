import random

from loop_nn.config import Matrix


class DataLoader:
    """Splits inputs and targets into batches; loop over it with `for inputs, targets in loader`."""

    def __init__(self, inputs: Matrix, targets: Matrix, batch_size: int, shuffle: bool):
        if len(inputs) != len(targets):
            raise ValueError(f"{len(inputs)} inputs but {len(targets)} targets")

        self.inputs = [[float(value) for value in row] for row in inputs]
        self.targets = [[float(value) for value in row] for row in targets]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.inputs)))
        if self.shuffle:
            random.shuffle(indices)

        return self._batches(indices)

    def _batches(self, indices: list[int]):
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield [self.inputs[index] for index in batch], [self.targets[index] for index in batch]
