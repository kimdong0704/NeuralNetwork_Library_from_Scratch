import numpy as np

from numpy_nn.config import DTYPE


class DataLoader:
    """Splits inputs and targets into batches; loop over it with `for inputs, targets in loader`."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool):
        if len(inputs) != len(targets):
            raise ValueError(f"{len(inputs)} inputs but {len(targets)} targets")

        self.inputs = np.asarray(inputs, dtype=DTYPE)
        self.targets = np.asarray(targets, dtype=DTYPE)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.inputs))
        if self.shuffle:
            np.random.shuffle(indices)

        return self._batches(indices)

    def _batches(self, indices: np.ndarray):
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield self.inputs[batch], self.targets[batch]
