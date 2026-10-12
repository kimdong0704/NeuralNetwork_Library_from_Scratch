import numpy as np

from numpy_utils.config import DTYPE

class Pad:
    def __init__(self, amount: int):
        if amount < 0:
            raise ValueError("Padding amount must be non-negative.")
        
        self.amount = amount

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input = np.array(inputs, dtype=DTYPE)
        
        border = (self.amount, self.amount)
        self.last_output = np.pad(self.last_input, ((0, 0), border, border), constant_values=0)
        
        return self.last_output
