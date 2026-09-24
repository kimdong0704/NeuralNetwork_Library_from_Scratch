import numpy as np

# Threshold of the STEP activation function that determines x >= t --> 1, otherwise 0
STEP_THRESHOLD = 0.5

# Floating point type of every weight, bias and input; np.float32 roughly halves the time of each
# matrix multiplication at the cost of precision, np.float64 keeps results identical to earlier runs
DTYPE = np.float64
