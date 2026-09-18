import numpy as np

# AND
# x1  x2  |  target
#  0   0  |    0
#  0   1  |    0
#  1   0  |    0
#  1   1  |    1
AND_INPUTS = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
AND_TARGETS = np.array([0, 0, 0, 1])

# XOR
# x1  x2  |  target
#  0   0  |    0
#  0   1  |    1
#  1   0  |    1
#  1   1  |    0
XOR_INPUTS = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
XOR_TARGETS = np.array([0, 1, 1, 0])