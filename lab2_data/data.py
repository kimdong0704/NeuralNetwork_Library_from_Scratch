# AND
# x1  x2  |  target
#  0   0  |    0
#  0   1  |    0
#  1   0  |    0
#  1   1  |    1
AND_INPUTS: list[list[float]] = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
AND_TARGETS: list[float] = [0, 0, 0, 1]

# XOR
# x1  x2  |  target
#  0   0  |    0
#  0   1  |    1
#  1   0  |    1
#  1   1  |    0
XOR_INPUTS: list[list[float]] = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
XOR_TARGETS: list[float] = [0, 1, 1, 0]
