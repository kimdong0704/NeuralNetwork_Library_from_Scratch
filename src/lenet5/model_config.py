from cnn.utils import window_count

# --- Input ---
IMAGE_SIZE = 28
INPUT_CHANNELS = 1
PADDING = 2
INPUT_SIZE = IMAGE_SIZE + 2 * PADDING

# --- Conv settings ---
KERNEL_SIZE = 5
CONV_STRIDE = 1

# --- Pool settings ---
POOL_SIZE = 2
POOL_STRIDE = 2

# --- Channels per layer ---
C1_MAPS = 6
C3_MAPS = 16
C5_MAPS = 120
F6_UNITS = 84
OUTPUT_UNITS = 10

# --- Weight initialization ---
WEIGHT_LOW = -0.1
WEIGHT_HIGH = 0.1


# --- Channel sizes ---
C1_SIZE = window_count(INPUT_SIZE, KERNEL_SIZE, CONV_STRIDE)    # 28
S2_SIZE = window_count(C1_SIZE, POOL_SIZE, POOL_STRIDE)         # 14
C3_SIZE = window_count(S2_SIZE, KERNEL_SIZE, CONV_STRIDE)       # 10
S4_SIZE = window_count(C3_SIZE, POOL_SIZE, POOL_STRIDE)         # 5
C5_SIZE = window_count(S4_SIZE, KERNEL_SIZE, CONV_STRIDE)       # 1
FLAT_SIZE = C5_MAPS * C5_SIZE * C5_SIZE                         # 120


# --- Weight and bias shapes ---
# conv weights: (out maps, in maps, kernel rows, kernel cols)
# conv bias: (out maps, 1, 1)
C1_WEIGHT_SHAPE = (C1_MAPS, INPUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)
C1_BIAS_SHAPE = (C1_MAPS, 1, 1)

C3_WEIGHT_SHAPE = (C3_MAPS, C1_MAPS, KERNEL_SIZE, KERNEL_SIZE)
C3_BIAS_SHAPE = (C3_MAPS, 1, 1)

C5_WEIGHT_SHAPE = (C5_MAPS, C3_MAPS, KERNEL_SIZE, KERNEL_SIZE)
C5_BIAS_SHAPE = (C5_MAPS, 1, 1)

# same as ffnet
# dense weights: (outputs, inputs)
# dense bias: (outputs, )
F6_WEIGHT_SHAPE = (F6_UNITS, FLAT_SIZE)
F6_BIAS_SHAPE = (F6_UNITS, )

OUTPUT_WEIGHT_SHAPE = (OUTPUT_UNITS, F6_UNITS)
OUTPUT_BIAS_SHAPE = (OUTPUT_UNITS, )