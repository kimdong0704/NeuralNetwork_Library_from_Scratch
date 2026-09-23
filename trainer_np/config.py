from .cost import COSTS

# Cost function used to measure prediction error and drive backpropagation
COST = COSTS.MSE

# Magnitude of the how weight and bias changes
LEARNING_RATE = 0.7

# if prediction is within the error threshold (compared by target - prediction), then it model has successfully predicted
ERROR_THRESHOLD = 0.00001

# maximum epoch runs for training by error
MAX_EPOCHS = 500

# number of samples averaged into a single gradient step; 1 reproduces
# per-sample online SGD, larger values trade update granularity for
# throughput via vectorized batch matrix ops
BATCH_SIZE = 32

# whether the training loop prints an epoch report at all
REPORT_EACH_EPOCH = True

# an epoch report is printed every this many epochs
REPORT_INTERVAL = 100
