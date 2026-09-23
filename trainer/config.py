from .cost import COSTS

# Cost function used to measure prediction error and drive backpropagation
COST = COSTS.MSE

# Magnitude of the how weight and bias changes
LEARNING_RATE = 0.7

# if prediction is within the error threshold (compared by target - prediction), then it model has successfully predicted
ERROR_THRESHOLD = 0.00001

# maximum epoch runs for training by error
MAX_EPOCHS = 500

# whether the training loop prints an epoch report at all
REPORT_EACH_EPOCH = True

# an epoch report is printed every this many epochs
REPORT_INTERVAL = 100

# output at/above this is classified as 1 for single-output networks; multi-output networks use the largest output
DECISION_THRESHOLD = 0.5

# number of decimal places shown for errors, weights and predictions in reports
REPORT_DIGITS = 4
