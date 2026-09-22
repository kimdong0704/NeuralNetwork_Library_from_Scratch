# Threshold of the STEP activation function that determines x >= t --> 1, otherwise 0
STEP_THRESHOLD = 0.5
# Magnitude of the how weight and bias changes
LEARNING_RATE = 0.00001
# if prediction is within the error threshold (compared by target - prediction), then it model has successfully predicted
ERROR_THRESHOLD = 0.00001
# maximum epoch runs for training by error
MAX_EPOCHS = 5000
# whether the training loop prints an epoch report at all
REPORT_EACH_EPOCH = True
# an epoch report is printed every this many epochs
REPORT_INTERVAL = 100