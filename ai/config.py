# Threshold of the STEP activation function that determines x >= t --> 1, otherwise 0
STEP_THRESHOLD = 0.5
# If it computes with gradient descent (Does not work with activation functions that can not be derived)
GRADIENT_DESCENT = False
# If weights is none --> sets to 0, if weights is "random" --> sets to random weights
WEIGHTS = None
# Bias Settings
BIAS = 0.0
# Magnitude of the how weight and bias changes
LEARNING_RATE = 0.5
# if prediction is within the error threshold (compared by target - prediction), then it model has successfully predicted
ERROR_THRESHOLD = 0.1
# maximum epoch runs for training by error
MAX_EPOCHS = 500
# epochs ran for training by epoch
EPOCHS = 20
