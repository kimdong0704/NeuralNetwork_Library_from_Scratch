from data import XOR_INPUTS, XOR_TARGETS
from ai.runner import run_model

if __name__ == "__main__":
    run_model(
        gate="xor",
        inputs=XOR_INPUTS,
        targets=XOR_TARGETS,
        train_method="error",
    )
