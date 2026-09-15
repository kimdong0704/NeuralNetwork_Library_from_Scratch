from data import AND_INPUTS, AND_TARGETS
from ai.runner import run_model

if __name__ == "__main__":
    run_model(
        gate="and",
        inputs=AND_INPUTS,
        targets=AND_TARGETS,
        train_method="error",
    )
