from ai.data import AND_INPUTS, AND_TARGETS
from ai.gate_runner import run_gate

if __name__ == "__main__":
    run_gate(
        gate="and",
        inputs=AND_INPUTS,
        targets=AND_TARGETS,
        train_method="error",
        activation_function="sigmoid",
        learning_rate=0.5,
        gradient_descent=True
    )
