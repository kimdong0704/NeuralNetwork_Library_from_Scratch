from ai.data import XOR_INPUTS, XOR_TARGETS
from ai.gate_runner import run_gate

if __name__ == "__main__":
    run_gate(
        gate="xor",
        inputs=XOR_INPUTS,
        targets=XOR_TARGETS,
        train_method="error",
        activation_function="sigmoid",
        learning_rate=0.5,
        gradient_descent=True
    )
