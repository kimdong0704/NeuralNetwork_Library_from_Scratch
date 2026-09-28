# Neural Networks from Scratch

[![tests](https://github.com/kimdong0704/Artificial_Intelligence/actions/workflows/tests.yml/badge.svg)](https://github.com/kimdong0704/Artificial_Intelligence/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-only-013243)

A feed-forward neural network library written **twice**: once node by node in pure Python, and once
fully vectorized with NumPy. The two are tested against each other to give the same results. Four
experiments take the library from a single perceptron that cannot learn XOR to a network that
classifies handwritten digits with **98% test accuracy**.

No PyTorch, TensorFlow or scikit-learn. The forward pass, backpropagation (including the full softmax
Jacobian), mini-batch gradient descent, the cost functions, the training loop and model saving/loading
are all written by hand.

## Highlights

- **Two implementations with one API.** In `loop_nn` every neuron is a `Node` object and every
  multiply-add is a Python loop. In `numpy_nn` a whole layer is one matrix multiply. Swap the import
  and the same training code runs on either.
- **Equal results, checked by tests.** From the same starting weights, both implementations produce
  the same losses, accuracies and final weights to within `1e-9` over complete training runs.
- **Backpropagation checked against numerical gradients** (finite differences) for sigmoid, identity
  and softmax + cross-entropy networks.
- **Over 200× faster when vectorized.** On MNIST-sized layers the NumPy version trains more than
  200 times faster, and gives the same numbers.
- **Four experiments with written analysis**: linear separability, hidden layers, generalization and
  overfitting, hyper-parameter studies, and how a softmax classifier behaves on inputs that are not
  digits.

## Results

| Lab | Problem | Network | Result |
|---|---|---|---|
| [1](lab1/lab1.ipynb) | AND and XOR gates | 2 → 1 (sigmoid) | AND **100%**. XOR **25%**: one node can only draw one line |
| [2](lab2/lab2.ipynb) | XOR gate | 2 → 4 → 1 (sigmoid) | **100%**: a hidden layer makes XOR learnable |
| [3](lab3/lab3.ipynb) | Two moons, 700 train / 300 validation points | 2 → 8 → 1 (sigmoid) | **100%** train, **99.0%** validation |
| [4](lab4/lab4.ipynb) | MNIST, 60,000 train / 10,000 test images | 784 → 300 → 10 (ReLU, softmax) | **98.0%** test (mean of 5 seeds) after 5–6 epochs; **98.5%** best |

<table>
  <tr>
    <td><img src="docs/images/xor_single_node.png" alt="A single node draws one straight line and misclassifies three XOR points"></td>
    <td><img src="docs/images/xor_hidden_layer.png" alt="A hidden layer separates XOR with a band between two lines"></td>
  </tr>
  <tr>
    <td align="center"><b>Lab 1:</b> a single node can only draw one line, so XOR fails</td>
    <td align="center"><b>Lab 2:</b> a hidden layer draws a band, so XOR is solved</td>
  </tr>
</table>

<img src="docs/images/two_moons_boundary.png" alt="Decision boundary between the two moons on the training and validation sets">

**Lab 3:** the learned boundary between the two crescents. The three validation mistakes are circled.

<img src="docs/images/mnist_non_digits.png" width="620" alt="Softmax probabilities of the MNIST network for random noise, a smiley face and an inverted 7">

**Lab 4, Task E:** the MNIST network gives a confident answer even for inputs that are not digits,
including a 99.9% "2" for a smiley face.

## One API, two implementations

```python
from numpy_nn import ACTIVATIONS, COSTS, Layer, Network, random_weights, zero_bias
from numpy_trainer import DataLoader, Trainer

network = Network(
    Layer.build(784, 300, random_weights(784, 300, -0.05, 0.05), zero_bias(300), ACTIVATIONS.RELU),
    Layer.build(300, 10, random_weights(300, 10, -0.05, 0.05), zero_bias(10), ACTIVATIONS.SOFTMAX),
)

trainer = Trainer(network, cost=COSTS.CROSS_ENTROPY, learning_rate=0.3)
trainer.train(
    DataLoader(train_x, train_y, batch_size=32, shuffle=True),
    epochs=20,
    validation=(test_x, test_y),
    print_interval=1,
    stop_when=lambda epoch: epoch.accuracy >= 0.99 and epoch.validation_accuracy >= 0.97,
)

network.save("models/mnist.npz")
```

Replace `numpy_` with `loop_` and the same code runs on plain Python lists.

| | `loop_nn` / `loop_trainer` | `numpy_nn` / `numpy_trainer` |
|---|---|---|
| A layer is | a list of `Node` objects, each with its own weights | one `(outputs × inputs)` weight matrix |
| Forward pass | for each sample → for each node → for each weight | `activation(inputs @ W.T + b)` |
| Weight gradients | each node sums `dz · input` over the batch | `dz.T @ inputs` |
| Data | `list[list[float]]` | `np.ndarray` |
| Saved as | JSON | `.npz` |

[`benchmarks/compare_implementations.py`](benchmarks/compare_implementations.py) trains both on the
same data from the same starting weights:

```text
Two moons  (2 -> 8 -> 1 | batch size: 1 | epochs: 20)
  loop_nn : epoch   20 | loss: 0.0058 | accuracy: 99.43%  |    0.30 s
  numpy_nn: epoch   20 | loss: 0.0058 | accuracy: 99.43%  |    0.47 s
  largest weight difference: 1.5e-14 | numpy_nn speedup: 0.6x

MNIST slice  (784 -> 32 -> 10 | batch size: 32 | epochs: 1)
  loop_nn : epoch    1 | loss: 1.5393 | accuracy: 53.70%  |    7.26 s
  numpy_nn: epoch    1 | loss: 1.5393 | accuracy: 53.70%  |    0.03 s
  largest weight difference: 7.2e-16 | numpy_nn speedup: 216.1x
```

Vectorizing only pays off when there is enough work to vectorize. With a 2-8-1 network and one sample
per batch, NumPy's per-call overhead makes it *slower* than plain loops. With 784 inputs and batches
of 32 it is hundreds of times faster.

## How it works

Every layer computes `y = f(z)` with `z = x Wᵀ + b`, one row per sample. Training runs in two
passes, so every gradient is computed from the weights that produced the output:

1. **`network.backward(dC/dy)`** runs from the last layer to the first. Each layer turns the incoming
   delta `dC/dy` into `dC/dz` through its activation derivative, stores `∇W = dzᵀ x` and `∇b = Σ dz`,
   and passes `dz W` back to the previous layer.
2. **`network.step(learning_rate)`** then updates every layer with `W ← W − η∇W`.

Some details:

- **Softmax uses the full Jacobian**: `dz = y ⊙ (δ − ⟨δ, y⟩)`. The derivative therefore works with
  any cost, and with cross-entropy it reduces to the familiar `(y − t) / n`.
- **Numerical stability**: sigmoid is computed as `½(1 + tanh(x/2))`, so it cannot overflow. Softmax
  subtracts the row maximum first. Cross-entropy clips probabilities at `1e-12` before taking the log.
- **Non-differentiable output layers**: a `STEP` output layer passes deltas straight through, as in
  the perceptron learning rule. The trainer rejects `STEP` in hidden layers, where backpropagation
  needs a derivative.
- **BLAS threading**: mini-batch matrices are small, so the NumPy trainer limits BLAS to 4 threads.
  With the default of one thread per core, each MNIST batch was about 10× slower because the threads
  spent most of their time coordinating.

## Project structure

```text
numpy_nn/             vectorized network: activations, costs, initializers, Layer, Network (save/load .npz)
numpy_trainer/        DataLoader, Trainer, accuracy metrics, Reporter (per-epoch history), plots
loop_nn/              pure-Python network with the same modules, plus Node (a single neuron)
loop_trainer/         the same trainer, written with loops over lists
tests/
  test_equivalence.py loop_nn and numpy_nn agree on forward, backward and whole training runs
  test_gradients.py   backpropagation matches finite-difference gradients
  test_training.py    XOR is learned; save/load round-trips for both implementations
benchmarks/           side-by-side accuracy and speed comparison
lab1/ … lab4/         experiment notebooks (see below)
docs/images/          figures used in this README
```

## The labs

Each lab is a Jupyter notebook that builds its networks with `numpy_nn` and trains them with
`numpy_trainer`.

- **[Lab 1: single-node perceptron](lab1/lab1.ipynb).** One sigmoid node learns AND but cannot learn
  XOR: it gets stuck with every output near 0.5 and three of the four points wrong.
  ([reflection](lab1/CSSE313_Lab1_Reflection.pdf))
- **[Lab 2: hidden-layer perceptron](lab2/lab2.ipynb).** Adding a hidden layer of four units solves
  XOR. The loss curve shows the usual slow start followed by a sharp drop once the hidden units
  specialise. ([reflection](lab2/CSSE313_Lab2_Reflection.pdf))
- **[Lab 3: two moons](lab3/lab3.ipynb).** On 700 training and 300 validation points the network
  fits the training set perfectly and reaches 99.0% on validation. Validation loss is lowest at
  epoch 26 and rises slowly after that, which shows the onset of overfitting.
- **[Lab 4: MNIST](lab4/lab4.ipynb).** A 784-300-10 ReLU/softmax network trained on all 60,000
  images with mini-batch SGD and cross-entropy. ([reflection](lab4/CSSE313_Lab4_Reflection.pdf))
  - *Hyper-parameters*: a sweep over hidden size (25 / 256 / 1024), initial weight range (all
    positive, all negative, or both signs) and learning rate. When all the starting weights share one
    sign, the network never learns in 20 epochs: it stays at about 10% accuracy, no better than guessing.
  - *Input curation*: turning grey pixels into pure black and white reaches the goal faster (5.0 vs
    5.6 epochs) but lowers test accuracy for every one of the 5 seeds (97.64% vs 98.01%).
  - *Training too long*: after 100 epochs training accuracy is 100%, but test loss has been rising
    since epoch 10, while test accuracy stays flat at about 98.5%.
  - *Non-digit inputs*: noise, a smiley face and an inverted 7 are each assigned a digit with 73–99.9%
    confidence. A softmax classifier has no way to say "none of the above".

## Getting started

```bash
git clone https://github.com/kimdong0704/Artificial_Intelligence.git
cd Artificial_Intelligence
pip install -e ".[dev]"

pytest                                      # 21 tests
python benchmarks/compare_implementations.py
jupyter lab                                 # open lab1/ … lab4/
```

The notebooks add the repository root to `sys.path` themselves, so installing only the dependencies
(`pip install numpy matplotlib threadpoolctl jupyter`) is enough to run them. Labs 1–3 each run in
seconds. Lab 4 retrains several MNIST networks and takes a while. Its trained models are included in
[`lab4/models/`](lab4/models), so the Task E cells can load them without retraining.
