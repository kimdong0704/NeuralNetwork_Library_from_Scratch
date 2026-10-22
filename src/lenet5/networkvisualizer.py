"""
    Python port of NetworkVisualizer.java: shows every layer of LeNet5 for one input image.
	The Swing window becomes a matplotlib figure with one row per layer.
	CONVERTED FROM JAVA TO PYTHON W/ AI ASSISTANCE
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

# run directly (e.g. VS Code's Run button): put src/ on the path so the packages import
if not __package__:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lenet5.lenet5 import LeNet5
from lenet5.model_config import INPUT_SIZE
from lenet5.padded_input import forward

# the Java window is 1200 x 900 pixels and scrolls; the figure shows every row at once
FIGURE_SIZE = (12, 13)
# Java panel heights: feature maps 150 (the single input image 200), bar charts 120, then the prediction label
MAP_ROW_HEIGHT = 150
INPUT_ROW_HEIGHT = 200
BAR_ROW_HEIGHT = 120
LABEL_ROW_HEIGHT = 40

# bar colours run from blue (hue 0.55) for small values to red (hue 0) for the largest
BAR_HUE = 0.55
BAR_SATURATION = 0.7
BAR_BRIGHTNESS = 0.9
BORDER_COLOR = "gray"

# backends that only write files: like Java's GraphicsEnvironment.isHeadless()
NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def has_display() -> bool:
	return plt.get_backend().lower() not in NON_INTERACTIVE_BACKENDS


def normalize(data: np.ndarray) -> np.ndarray:
	# each map scaled on its own: min -> black, max -> white
	low, high = data.min(), data.max()
	value_range = high - low
	if value_range == 0:
		value_range = 1
	return (data - low) / value_range


def bar_colors(scaled: np.ndarray) -> np.ndarray:
	# Java's Color.getHSBColor keeps only the fractional part of the hue
	hue = np.mod(BAR_HUE - scaled * BAR_HUE, 1.0)
	hsv = np.stack([hue, np.full_like(hue, BAR_SATURATION), np.full_like(hue, BAR_BRIGHTNESS)], axis=-1)
	return hsv_to_rgb(hsv)


class NetworkVisualizer:
	def __init__(self, network: LeNet5):
		self.network = network
		self.figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
		# only a window has a title bar (not a notebook or a file backend)
		if self.figure.canvas.manager is not None:
			self.figure.canvas.manager.set_window_title("LeNet-5 Network Visualization")

	def update_visualization(self) -> None:
		"""Draws the layer outputs of the network's last forward pass."""
		net = self.network
		if not hasattr(net.output_layer, "last_output"):
			raise ValueError("run a forward pass before visualizing the network")

		# C1's input is the padded 32x32 image, whichever forward ran
		maps = [
			("Input (32x32)", net.c1_layer.last_inputs[0]),
			("Conv1 Output (6x28x28)", net.c1_layer.last_outputs[0]),
			("Pool1 Output (6x14x14)", net.s2_layer.last_outputs[0]),
			("Conv2 Output (16x10x10)", net.c3_layer.last_outputs[0]),
			("Pool2 Output (16x5x5)", net.s4_layer.last_outputs[0]),
		]
		bars = [
			("FC1 Output (120 neurons)", net.c5_layer.last_outputs[0].reshape(-1)),
			("FC2 Output (84 neurons)", net.f6_layer.last_output[0]),
			("Output Probabilities", net.output_layer.last_output[0]),
		]

		self.figure.clear()
		heights = [INPUT_ROW_HEIGHT] + [MAP_ROW_HEIGHT] * (len(maps) - 1) + [BAR_ROW_HEIGHT] * len(bars) + [LABEL_ROW_HEIGHT]
		rows = self.figure.subfigures(len(heights), 1, height_ratios=heights)

		for row, (title, feature_maps) in zip(rows, maps):
			self._draw_feature_maps(row, title, feature_maps)
		for row, (title, values) in zip(rows[len(maps):], bars):
			self._draw_bars(row, title, values)

		output = net.output_layer.last_output[0]
		prediction = int(np.argmax(output))
		rows[-1].text(0.5, 0.5, f"Prediction: {prediction} ({output[prediction] * 100:.2f}%)",
			ha="center", va="center", fontsize=18, fontweight="bold")

	def show(self) -> None:
		plt.show()

	def process_and_display(self, image32: np.ndarray) -> None:
		"""Runs a forward pass on one padded 32x32 image and shows its layers."""
		forward(self.network, image32)
		self.update_visualization()

	@staticmethod
	def _draw_feature_maps(row, title: str, feature_maps: np.ndarray) -> None:
		# all feature maps of a layer in a single row, as grayscale images
		row.suptitle(title, x=0.01, ha="left", fontsize=10)
		for ax, feature_map in zip(row.subplots(1, len(feature_maps), squeeze=False)[0], feature_maps):
			ax.imshow(normalize(feature_map), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
			ax.set_xticks([])
			ax.set_yticks([])
			for spine in ax.spines.values():
				spine.set_color(BORDER_COLOR)

	@staticmethod
	def _draw_bars(row, title: str, values: np.ndarray) -> None:
		row.suptitle(title, x=0.01, ha="left", fontsize=10)
		ax = row.subplots()

		# bars scaled to the largest value (at least 0, as in Java)
		top = max(float(values.max()), 0.0)
		if top == 0:
			top = 1
		scaled = values / top

		positions = np.arange(len(values))
		ax.bar(positions, scaled, width=1.0, color=bar_colors(scaled), edgecolor=BORDER_COLOR, linewidth=0.5)
		ax.set_xlim(-0.5, len(values) - 0.5)
		ax.set_ylim(0, 1)
		ax.set_yticks([])
		for side in ("top", "right", "left"):
			ax.spines[side].set_visible(False)

		# digit labels only under the output layer
		if len(values) == 10:
			ax.set_xticks(positions)
			ax.tick_params(length=0, labelsize=8)
		else:
			ax.set_xticks([])


def main() -> None:
	# example usage: an untrained network and a ring (a simple 0) as dummy input
	network = LeNet5(seed=1)

	image32 = np.zeros((INPUT_SIZE, INPUT_SIZE))
	for i in range(8, 24):
		for j in range(8, 24):
			di = i - 16
			dj = j - 16
			if 36 < di * di + dj * dj < 64:
				image32[i, j] = 1.0

	viz = NetworkVisualizer(network)
	viz.process_and_display(image32)
	viz.show()


if __name__ == "__main__":
	main()
