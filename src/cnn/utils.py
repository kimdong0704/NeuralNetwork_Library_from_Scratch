import numpy as np
from numpy.lib.stride_tricks import as_strided

from numpy_utils.config import DTYPE

from .config import BATCH_AXIS, CHANNEL_AXIS, ROW_AXIS, COL_AXIS, IMAGE_BATCH_NDIM

def check_image_batch(inputs: np.ndarray) -> None:
	if inputs.ndim != IMAGE_BATCH_NDIM:
		raise ValueError(f"inputs must be (batch, channels, rows, cols), got shape {inputs.shape}")

def window_count(size: int, window: int, stride: int) -> int:
	return ((size - window) // stride) + 1

def output_size(
	inputs: np.ndarray,
	window_rows: int,
	window_cols: int,
	stride: int
) -> tuple[int, int]:
	in_r, in_c = inputs.shape[ROW_AXIS], inputs.shape[COL_AXIS]
	
	if window_rows > in_r or window_cols > in_c:
		raise ValueError(f"window {(window_rows, window_cols)} is larger than the input {(in_r, in_c)}")

	out_r = window_count(in_r, window_rows, stride)
	out_c = window_count(in_c, window_cols, stride)

	return out_r, out_c

def windows_fast(
	inputs: np.ndarray,
	window_rows: int,
	window_cols: int,
	stride: int
) -> np.ndarray:
	batch = inputs.shape[BATCH_AXIS]
	in_channels = inputs.shape[CHANNEL_AXIS]
	
	out_rows, out_cols = output_size(inputs, window_rows, window_cols, stride)
	
	batch_step = inputs.strides[BATCH_AXIS]
	channel_step = inputs.strides[CHANNEL_AXIS]
	r_step = inputs.strides[ROW_AXIS]
	c_step = inputs.strides[COL_AXIS]

	return as_strided(
		inputs,
		shape=(batch, in_channels, window_rows, window_cols, out_rows, out_cols),
		strides=(batch_step, channel_step, r_step, c_step, stride * r_step, stride * c_step),
		writeable=False,
	)

def windows_loop(
	inputs: np.ndarray,
	window_rows: int,
	window_cols: int,
	stride: int
) -> np.ndarray:
	# [:, :, :, :, row, col]
	# (batch, in_ch, win_r, win_c) --> weight matmul || max over win_r, win_c
	batch = inputs.shape[BATCH_AXIS]
	in_channels = inputs.shape[CHANNEL_AXIS]
	out_rows, out_cols = output_size(inputs, window_rows, window_cols, stride)

	windows = np.empty((batch, in_channels, window_rows, window_cols, out_rows, out_cols), dtype=DTYPE)

	for r in range(window_rows):
		for c in range(window_cols):
			windows[:, :, r, c] = inputs[:, :, r:r + stride * out_rows:stride, c:c + stride * out_cols:stride]

	return windows

def fold_windows(
	window_grads: np.ndarray,
	input_shape: tuple[int, ...],
	stride: int
) -> np.ndarray:
	# reverse of windows_loop: window_grads (batch, in_ch, win_r, win_c, out_r, out_c) -> input_shape
	# sum of window gradients by input pixels
	_, _, window_rows, window_cols, out_rows, out_cols = window_grads.shape

	input_grads = np.zeros(input_shape, dtype=DTYPE)

	for r in range(window_rows):
		for c in range(window_cols):
			input_grads[:, :, r:r + stride * out_rows:stride, c:c + stride * out_cols:stride] += window_grads[:, :, r, c]

	return input_grads
