import numpy as np
from numpy.lib.stride_tricks import as_strided

from numpy_utils.config import DTYPE

from .config import CHANNEL_AXIS, ROW_AXIS, COL_AXIS

def output_size(
	inputs: np.ndarray,
	window_rows: int,
	window_cols: int,
	stride: int
) -> tuple[int, int]:
	in_r, in_c = inputs.shape[ROW_AXIS], inputs.shape[COL_AXIS]
	
	if window_rows > in_r or window_cols > in_c:
		raise ValueError(f"window {(window_rows, window_cols)} is larger than the input {(in_r, in_c)}")

	out_r = (in_r - window_rows) // stride + 1
	out_c = (in_c - window_cols) // stride + 1

	return out_r, out_c

def windows_fast(
	inputs: np.ndarray,
	window_rows: int,
	window_cols: int,
	stride: int
) -> np.ndarray:
	in_channels = inputs.shape[CHANNEL_AXIS]
	
	out_rows, out_cols = output_size(inputs, window_rows, window_cols, stride)
	
	channel_step = inputs.strides[CHANNEL_AXIS]
	r_step = inputs.strides[ROW_AXIS]
	c_step = inputs.strides[COL_AXIS]

	return as_strided(
		inputs,
		shape=(in_channels, window_rows, window_cols, out_rows, out_cols),
		strides=(channel_step, r_step, c_step, stride * r_step, stride * c_step),
		writeable=False,
	)

def windows_loop(
	inputs: np.ndarray,
	window_rows: int,
	window_cols: int,
	stride: int
) -> np.ndarray:
	# [:, :, :, row, col]
	# (in_ch, win_r, win_c) --> weight matmul || max over win_r, win_c
	in_channels = inputs.shape[CHANNEL_AXIS]
	out_rows, out_cols = output_size(inputs, window_rows, window_cols, stride)

	windows = np.empty((in_channels, window_rows, window_cols, out_rows, out_cols), dtype=DTYPE)

	for r in range(window_rows):
		for c in range(window_cols):
			windows[:, r, c] = inputs[:, r:r + stride * out_rows:stride, c:c + stride * out_cols:stride]

	return windows