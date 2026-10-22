"""
    java.util.Random in numpy, so PerturbedMNIST's shifted and noisy images match the Java ones exactly.
	One generator per seed; all generators step together, so a whole test set is perturbed at once.
"""
import numpy as np

# java.util.Random's 48-bit linear congruential generator
MULTIPLIER = np.uint64(0x5DEECE66D)
ADDEND = np.uint64(0xB)
MASK = np.uint64((1 << 48) - 1)
STATE_BITS = 48

# nextDouble: 53 random bits scaled to [0, 1[
DOUBLE_UNIT = 2.0 ** -53

# fdlibm's log (Java's StrictMath.log); np.log can differ from it in the last bit
LN2_HI = 6.93147180369123816490e-01
LN2_LO = 1.90821492927058770002e-10
LG1 = 6.666666666666735130e-01
LG2 = 3.999999999940941908e-01
LG3 = 2.857142874366239149e-01
LG4 = 2.222219843214978396e-01
LG5 = 1.818357216161805012e-01
LG6 = 1.531383769920937332e-01
LG7 = 1.479819860511658591e-01


def strict_log(x: np.ndarray) -> np.ndarray:
	"""fdlibm's __ieee754_log, line by line, for positive normal x (nextGaussian's s is always one)."""
	bits = np.asarray(x, dtype=np.float64).view(np.uint64)
	hx = (bits >> np.uint64(32)).astype(np.int64)
	lo = bits & np.uint64(0xFFFFFFFF)

	k = (hx >> 20) - 1023
	hx = hx & 0x000FFFFF
	i = (hx + 0x95F64) & 0x100000
	# normalize x or x/2 into [sqrt(2)/2, sqrt(2)[
	x = ((((hx | (i ^ 0x3FF00000)).astype(np.uint64)) << np.uint64(32)) | lo).view(np.float64)
	k = k + (i >> 20)
	f = x - 1.0
	dk = k.astype(np.float64)

	# |f| < 2^-20
	tiny_r = f * f * (0.5 - 0.33333333333333333 * f)
	tiny = np.where(k == 0, f - tiny_r, dk * LN2_HI - ((tiny_r - dk * LN2_LO) - f))
	tiny = np.where(f == 0, dk * LN2_HI + dk * LN2_LO, tiny)

	s = f / (2.0 + f)
	z = s * s
	w = z * z
	t1 = w * (LG2 + w * (LG4 + w * LG6))
	t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)))
	r = t2 + t1
	hfsq = 0.5 * f * f
	with_hfsq = np.where(k == 0, f - (hfsq - s * (hfsq + r)), dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f))
	without_hfsq = np.where(k == 0, f - s * (f - r), dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f))
	general = np.where(((hx - 0x6147A) | (0x6B851 - hx)) > 0, with_hfsq, without_hfsq)

	return np.where((0x000FFFFF & (2 + hx)) < 3, tiny, general)


class JavaRandom:
	def __init__(self, seeds: np.ndarray | int):
		# same as new Random(seed) for each seed
		self.state = (np.atleast_1d(np.asarray(seeds, dtype=np.uint64)) ^ MULTIPLIER) & MASK

	def next_bits(self, bits: int) -> np.ndarray:
		# uint64 products wrap modulo 2^64, which keeps the low 48 bits correct
		self.state = (self.state * MULTIPLIER + ADDEND) & MASK
		return (self.state >> np.uint64(STATE_BITS - bits)).astype(np.int64)

	def next_boolean(self) -> np.ndarray:
		return self.next_bits(1) != 0

	def next_double(self) -> np.ndarray:
		return ((self.next_bits(26) << 27) + self.next_bits(27)) * DOUBLE_UNIT

	def next_gaussians(self, count: int) -> np.ndarray:
		"""The first `count` nextGaussian() values of each generator, shape (generators, count)."""
		generators = len(self.state)
		values = np.empty((generators, count))
		filled = np.zeros(generators, dtype=np.int64)

		# Java's polar method: draw (v1, v2) until 0 < s < 1, then return v1 * m and keep v2 * m for the next call.
		# Every generator draws on every round; draws after a generator is full are never used.
		while np.any(filled < count):
			v1 = 2 * self.next_double() - 1
			v2 = 2 * self.next_double() - 1
			s = v1 * v1 + v2 * v2

			accepted = np.flatnonzero((s < 1) & (s != 0) & (filled < count))
			s = s[accepted]
			multiplier = np.sqrt(-2 * strict_log(s) / s)
			column = filled[accepted]

			values[accepted, column] = v1[accepted] * multiplier
			second = column + 1 < count
			values[accepted[second], column[second] + 1] = v2[accepted[second]] * multiplier[second]

			filled[accepted] += 2

		return values
