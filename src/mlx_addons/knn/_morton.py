"""Z-order (Morton) encoding and sorting — float-bit preserving.

Uses IEEE 754 float bit reinterpretation to produce Morton codes that
exactly preserve the natural float ordering, without any normalization
or quantization.  This matches the approach used in jztree's CUDA
``PosZorderSort`` kernel.

Supports 2-10 dimensions.
"""

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
#  Float-bit -> sortable integer conversion
# ---------------------------------------------------------------------------

def float_to_sortable_uint(x: np.ndarray) -> np.ndarray:
    """Convert float32 to uint32 preserving total sort order.

    Positive floats already sort correctly as unsigned integers because
    IEEE 754 encodes (exponent, mantissa) in MSB-first order.
    Negative floats sort in reverse, so we flip all their bits.
    We also flip the sign bit of positive floats so that all positives
    sort *after* all (flipped) negatives.

    Args:
        x: float32 numpy array.

    Returns:
        uint32 numpy array with the same sort order as *x*.
    """
    bits = x.view(np.uint32)
    negative = (bits & np.uint32(0x80000000)) != 0
    return np.where(negative, ~bits, bits ^ np.uint32(0x80000000))


def sortable_uint_to_float(u: np.ndarray) -> np.ndarray:
    """Inverse of ``float_to_sortable_uint``."""
    has_sign = (u & np.uint32(0x80000000)) == 0  # was originally negative
    bits = np.where(has_sign, ~u, u ^ np.uint32(0x80000000))
    return bits.view(np.float32)


# ---------------------------------------------------------------------------
#  Bit interleaving (generic dim)
# ---------------------------------------------------------------------------

def _spread_bits_3d(x: np.ndarray) -> np.ndarray:
    """Spread 21-bit integer so bits are spaced 3 apart (optimised 3-D path)."""
    x = x.astype(np.uint64)
    x = (x | (x << 32)) & np.uint64(0x001F00000000FFFF)
    x = (x | (x << 16)) & np.uint64(0x001F0000FF0000FF)
    x = (x | (x <<  8)) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x <<  4)) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x <<  2)) & np.uint64(0x1249249249249249)
    return x


def interleave_bits(coords: np.ndarray, bits_per_dim: int) -> np.ndarray:
    """Interleave bits from *dim* coordinates into a single uint64 Morton code.

    Args:
        coords: (N, dim) uint32 — only the top *bits_per_dim* bits matter.
        bits_per_dim: number of significant bits per coordinate.

    Returns:
        (N,) uint64 Morton codes.
    """
    n, dim = coords.shape
    total_bits = bits_per_dim * dim
    assert total_bits <= 64, f"Need {total_bits} bits but only 64 available"

    # Fast path for 3-D with 21 bits (most common case)
    if dim == 3 and bits_per_dim == 21:
        c = coords >> (32 - bits_per_dim)  # keep top bits
        return (_spread_bits_3d(c[:, 0]) << 2) | (_spread_bits_3d(c[:, 1]) << 1) | _spread_bits_3d(c[:, 2])

    # Generic path — loop over bits (still vectorised over N)
    c = coords >> (32 - bits_per_dim)  # keep top bits_per_dim bits
    codes = np.zeros(n, dtype=np.uint64)
    for bit in range(bits_per_dim):
        for d in range(dim):
            bit_val = ((c[:, d] >> np.uint32(bits_per_dim - 1 - bit)) & np.uint32(1)).astype(np.uint64)
            codes |= bit_val << np.uint64(total_bits - 1 - (bit * dim + d))
    return codes


# ---------------------------------------------------------------------------
#  Morton encoding (float-bit preserving)
# ---------------------------------------------------------------------------

def morton_encode(pos: np.ndarray, bits_per_dim: int | None = None) -> np.ndarray:
    """Encode float32 positions into Morton codes using float-bit preservation.

    Each coordinate is reinterpreted as a sortable unsigned integer (IEEE 754
    sign-flip trick), then the top *bits_per_dim* bits are interleaved across
    dimensions to form a uint64 Morton code.

    Args:
        pos: (N, dim) float32 positions.  Supports dim 2-10.
        bits_per_dim: bits per coordinate.  Defaults to floor(63/dim) so the
            code fits in 63 bits of int64.

    Returns:
        (N,) uint64 Morton codes.
    """
    assert pos.dtype == np.float32, "Only float32 supported"
    n, dim = pos.shape
    assert 2 <= dim <= 10, f"Dimension {dim} out of range [2,10]"

    if bits_per_dim is None:
        bits_per_dim = 63 // dim  # e.g. 21 for 3-D, 31 for 2-D

    # Float bits -> sortable unsigned integers (per coordinate)
    sortable = float_to_sortable_uint(pos)  # (N, dim) uint32

    # Interleave
    codes = interleave_bits(sortable, bits_per_dim)
    return codes


# ---------------------------------------------------------------------------
#  Z-order sorting
# ---------------------------------------------------------------------------

def zsort(pos):
    """Sort positions into Z-order (Morton order).

    Args:
        pos: (N, dim) mlx.array *or* numpy float32 positions.

    Returns:
        (pos_sorted_mx, indices_mx, codes_np):
            pos_sorted_mx, indices_mx are mlx arrays.
            codes_np is a numpy uint64 array (kept in numpy for tree building).
    """
    if isinstance(pos, mx.array):
        mx.eval(pos)
        pos_np = np.array(pos).astype(np.float32)
    else:
        pos_np = np.asarray(pos, dtype=np.float32)

    codes = morton_encode(pos_np)

    # Sort by Morton code (uint64 — use int64 view for argsort)
    indices_np = np.argsort(codes.view(np.int64), kind="stable")

    pos_sorted_np = pos_np[indices_np]
    codes_sorted  = codes[indices_np]

    return mx.array(pos_sorted_np), mx.array(indices_np.astype(np.int32)), codes_sorted
