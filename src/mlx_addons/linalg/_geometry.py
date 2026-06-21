"""Small rigid-alignment helpers built on MLX primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class KabschRMSDResult:
    """Paired rigid-alignment result."""

    rmsd: mx.array
    aligned: mx.array
    rotations: mx.array


def kabsch_rmsd(
    probe_coords: Any,
    reference_coords: Any,
    *,
    weights: Any | None = None,
    power_iters: int = 48,
    eps: float = 1.0e-8,
) -> KabschRMSDResult:
    """Align paired conformers with Horn/Kabsch and return RMSD.

    ``probe_coords`` and ``reference_coords`` may be ``(N, 3)`` or
    ``(B, N, 3)``. The implementation uses Horn's quaternion formulation and a
    fixed-iteration 4x4 power solve, so it avoids calling MLX CPU-only
    ``svd``/``eigh`` in the hot path.
    """

    probe = _as_float_array(probe_coords)
    reference = _as_float_array(reference_coords)
    squeezed = False
    if probe.ndim == 2:
        probe = probe[None, :, :]
        squeezed = True
    if reference.ndim == 2:
        reference = reference[None, :, :]
    if probe.ndim != 3 or reference.ndim != 3 or probe.shape[-1] != 3 or reference.shape[-1] != 3:
        raise ValueError("coords must have shape (N, 3) or (B, N, 3)")
    if probe.shape != reference.shape:
        raise ValueError("probe_coords and reference_coords must have identical shape")

    n_atoms = int(probe.shape[1])
    if weights is None:
        atom_weights = mx.ones((n_atoms,), dtype=probe.dtype)
    else:
        atom_weights = _as_float_array(weights)
        if atom_weights.shape != (n_atoms,):
            raise ValueError("weights must have shape (N,)")
    atom_weights = atom_weights / mx.maximum(mx.sum(atom_weights), mx.array(float(eps), dtype=probe.dtype))

    probe_centroid = mx.sum(probe * atom_weights[None, :, None], axis=1)
    ref_centroid = mx.sum(reference * atom_weights[None, :, None], axis=1)
    probe_centered = probe - probe_centroid[:, None, :]
    ref_centered = reference - ref_centroid[:, None, :]

    weighted_probe = probe_centered * atom_weights[None, :, None]
    covariance = mx.matmul(mx.swapaxes(ref_centered, -1, -2), weighted_probe)
    horn_key = _horn_key_matrix(covariance)
    quat = _symmetric4_top_eigenvector_power(horn_key, n_iters=power_iters, eps=eps)
    rotations = _quaternion_to_row_rotation(quat)

    aligned = mx.matmul(probe_centered, rotations) + ref_centroid[:, None, :]
    diff = aligned - reference
    rmsd = mx.sqrt(mx.sum(mx.sum(diff * diff, axis=-1) * atom_weights[None, :], axis=-1))
    if squeezed:
        return KabschRMSDResult(rmsd=rmsd[0], aligned=aligned[0], rotations=rotations[0])
    return KabschRMSDResult(rmsd=rmsd, aligned=aligned, rotations=rotations)


def _as_float_array(x: Any) -> mx.array:
    arr = x if isinstance(x, mx.array) else mx.array(x)
    return arr.astype(mx.float32) if arr.dtype != mx.float32 else arr


def _horn_key_matrix(covariance: mx.array) -> mx.array:
    sxx = covariance[..., 0, 0]
    sxy = covariance[..., 0, 1]
    sxz = covariance[..., 0, 2]
    syx = covariance[..., 1, 0]
    syy = covariance[..., 1, 1]
    syz = covariance[..., 1, 2]
    szx = covariance[..., 2, 0]
    szy = covariance[..., 2, 1]
    szz = covariance[..., 2, 2]

    row0 = mx.stack([sxx + syy + szz, syz - szy, szx - sxz, sxy - syx], axis=-1)
    row1 = mx.stack([syz - szy, sxx - syy - szz, sxy + syx, szx + sxz], axis=-1)
    row2 = mx.stack([szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy], axis=-1)
    row3 = mx.stack([sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz], axis=-1)
    return mx.stack([row0, row1, row2, row3], axis=-2)


def _symmetric4_top_eigenvector_power(
    matrix: mx.array,
    *,
    n_iters: int,
    eps: float,
) -> mx.array:
    mat = _as_float_array(matrix)
    starts = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=mat.dtype,
    )
    starts = starts / mx.sqrt(mx.sum(starts * starts, axis=-1, keepdims=True))
    q = mx.broadcast_to(starts, mat.shape[:-2] + starts.shape)

    eye = mx.eye(4, dtype=mat.dtype)
    shift = mx.max(mx.sum(mx.abs(mat), axis=-1), axis=-1) + mx.array(1.0, dtype=mat.dtype)
    shifted = mat + shift[..., None, None] * eye

    for _ in range(int(n_iters)):
        q = mx.matmul(shifted[..., None, :, :], q[..., :, None])[..., 0]
        q = q / mx.sqrt(mx.maximum(mx.sum(q * q, axis=-1, keepdims=True), mx.array(float(eps), dtype=mat.dtype)))

    kq = mx.matmul(mat[..., None, :, :], q[..., :, None])[..., 0]
    rayleigh = mx.sum(q * kq, axis=-1)
    best = mx.argmax(rayleigh, axis=-1)
    selector = (mx.arange(starts.shape[0]) == best[..., None]).astype(mat.dtype)
    out = mx.sum(q * selector[..., None], axis=-2)
    return out / mx.sqrt(mx.maximum(mx.sum(out * out, axis=-1, keepdims=True), mx.array(float(eps), dtype=mat.dtype)))


def _quaternion_to_row_rotation(quaternion: mx.array) -> mx.array:
    q = _as_float_array(quaternion)
    q = q / mx.sqrt(mx.maximum(mx.sum(q * q, axis=-1, keepdims=True), mx.array(1.0e-12, dtype=q.dtype)))
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    row0 = mx.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], axis=-1)
    row1 = mx.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], axis=-1)
    row2 = mx.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)
