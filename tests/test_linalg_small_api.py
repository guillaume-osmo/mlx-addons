import mlx.core as mx
import numpy as np

from mlx_addons.linalg import (
    eigh_small_batch,
    eigh_symmetric_3x3,
    kabsch_rmsd,
    principal_axes_3x3,
)


def _row_rotation(axis: int, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    if axis == 0:
        col = np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == 1:
        col = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    else:
        col = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return col.T


def _kabsch_np(probe, reference, weights=None):
    if weights is None:
        w = np.ones((probe.shape[0],), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1.0e-12)
    pc = np.sum(probe * w[:, None], axis=0)
    rc = np.sum(reference * w[:, None], axis=0)
    p0 = probe - pc[None, :]
    r0 = reference - rc[None, :]
    cov = (p0 * w[:, None]).T @ r0
    u, _, vt = np.linalg.svd(cov)
    fix = np.eye(3)
    fix[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rot = u @ fix @ vt
    return p0 @ rot + rc[None, :]


def test_eigh_small_batch_matches_numpy_reconstruction():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(10, 8, 8)).astype("f4")
    mats = 0.5 * (raw + np.swapaxes(raw, -1, -2))

    values, vectors = eigh_small_batch(mx.array(mats))
    mx.eval(values, vectors)
    values_np = np.asarray(values)
    vectors_np = np.asarray(vectors)

    expected_values = np.linalg.eigvalsh(mats.astype(np.float64))
    assert np.allclose(values_np, expected_values, atol=2.5e-4)

    recon = vectors_np @ (values_np[..., None] * np.swapaxes(vectors_np, -1, -2))
    denom = np.linalg.norm(mats, axis=(-2, -1))
    resid = np.linalg.norm(recon - mats, axis=(-2, -1)) / np.maximum(denom, 1e-12)
    assert np.max(resid) < 5e-4


def test_eigh_symmetric_3x3_and_principal_axes():
    mat = np.asarray(
        [
            [2.0, 0.2, -0.1],
            [0.2, 1.0, 0.3],
            [-0.1, 0.3, 0.5],
        ],
        dtype=np.float32,
    )

    values, vectors = eigh_symmetric_3x3(mx.array(mat))
    desc, axes = principal_axes_3x3(mx.array(mat))
    mx.eval(values, vectors, desc, axes)

    assert values.shape == (3,)
    assert vectors.shape == (3, 3)
    assert np.allclose(np.asarray(values), np.linalg.eigvalsh(mat), atol=1e-5)
    assert np.all(np.diff(np.asarray(desc)) <= 1e-6)
    assert np.linalg.det(np.asarray(axes)) > 0.0


def test_kabsch_rmsd_matches_numpy_for_paired_batches():
    base = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [1.42, 0.13, -0.18],
            [-0.38, 1.21, 0.32],
            [0.23, -0.44, 1.71],
            [1.91, 0.78, 0.65],
        ],
        dtype=np.float32,
    )
    weights = np.asarray([1.0, 1.0, 1.0, 1.0, 0.2], dtype=np.float32)
    rotations = [
        _row_rotation(0, 31.0) @ _row_rotation(2, -17.0),
        _row_rotation(1, -42.0) @ _row_rotation(2, 11.0),
    ]
    translations = [
        np.asarray([3.0, -1.0, 0.5], dtype=np.float32),
        np.asarray([-2.0, 0.7, 1.2], dtype=np.float32),
    ]
    probes = np.stack([base @ rot + shift for rot, shift in zip(rotations, translations, strict=True)]).astype("f4")
    refs = np.stack([base, base @ _row_rotation(2, 12.0) + np.asarray([0.1, -0.2, 0.3], dtype=np.float32)]).astype("f4")

    result = kabsch_rmsd(mx.array(probes), mx.array(refs), weights=mx.array(weights))
    mx.eval(result.rmsd, result.aligned, result.rotations)

    aligned_np = np.asarray(result.aligned)
    rmsd_np = np.asarray(result.rmsd)
    for i in range(probes.shape[0]):
        expected = _kabsch_np(probes[i], refs[i], weights)
        diff2 = np.sum((expected - refs[i]) ** 2, axis=1)
        expected_rmsd = np.sqrt(np.sum(diff2 * weights) / np.sum(weights))
        assert np.allclose(aligned_np[i], expected, atol=2.0e-4)
        assert np.isclose(rmsd_np[i], expected_rmsd, atol=2.0e-4)
