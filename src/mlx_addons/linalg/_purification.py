# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Density-matrix purification — eigendecomposition-free construction of the
one-electron density matrix from a Fock-like operator.

Two iterations:

- :func:`mcweeny_purify` — McWeeny's canonical iteration
  ``rho_{n+1} = 3 rho_n^2 - 2 rho_n^3``. Initial guess via Palser-Manolopoulos
  shift-and-scale so the spectrum starts in [0, 1] with trace ``n_occ``.
  Quadratic convergence near the {0, 1} fixed points. Stalls in the presence
  of fractional / near-degenerate occupations.

- :func:`sp2_purify` — Niklasson's SP2 with TC2 trace-correcting branches.
  Iteration: ``rho <- rho^2`` if ``Tr(rho) > n_occ``, else ``rho <- 2 rho - rho^2``.
  More robust when the gap is small; standard fallback when McWeeny stalls.

Both return the natural one-electron density ``rho`` with eigenvalues in [0, 1]
and ``Tr(rho) = n_occ``. For closed-shell SCF the consumer multiplies by 2 to
get the conventional electron-counting density ``P = 2 * rho``.

References:

- R. McWeeny, *Rev. Mod. Phys.* 32, 335 (1960).
- A. H. R. Palser, D. E. Manolopoulos, *Phys. Rev. B* 58, 12704 (1998).
- A. M. N. Niklasson, *Phys. Rev. B* 66, 155115 (2002) — arXiv:cond-mat/0209502.
"""

from __future__ import annotations

import mlx.core as mx

from ._eig import gershgorin_bounds


def _frob_idem_err(rho: mx.array) -> mx.array:
    """Frobenius idempotency residual ``||rho^2 - rho||_F`` per batch element.

    Returns an array of shape ``(...,)`` matching the leading dims of ``rho``.
    """
    diff = rho @ rho - rho
    return mx.sqrt(mx.sum(diff * diff, axis=(-2, -1)))


def _broadcast_scalar(s, target_shape, dtype):
    """Bring a scalar / batched scalar up to ``(..., 1, 1)`` for matrix ops."""
    if not isinstance(s, mx.array):
        s = mx.array(s, dtype=dtype)
    else:
        s = s.astype(dtype)
    # Broadcast to leading batch dims of target, then add (1, 1) trailing.
    while s.ndim < len(target_shape) - 2:
        s = mx.expand_dims(s, axis=-1)
    return s[..., None, None]


def _tight_spectral_bounds(F: mx.array) -> tuple[mx.array, mx.array]:
    """Tight (lo, hi) spectral bracket via power iteration on the squared
    centered matrix.

    Gershgorin is provably safe but typically 2-3x wider than the true
    spectrum on dense matrices, which compresses the SP2 / McWeeny initial
    guess so heavily that all eigenvalues collapse below the unstable fixed
    point at 0.5 and the iteration cannot converge.

    Strategy: estimate the spectral radius of the centered matrix
    ``Fc = F - mu I`` via power iteration on ``Fc^2``, then pad by a 10%
    safety margin so the bracket strictly contains the spectrum even with
    only a few power iterations. 24 power-iter steps on ``Fc^2`` is plenty
    for symmetric matrices in our target dim range (N=5..2000); the cost is
    negligible compared to the SP2 inner loop.

    Args:
        F: (..., N, N) symmetric matrix.

    Returns:
        ``(lo, hi)`` arrays of shape ``(...,)``.
    """
    dtype = F.dtype
    *batch, N, _ = F.shape

    mu = mx.trace(F, axis1=-2, axis2=-1) / N
    eye = mx.eye(N, dtype=dtype)
    mu_b = _broadcast_scalar(mu, F.shape, dtype)
    Fc = F - mu_b * eye
    v = mx.random.normal(shape=(*batch, N, 1)).astype(dtype)
    Fc_sq = Fc @ Fc
    eps = mx.array(1e-30, dtype=dtype)
    for _ in range(24):
        v = Fc_sq @ v
        norm = mx.sqrt(mx.sum(v * v, axis=-2, keepdims=True))
        v = v / mx.maximum(norm, eps)
    Fc_v = Fc @ v
    rho_spec = mx.sqrt(mx.sum(Fc_v * Fc_v, axis=-2)).reshape(mu.shape)
    rho_spec = rho_spec * mx.array(1.10, dtype=dtype)    # 10% safety margin
    return mu - rho_spec, mu + rho_spec


def mcweeny_purify(
    F: mx.array,
    rho_init: mx.array,
    *,
    max_iter: int = 50,
    idem_tol: float = 1e-6,
) -> mx.array:
    """McWeeny canonical purification — warm-start refinement only.

    Iterates ``rho <- 3 rho^2 - 2 rho^3``. The scalar map ``f(x) = 3x^2 - 2x^3``
    has stable fixed points at 0 and 1 with ``f'(0) = f'(1) = 0`` (quadratic
    convergence) and an unstable fixed point at 0.5. **McWeeny is not
    trace-correcting**: every eigenvalue independently flows to whichever
    stable fixed point it is closer to, so the converged trace equals the
    count of input eigenvalues above 0.5. This means the caller MUST supply a
    starting density ``rho_init`` whose count of eigenvalues above 0.5 is
    already the desired ``n_occ`` — McWeeny only polishes; it cannot fix the
    count.

    Use this when you already have a near-idempotent density (e.g. a
    converged SCF density from the previous Fock iteration). For cold-start
    cases use :func:`sp2_purify`, which is trace-correcting by construction
    and handles arbitrary occupation fractions.

    Args:
        F: (..., N, N) symmetric Fock-like matrix. Used only as the dtype/
            shape reference; McWeeny does not actually use ``F`` inside the
            iteration (the iteration is purely ``rho`` -> ``3 rho^2 - 2 rho^3``).
        rho_init: ``(..., N, N)`` starting density. Eigenvalues should be in
            [0, 1] with the desired count above 0.5.
        max_iter: maximum refinement iterations.
        idem_tol: stopping threshold on ``||rho^2 - rho||_F``. ``1e-6`` is
            the float32 floor; ``1e-12`` reachable in float64.

    Returns:
        rho: (..., N, N) refined density matrix.
    """
    rho = rho_init.astype(F.dtype) if rho_init.dtype != F.dtype else rho_init

    best_rho = rho
    best_err = float(mx.max(_frob_idem_err(rho)).item())
    drift_count = 0
    for _ in range(max_iter):
        rho_sq = rho @ rho
        rho_cube = rho_sq @ rho
        rho_new = 3.0 * rho_sq - 2.0 * rho_cube
        err = _frob_idem_err(rho_new)
        max_err = float(mx.max(err).item())
        if max_err < best_err:
            best_err = max_err
            best_rho = rho_new
            drift_count = 0
        else:
            drift_count += 1
        rho = rho_new
        if max_err < idem_tol:
            break
        if drift_count >= 2:
            break

    return best_rho


def sp2_purify(
    F: mx.array,
    n_occ,
    *,
    max_iter: int = 100,
    idem_tol: float = 1e-6,
) -> mx.array:
    """Niklasson SP2 / TC2 trace-correcting purification.

    Initial guess
    ``rho_0 = (hi * I - F) / (2 (hi - lo))``
    places the spectrum in [0, 1]. Each iteration takes the trace-correcting
    branch:

    - if ``Tr(rho) > n_occ``: ``rho <- rho^2`` (lowers the trace)
    - else:                  ``rho <- 2 rho - rho^2`` (raises the trace)

    The branch decision is per-batch-element via ``mx.where``, so batches
    with different ``n_occ`` per molecule are handled in one pass.

    Stops on the first iterate that reaches ``||rho^2 - rho||_F < idem_tol``
    or on detected float-precision-floor drift (``err`` increasing two
    iterations in a row). Returns the best-so-far iterate.

    Args:
        F: (..., N, N) symmetric Fock-like matrix.
        n_occ: int, float, or batched ``(...,)`` array.
        max_iter: maximum iterations.
        idem_tol: idempotency stopping threshold. Default ``1e-6`` is the
            practical float32 floor; float64 inputs can reach ``1e-12``.

    Returns:
        rho: (..., N, N) natural density matrix.
    """
    dtype = F.dtype
    *_, N, _ = F.shape

    lo, hi = _tight_spectral_bounds(F)
    n_occ_arr = mx.array(n_occ, dtype=dtype) if not isinstance(n_occ, mx.array) else n_occ.astype(dtype)

    eye = mx.eye(N, dtype=dtype)
    hi_b = _broadcast_scalar(hi, F.shape, dtype)
    lo_b = _broadcast_scalar(lo, F.shape, dtype)
    eps = mx.array(1e-30, dtype=dtype)
    rho = (hi_b * eye - F) / mx.maximum(2.0 * (hi_b - lo_b), eps)

    # SP2 has two phases: an oscillating "dance" while the spectrum is split
    # at 0.5, then quadratic convergence to {0, 1}. After the lock-in point
    # the iteration can drift back up due to float-precision overshoot —
    # track best-so-far and bail when post-lock-in err exceeds 2x best.
    best_rho = rho
    best_err = float("inf")
    lockin_floor = idem_tol * 10.0
    locked_in = False
    for _ in range(max_iter):
        rho_sq = rho @ rho
        tr_rho = mx.trace(rho, axis1=-2, axis2=-1)       # (...,)
        too_high = (tr_rho > n_occ_arr).astype(dtype)
        too_high_b = _broadcast_scalar(too_high, F.shape, dtype)
        rho_new = too_high_b * rho_sq + (1.0 - too_high_b) * (2.0 * rho - rho_sq)
        err = _frob_idem_err(rho_new)
        max_err = float(mx.max(err).item())
        if max_err < best_err:
            best_err = max_err
            best_rho = rho_new
            if best_err < lockin_floor:
                locked_in = True
        rho = rho_new
        if max_err < idem_tol:
            break
        if locked_in and max_err > 2.0 * best_err:
            break

    return best_rho
